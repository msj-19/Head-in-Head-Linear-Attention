# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# from tkinter import N
from typing import Optional

import torch
import triton
import triton.language as tl

from ....ops.utils import prepare_chunk_indices
from ....ops.utils.op import exp, gather
from ....utils import check_shared_mem, is_gather_supported, use_cuda_graph


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=['BK', 'BT', 'K'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def mask_chunk_dplr_bwd_kernel_intra(
    q,
    k,
    a,
    b,
    gi,
    ge,
    mask,
    dAqk,
    dAqb,
    dAak,
    dAab,
    dmask,
    dq,
    dk,
    da,
    db,
    dqg,
    dkg,
    dag,
    dbg,
    dgk,
    dgk_offset,
    cu_seqlens,
    chunk_indices,
    scale: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    r: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    GATHER_SUPPORTED: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = (i_b * T).to(tl.int32), (i_b * T + T).to(tl.int32)

    if i_t * BT >= T:
        return
    ##BT = BC 不需要在意
    # offset calculation
    ge += (bos*H + i_h) * K
    gi += (bos*H + i_h) * K
    q += (bos*H + i_h) * K
    a += (bos*H + i_h) * K
    b += (bos*H + i_h) * K
    k += (bos*H + i_h) * K
    dq += (bos*H + i_h) * K
    dk += (bos*H + i_h) * K
    da += (bos*H + i_h) * K
    db += (bos*H + i_h) * K
    dqg += (bos*H + i_h) * K
    dag += (bos*H + i_h) * K
    dkg += (bos*H + i_h) * K
    dbg += (bos*H + i_h) * K
    dgk += (bos*H + i_h) * K
    dgk_offset += (bos*H + i_h) * K
    dmask += (bos*H + i_h) * r *r
    mask += (bos*H + i_h)  * r *r
    dAqk += (bos*H + i_h) * BT
    dAqb += (bos*H + i_h) * BT *r
    dAak += (bos*H + i_h) * BT *r *r
    dAab += (bos*H + i_h) * BT *r * r
    stride_qk = H*K
    stride_A = H*BT
    p_ge = tl.make_block_ptr(ge, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_gi = tl.make_block_ptr(gi, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    # [BC, BK]
    b_ge = tl.load(p_ge, boundary_check=(0, 1))
    b_gi = tl.load(p_gi, boundary_check=(0, 1))
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    b_da = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    b_db = tl.zeros([BC, BK], dtype=tl.float32)
    # intra chunk gradient calculation
    p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (stride_A, 1), (i_t*BT, 0), (BC, BC), (1, 0))
    p_dAqb = tl.make_block_ptr(dAqb, (T, BT,r), (stride_A*r, r, 1), (i_t*BT, 0,0), (BC, BC,r), (2,1, 0))

    p_dAab = tl.make_block_ptr(dAab, (T, BT,r,r), (stride_A*r*r,r*r,r,1), (i_t*BT,0,0,0), (BC, BC,r,r), (3,2,1, 0))
    p_dAak = tl.make_block_ptr(dAak, (T, BT,r,r), (stride_A*r*r,r*r,r, 1), (i_t*BT, 0,0,0), (BC, BC,r,r), (3,2,1, 0))


    o_i = tl.arange(0, BC)
    p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t*BT, i_k*BK), (BC, BK), (1, 0))
    p_b = tl.make_block_ptr(b, (T, K), (stride_qk, 1), (i_t*BT, i_k*BK), (BC, BK), (1, 0))
    p_a = tl.make_block_ptr(a, (T, K), (stride_qk, 1), (i_t*BT, i_k*BK), (BC, BK), (1, 0))
    p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t*BT, i_k*BK), (BC, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_b = tl.load(p_b, boundary_check=(0, 1))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_a = tl.load(p_a, boundary_check=(0, 1))
    b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
    b_dAqb = tl.load(p_dAqb, boundary_check=(0, 1,2))
    b_dAab = tl.load(p_dAab, boundary_check=(0, 1,2,3))
    b_dAak = tl.load(p_dAak, boundary_check=(0, 1,2,3))

    # inter chunk gradient calculation
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    b_darr = tl.zeros([BC, r,r,BK//r], dtype=tl.float32)

    p_mask = tl.make_block_ptr(mask, (T,r,r),(H*r*r,r,1), (i_t*BT,0,0),(BC,r,r),(2,1,0))
    b_mask = tl.load(p_mask,boundary_check=(0,1,2))#BT,r,r
    b_arr = tl.reshape(b_a,(BC,r,BK//r))[:,None,:,:]*b_mask[:,:,:,None]#BC r r BK//r

    # intra chunk gradient calculation
    for j in range(0, min(BC, T - i_t * BT)):
        # trick to index the block
        # if GATHER_SUPPORTED:
        #     row_idx = tl.full([1, BK], j, dtype=tl.int16)
        #     col_idx = tl.full([BC, 1], j, dtype=tl.int16)
        #     row_idx_bc = tl.full([1, BC], j, dtype=tl.int16)
        #     # [1, BK]
        #     b_kj = gather(b_k, row_idx, axis=0)
        #     b_bj = gather(b_b, row_idx, axis=0)
        #     b_gij = gather(b_gi, row_idx, axis=0)
        #     b_gej = gather(b_ge, row_idx, axis=0)
        #     b_qj = gather(b_q, row_idx, axis=0)
        #     b_aj = gather(b_a, row_idx, axis=0)
        #     # [BC, 1]
        #     b_dAqk_j = gather(b_dAqk, col_idx, axis=1)
        #     b_dAab_j = gather(b_dAab, col_idx, axis=1)
        #     b_dAqb_j = gather(b_dAqb, col_idx, axis=1)
        #     b_dAak_j = gather(b_dAak, col_idx, axis=1)
        #     # [1, BC] -> [BC, 1]
        #     b_dA_qk_j = tl.sum(gather(b_dAqk, row_idx_bc, axis=0), 0)[:, None]
        #     b_dA_qk_j = tl.sum(gather(b_dAqk, row_idx_bc, axis=0), 0)[:, None]
        #     b_dA_ab_j = tl.sum(gather(b_dAab, row_idx_bc, axis=0), 0)[:, None]
        #     b_dA_qb_j = tl.sum(gather(b_dAqb, row_idx_bc, axis=0), 0)[:, None]
        #     b_dA_ak_j = tl.sum(gather(b_dAak, row_idx_bc, axis=0), 0)[:, None]
        # else:
        mask_idx = tl.arange(0, BC) == j
        b_kj = tl.sum(tl.where(mask_idx[:, None], b_k, 0), 0)[None, :]
        b_bj = tl.sum(tl.where(mask_idx[:, None], b_b, 0), 0)[None, :]##None for T维度 BK
        b_gij = tl.sum(tl.where(mask_idx[:, None], b_gi, 0), 0)[None, :]
        b_gej = tl.sum(tl.where(mask_idx[:, None], b_ge, 0), 0)[None, :]

        ###横向，纵向划分读取
        b_dAqk_j = tl.sum(tl.where(mask_idx[None, :], b_dAqk, 0), 1)[:, None]
        b_dAqb_j = tl.sum(tl.where(mask_idx[None, :,None], b_dAqb, 0), 1)[:,:,None]###BC r None for k维度

        b_dAab_j = tl.sum(tl.where(mask_idx[None, :,None,None], b_dAab, 0), 1)[:,:,:,None]#BT r r None for k维度
        b_dAak_j = tl.sum(tl.where(mask_idx[None, :,None,None], b_dAak, 0), 1)[:,:,:,None]##BT r None for k维度

        b_dA_qk_j = tl.sum(tl.where(mask_idx[:, None], b_dAqk, 0), 0)[:, None]
        b_dA_qb_j = tl.sum(tl.where(mask_idx[:, None,None], b_dAqb, 0), 0)[:,:,None]
        b_dA_ab_j = tl.sum(tl.where(mask_idx[:, None,None,None], b_dAab, 0), 0)[:,: ,:,None]
        b_dA_ak_j = tl.sum(tl.where(mask_idx[:, None,None,None], b_dAak, 0), 0)[:,:,:,None]###BC r r 1 for BK//r
        ###BC r r 1 for BK//r

        # [1, BK] b_qj, b_aj
        b_qj = tl.sum(tl.where(mask_idx[:, None], b_q, 0), 0)[None, :]
        b_arrj = tl.sum(tl.where(mask_idx[:, None,None,None], b_arr, 0), 0)[None,:,:,:]#None for BC维度 1 r r BK//r

        m_e = o_i[:, None] > j
        m_i = o_i[:, None] >= j#BC None 
        tmp1 = exp(b_gi - b_gij)##BC BK
        tmp2 = exp(b_ge - b_gij)
        #####BC BK
        b_dq += tl.where(m_i, b_dAqk_j * b_kj * tmp1, 0.)###不改
        b_dq += tl.reshape(tl.where(m_i[:,:,None], b_dAqb_j * tl.reshape(b_bj*tmp1,(BC,r,BK//r)), 0.), (BC, BK))

        b_darr += tl.where(m_e[:,:,None,None], b_dAak_j * tl.reshape(b_kj * tmp2,(BC,r,BK//r))[:,None,:,:], 0.)
        b_darr += tl.where(m_e[:,:,None,None], b_dAab_j * tl.reshape(b_bj * tmp2,(BC,r,BK//r))[:,None,:,:], 0.)
        ###BC r r BK//r

        m_i = o_i[:, None] <= j
        m_e = o_i[:, None] < j
        tmp1 = exp(b_gij - b_gi)
        tmp2 = exp(b_gej - b_gi)
        b_dk += tl.where(m_i, b_dA_qk_j * b_qj * tmp1, 0.)##不动
        b_dk += tl.reshape(tl.sum(tl.where(m_e[:,:,None,None], b_dA_ak_j * b_arrj * tl.reshape(tmp2,(BC,r,BK//r))[:,None,:,:], 0.),1),(BC,BK))

        b_db += tl.reshape(tl.where(m_i[:,:,None], b_dA_qb_j * tl.reshape(b_qj*tmp1,(BC,r,BK//r)), 0.),(BC,BK))
        b_db += tl.reshape(tl.sum(tl.where(m_e[:,:,None,None], b_dA_ab_j * b_arrj * tl.reshape(tmp2,(BC,r,BK//r))[:,None,:,:], 0.),1),(BC,BK))

    b_dmask = tl.sum(b_darr * tl.reshape(b_a,(BC,r,BK//r))[:,None,:,:],-1)
    b_da += tl.reshape(tl.sum(b_darr * b_mask[:,:,:,None],1),(BC,BK))

    # # post processing
    p_dq = tl.make_block_ptr(dq, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_da = tl.make_block_ptr(da, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_db = tl.make_block_ptr(db, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_dgk = tl.make_block_ptr(dgk, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_dgk_offset = tl.make_block_ptr(dgk_offset, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_dqg = tl.make_block_ptr(dqg, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_dkg = tl.make_block_ptr(dkg, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_dag = tl.make_block_ptr(dag, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_dbg = tl.make_block_ptr(dbg, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_dmask = tl.make_block_ptr(dmask, (T, r, r), (H*r*r, r, 1), (i_t * BT, 0, 0), (BC, r, r), (2, 1, 0))

    p_gn = gi + (min(i_t * BT + BT, T) - 1)*stride_qk + o_k
    p_gn = tl.max_contiguous(tl.multiple_of(p_gn, BK), BK)
    b_gn = tl.load(p_gn, mask=m_k, other=0)
    b_da += tl.load(p_dag, boundary_check=(0, 1)) * exp(b_ge)
    b_dq += tl.load(p_dqg, boundary_check=(0, 1)) * exp(b_gi) * scale
    tmp = exp(b_gn[None, :] - b_gi)
    b_dk += tl.load(p_dkg, boundary_check=(0, 1)).to(tl.float32) * tmp
    b_db += tl.load(p_dbg, boundary_check=(0, 1)).to(tl.float32) * tmp


    tl.store(p_dmask, b_dmask.to(p_dmask.dtype.element_ty), boundary_check=(0, 1, 2))
    tl.store(p_dq, (b_dq).to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_da, b_da.to(p_da.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0, 1))


    b_dgk = (b_dq * b_q + b_da * b_a - b_dk * b_k - b_db * b_b).to(tl.float32)
    b_dgk_offset = b_da * b_a
    #####True or false?
    tl.store(p_dgk, b_dgk.to(p_dgk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dgk_offset, b_dgk_offset.to(p_dgk_offset.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4]
        for BK in [32, 64]
    ],
    key=['BK', 'BT', 'K'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_bwd_dgk_kernel(
    dgk,
    dgk_offset,
    dgk_last,
    dgk_output,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int32)
        bos, eos = (i_b * T).to(tl.int32), (i_b * T + T).to(tl.int32)

    stride_qk = H * K
    dgk += (bos * H + i_h) * K
    dgk_offset += (bos * H + i_h) * K
    dgk_last += (i_tg * H + i_h) * K
    dgk_output += (bos * H + i_h) * K
    p_dgk_last = dgk_last + tl.arange(0, BK) + i_k * BK
    m_k = tl.arange(0, BK) + i_k * BK < K
    b_dgk_last = tl.load(p_dgk_last, mask=m_k, other=0)
    p_dgk_offset = tl.make_block_ptr(dgk_offset, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dgk = tl.make_block_ptr(dgk, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_dgk = tl.load(p_dgk, boundary_check=(0, 1))
    b_dgk_offset = tl.load(p_dgk_offset, boundary_check=(0, 1))
    # m_inv_cumsum = (tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :]).to(tl.float32)
    # b_dgk_cumsum = tl.dot(m_inv_cumsum, b_dgk, allow_tf32=False)
    b_dgk_cumsum = tl.cumsum(b_dgk, 0, reverse=True)
    b_dgk_cumsum += b_dgk_last[None, :]
    b_dgk_cumsum -= b_dgk_offset
    p_dgk_output = tl.make_block_ptr(dgk_output, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dgk_output, b_dgk_cumsum.to(p_dgk_output.dtype.element_ty), boundary_check=(0, 1))


def mask_chunk_dplr_bwd_dqk_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    dAqk: torch.Tensor,
    dAqb: torch.Tensor,
    dAak: torch.Tensor,
    dAab: torch.Tensor,
    dqg: torch.Tensor,
    dkg: torch.Tensor,
    dag: torch.Tensor,
    dbg: torch.Tensor,
    dgk_last: torch.Tensor,
    scale: float = 1.0,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
):
    B, T, H, K = q.shape
    r = mask.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    # BK = min(64, triton.next_power_of_2(K)) if check_shared_mem() else min(32, triton.next_power_of_2(K))

    BK = triton.next_power_of_2(K)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NK = triton.cdiv(K, BK)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    dgk = torch.empty_like(gi, dtype=torch.float)
    dgk_offset = torch.empty_like(gi, dtype=torch.float)
    dmask = torch.empty_like(mask)

    grid = (NK, NT, B * H)
    mask_chunk_dplr_bwd_kernel_intra[grid](
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        mask=mask,
        dAqk=dAqk,
        dAqb=dAqb,
        dAak=dAak,
        dAab=dAab,
        dmask=dmask,
        dq=dq,
        dk=dk,
        dgk=dgk,
        dgk_offset=dgk_offset,
        dqg=dqg,
        dkg=dkg,
        dag=dag,
        dbg=dbg,
        da=da,
        db=db,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        r=r,
        BT=BT,
        BC=BT,
        BK=BK,
        GATHER_SUPPORTED=is_gather_supported
    )

    ###应该不需要动
    dgk_output = torch.empty_like(dgk)

    def grid(meta): return (NT, triton.cdiv(K, meta['BK']), B * H)
    chunk_dplr_bwd_dgk_kernel[grid](
        dgk=dgk,
        dgk_offset=dgk_offset,
        dgk_last=dgk_last,
        dgk_output=dgk_output,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
    )
    return dq, dk, da, db, dgk_output,dmask
