# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# from tkinter.constants import N
from typing import Optional

import torch
import triton
import triton.language as tl

from ....ops.utils import prepare_chunk_indices
from ....ops.utils.op import exp, gather
from ....utils import is_gather_supported, use_cuda_graph


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=['BK', 'BT','r'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def mask_chunk_dplr_fwd_A_kernel_intra_sub_intra(
    q,
    k,
    a,
    b,
    mask,
    gi,
    ge,
    qg,
    kg,
    ag,
    bg,
    Aqk,
    Aqb,
    Aab,
    Aak,
    cu_seqlens,
    chunk_indices,
    scale: tl.constexpr,
    T,
    r: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    GATHER_SUPPORTED: tl.constexpr
):
    i_t, i_b, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + tl.arange(0, BC)) < T
    last_idx = min((i_t+1) * BT, T) - 1
    o_A = (bos + i_t * BT + tl.arange(0, BC)) * H*BT + i_h * BT


    p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_a = tl.make_block_ptr(a + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_b = tl.make_block_ptr(b + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_gi = tl.make_block_ptr(gi + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_ge = tl.make_block_ptr(ge + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_g_last = gi + (bos * H + i_h) * K + last_idx * H * K + tl.arange(0, BK)
    b_g_last = tl.load(p_g_last, mask=m_k, other=0)
    p_qg = tl.make_block_ptr(qg + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_kg = tl.make_block_ptr(kg + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_ag = tl.make_block_ptr(ag + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_bg = tl.make_block_ptr(bg + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = b_q * scale
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_a = tl.load(p_a, boundary_check=(0, 1))
    b_b = tl.load(p_b, boundary_check=(0, 1))
    b_gi = tl.load(p_gi, boundary_check=(0, 1)).to(tl.float32)
    b_ge = tl.load(p_ge, boundary_check=(0, 1)).to(tl.float32)

    # deal with decay term.
    g_exp = exp(b_gi)
    g_exp_inv = exp(-b_gi + b_g_last[None, :])
    b_qg = b_q * g_exp
    b_kg = b_k * g_exp_inv
    b_bg = b_b * g_exp_inv
    b_ag = b_a * exp(b_ge)
    tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_bg, b_bg.to(p_bg.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_ag, b_ag.to(p_ag.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    b_q = b_q.to(b_k.dtype)
    p_mask = tl.make_block_ptr(mask + (bos*H + i_h)*r*r, (T,r,r),(H*r*r,r,1), (i_t*BT,0,0),(BC,r,r),(2,1,0))
    b_mask = tl.load(p_mask,boundary_check=(0,1,2))#BT,r,r
    b_arr = tl.reshape(b_a,(BC,r,BK//r))[:,None,:,:].to(tl.float32)*b_mask[:,:,:,None].to(tl.float32)#BC r r BK//r
    b_arr = tl.reshape(b_arr,(BC,r,BK))

    for j in range(0, min(BC, T - i_t * BT)):
        jmask = tl.arange(0, BC) == j ###取出第j行元素
        b_k_j = tl.sum(tl.where(jmask[:, None], b_k, 0), 0)[None, :]
        b_gk_j = tl.sum(tl.where(jmask[:, None], b_gi, 0), 0)[None, :]
        b_b_j = tl.sum(tl.where(jmask[:, None], b_b, 0), 0)[None, :]

        tmp = exp(b_gi - b_gk_j)
        b_A_qk = tl.sum(b_q * b_k_j * tmp,1)###get BC
        m_i = (o_i >= j).to(tl.float32)
        b_A_qk = b_A_qk * m_i
        
        b_A_qb = tl.sum(tl.reshape(b_q * b_b_j * tmp,(BC,r,BK//r)),-1)###BC r
        b_A_qb = b_A_qb * m_i[:,None]

        tmp2 = exp(b_ge - b_gk_j)
        b_A_ak = (b_arr * b_k_j[:,None,:] * tmp2[:,None,:])#BC r BK
        b_A_ak = tl.sum(tl.reshape(b_A_ak,(BC,r,r,BK//r)),-1)
        
        m_i2 = (o_i > j).to(tl.float32)
        b_A_ak = b_A_ak * m_i2[:,None,None]

        b_A_ab = (b_arr * b_b_j[:,None,:] * tmp2[:,None,:])#BC r BK
        b_A_ab = tl.sum(tl.reshape(b_A_ab,(BC,r,r,BK//r)),-1)
        b_A_ab = b_A_ab * m_i2[:,None,None]
        tl.store(Aqk + o_A + j, b_A_qk.to(dtype=Aqk.dtype.element_ty, fp_downcast_rounding="rtne"), mask=m_A)
        aqb = tl.make_block_ptr(Aqb + ((bos + i_t * BT) * H*BT + i_h * BT + j) * r, (BC,r),(H*BT*r,1), (0,0),(BC,r),(1,0))
        tl.store(aqb, b_A_qb.to(dtype=Aqb.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0,1))
        aak = tl.make_block_ptr(Aak + ((bos + i_t * BT) * H*BT + i_h * BT + j) * r * r, (BC,r,r),(H*BT*r*r,r,1), (0,0,0),(BC,r,r),(2,1,0))
        tl.store(aak, b_A_ak.to(dtype=Aak.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0,1,2))
        ab = tl.make_block_ptr(Aab + ((bos + i_t * BT) * H*BT + i_h * BT + j) * r * r, (BC,r,r),(H*BT*r*r,r,1), (0,0,0),(BC,r,r),(2,1,0))
        tl.store(ab, b_A_ab.to(dtype=Aab.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0,1,2))


def mask_chunk_dplr_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    scale: float,
    chunk_size: int,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, K = k.shape
    r = mask.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    Aqk = q.new_empty(B, T, H, BT, dtype=torch.float)#q.dtype)
    Aqb = q.new_empty(B, T, H, BT,r, dtype=torch.float)#q.dtype)
    Aak = q.new_empty(B, T, H, BT,r,r, dtype=torch.float)####考虑预先加mask
    Aab = q.new_empty(B, T, H, BT,r,r, dtype=torch.float)

    grid = (NT, B, H)
    BK = triton.next_power_of_2(K)
    qg = torch.empty_like(q)
    kg = torch.empty_like(k, dtype=q.dtype)
    ag = torch.empty_like(a, dtype=q.dtype)
    bg = torch.empty_like(b, dtype=q.dtype)
    mask_chunk_dplr_fwd_A_kernel_intra_sub_intra[grid](
        q=q,
        k=k,
        a=a,
        b=b,
        mask=mask,
        gi=gi,
        ge=ge,
        Aqk=Aqk,
        Aqb=Aqb,
        Aab=Aab,
        Aak=Aak,
        qg=qg,
        kg=kg,
        ag=ag,
        bg=bg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BT,
        BK=BK,
        r=r,
        GATHER_SUPPORTED=is_gather_supported
    )
    return Aab, Aqk, Aak, Aqb, qg, kg, ag, bg
