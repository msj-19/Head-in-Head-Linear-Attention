from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ....ops.utils import prepare_chunk_indices
from ....utils import check_shared_mem, is_intel_alchemist, use_cuda_graph

# https://github.com/intel/intel-xpu-backend-for-triton/issues/3449
triton_config = {'grf_mode': 'large'} if is_intel_alchemist else {}


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config(triton_config, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def mask_prepare_wy_repr_bwd_kernel(
    A_ab_inv,
    A_ak,
    ag,
    v,
    mask,
    dmask,
    dw,
    du,
    dv,
    dv0,
    dag,
    dAak,
    dAab,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    r: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    # 修改后的 Triton 3.1 兼容版本
    p_dAak = tl.make_block_ptr(dAak + (bos*H + i_h) * BT*r*r, (T,BT,r,r), (H*BT*r*r,r*r,r,1), (i_t * BT,0,0,0), (BT, BT,r,r), (0, 1,2,3))
    p_dAab = tl.make_block_ptr(dAab + (bos*H + i_h) * BT*r*r, (T,BT,r,r), (H*BT*r*r,r*r,r,1), (i_t * BT,0,0,0), (BT, BT,r,r), (0, 1,2,3))
    
    o_s = tl.arange(0, BT)

    p_A_ab_inv = tl.make_block_ptr(A_ab_inv + (bos*H*r + i_h) * BT*r, (T*r, BT*r), (H*BT*r, 1), (i_t * BT*r, 0), (BT*r, BT*r), (1, 0))
    p_A_ak = tl.make_block_ptr(A_ak + (bos*H + i_h) * BT*r*r, (T, BT,r,r), (H*BT*r*r, r*r, r, 1), (i_t * BT, 0, 0, 0), (BT, BT,r,r), (3,2, 1, 0))
    ####A ak

    b_Aab_inv = tl.load(p_A_ab_inv, boundary_check=(0, 1))
    b_Aab_inv = tl.reshape(b_Aab_inv,(BT,r,BT,r))
    b_Aab_inv = tl.where((o_s[:, None] >= o_s[None, :])[:,None,:,None], b_Aab_inv, 0)###如何mask
    b_Aab_inv = tl.reshape(b_Aab_inv,(BT*r,BT*r))
    b_A_ab_inv_t = tl.trans(b_Aab_inv)##BT r BT

    b_Aak = tl.load(p_A_ak,boundary_check=(0,1,2,3))
    b_Aak = tl.where((o_s[:, None] > o_s[None, :])[:,:,None,None], b_Aak, 0)    ###如何mask
    b_Aak2 = tl.reshape(tl.permute(tl.sum(b_Aak, -1),(0,2,1)),(BT*r,BT)) ###BT*r BT
    b_A_tmp_t = tl.trans(tl.dot(b_Aab_inv.to(tl.float32), b_Aak2.to(tl.float32)))

    dk = K//r
    b_dA_tmp = tl.zeros([BT*r, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))##正常的shape
        p_dv = tl.make_block_ptr(dv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))##正常的shape
        p_dv0 = tl.make_block_ptr(dv0 + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))##正常的shape
        p_du = tl.make_block_ptr(du + (bos*r*H + i_h) * V, (T*r, V), (H*V, 1), (i_t * BT * r, i_v * BV), (BT*r, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_du = tl.load(p_du, boundary_check=(0, 1))#BT*r V
        b_dA_tmp += tl.dot(b_du.to(tl.float32), tl.trans(b_v.to(tl.float32)))
        b_dv0 = tl.load(p_dv0, boundary_check=(0, 1))
        b_dv = b_dv0.to(tl.float32) + tl.dot(b_A_tmp_t, b_du.to(tl.float32))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))###no need change
    ###########到这里计算dv是对的

    p_Aab_inv_t = tl.make_block_ptr(A_ab_inv + (bos*H*r + i_h) * BT*r, (BT*r, T*r), (1, H*BT*r), (0, i_t * BT*r), (BT*r, BT*r), (0, 1))
    p_Aak = tl.make_block_ptr(A_ak + (bos*H + i_h) * BT*r*r, (T,BT,r,r), (H*BT*r*r,r*r,r,1), (i_t * BT,0,0,0), (BT, BT,r,r), (0, 1,2,3))
    p_dAak = tl.make_block_ptr(dAak + (bos*H + i_h) * BT*r*r, (T,BT,r,r), (H*BT*r*r,r*r,r,1), (i_t * BT,0,0,0), (BT, BT,r,r), (0, 1,2,3))
    p_dAab = tl.make_block_ptr(dAab + (bos*H + i_h) * BT*r*r, (T,BT,r,r), (H*BT*r*r,r*r,r,1), (i_t * BT,0,0,0), (BT, BT,r,r), (0, 1,2,3))
    b_A_ab_inv_t = tl.load(p_Aab_inv_t, boundary_check=(0, 1))###BT*r BT*r
    b_A_ak = tl.load(p_Aak, boundary_check=(0,1,2,3))
    b_A_ak = tl.where((tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :])[:,:,None,None], b_A_ak, 0)
    b_A_ab_inv_t = tl.reshape(b_A_ab_inv_t, (BT,r,BT,r))
    b_A_ab_inv_t = tl.reshape(tl.where((tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :])[:,None,:,None], b_A_ab_inv_t, 0),(BT*r,BT*r))
    b_Aak2_t = tl.trans(tl.reshape(tl.permute(tl.sum(b_A_ak, -1),(0,2,1)),(BT*r,BT)))#BT BT*r
    b_A_tmp_t = tl.dot(b_Aak2_t, b_A_ab_inv_t).to(v.dtype.element_ty)###BT BT*r
    b_dA_tmp = tl.zeros([BT*r, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))##正常的shape
        p_dv = tl.make_block_ptr(dv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))##正常的shape
        p_dv0 = tl.make_block_ptr(dv0 + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))##正常的shape
        p_du = tl.make_block_ptr(du + (bos*r*H + i_h) * V, (T*r, V), (H*V, 1), (i_t * BT * r, i_v * BV), (BT*r, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_du = tl.load(p_du, boundary_check=(0, 1))#BT*r V
        b_dA_tmp += (tl.dot(b_du.to(b_v.dtype), tl.trans(b_v)))###BT*r BT的shape
    # [BT, 1, BT]，与原来 [BT,BT] 在 dim1 上等价（全复制到 r 维）
    m_i = tl.arange(0, BT)[:, None, None] > tl.arange(0, BT)[None, None, :]
    m_i = tl.broadcast_to(m_i, (BT, r, BT))
    m_i = tl.reshape(m_i, (BT * r, BT))
    b_dA_tmp = (tl.where(m_i, b_dA_tmp, 0))####BT*r BT
    b_dA_ak = tl.dot(b_A_ab_inv_t, b_dA_tmp)###BT*r BT
    b_dA_ak = tl.where(m_i, b_dA_ak, 0)###BT*r BT
    b_dA_ak = tl.broadcast_to(tl.permute(tl.reshape(b_dA_ak, (BT,r,BT)),(0,2,1))[:,:,:,None],(BT,BT,r,r))
    tl.store(p_dAak, b_dA_ak, boundary_check=(0,1,2,3))###ok
    b_dA_ab_inv = tl.dot(b_dA_tmp, b_Aak2_t)###BT*r BT*r

    dk = K//r


    m_i = tl.arange(0, BT)[:, None, None] > tl.arange(0, BT)[None, None, :]
    m_i = tl.broadcast_to(m_i, (BT, r, BT))
    m_i = tl.reshape(m_i, (BT * r, BT))
    b_dA_tmp = (tl.where(m_i, b_dA_tmp, 0))####BT*r BT
    b_dA_ak = tl.dot(b_A_ab_inv_t, b_dA_tmp)
    b_dA_ak = tl.where(m_i, b_dA_ak, 0)###BT*r BT
    b_dA_ak = tl.broadcast_to(tl.permute(tl.reshape(b_dA_ak, (BT,r,BT)),(0,2,1))[:,:,:,None],(BT,BT,r,r))
    tl.store(p_dAak, b_dA_ak, boundary_check=(0,1,2,3))###ok
    ####到这里前面应该能把dv算对



    b_dA_ab_inv = tl.dot(b_dA_tmp, tl.trans(b_Aak2.to(tl.float32)))
    dk = K//r
    for i_r in range(r):
        p_maskr = tl.make_block_ptr(mask + (bos*H + i_h)*r*r, (T,r,r),(H*r*r,r,1), (i_t*BT,0,i_r),(BT,r,1),(2,1,0))
        b_maskr = tl.load(p_maskr,boundary_check=(0,1,2))#BT,r,1
        p_dmask = tl.make_block_ptr(dmask + (bos*H + i_h)*r*r, (T,r,r),(H*r*r,r,1), (i_t*BT,0,i_r),(BT,r,1),(2,1,0))
        b_dmask = tl.zeros([BT, r], dtype=tl.float32)
        for i_k in range(tl.cdiv(dk, BK)):
            p_ag = tl.make_block_ptr(ag + (bos*H + i_h) * K, (T,K), (H*K,1), (i_t * BT, i_r*dk + i_k * BK), (BT,BK), (1, 0))
            p_dag = tl.make_block_ptr(dag + (bos*H + i_h) * K, (T,K), (H*K,1), (i_t * BT, i_r*dk + i_k * BK), (BT,BK), (1, 0))
            p_dw = tl.make_block_ptr(dw + (bos*r*H + i_h) * K, (T*r,K), (H*K,1), (i_t * BT * r,i_r*dk + i_k * BK), (BT*r,BK), (1, 0))

            b_ag = tl.load(p_ag, boundary_check=(0, 1))###BT BK
            b_dw = tl.load(p_dw, boundary_check=(0, 1))###BT*r BK
            # b_maskr: [BT, r, 1]；与 fwd 中 ag 按 mask 分块相乘一致，grad 在 r 维收缩
            b_agbm = b_ag[:, None, :] * b_maskr.to(b_ag.dtype)
            b_agbm = tl.reshape(b_agbm, (BT * r, BK))
            b_dA_ab_inv += tl.dot(b_dw.to(tl.float32), tl.trans(b_agbm.to(tl.float32)))
            b_dagbm = tl.reshape(
                tl.dot(b_A_ab_inv_t.to(tl.float32), b_dw.to(tl.float32)),
                (BT, r, BK),
            )
            b_dag = tl.sum(b_dagbm * b_maskr, 1)
            b_dmask += tl.sum(b_dagbm * b_ag[:,None,:],-1)##BT r 1
            tl.store(p_dag, b_dag.to(p_dag.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dmask, (b_dmask[:,:,None]).to(p_dmask.dtype.element_ty), boundary_check=(0, 1, 2))

    i = tl.arange(0, BT * r)[:, None]
    j = tl.arange(0, BT * r)[None, :]
    iB = i // r
    jB = j // r
    da_mask = iB >= jB
    b_dA_ab_inv = tl.where(da_mask, b_dA_ab_inv, 0)
    b_dA_ab_inv = tl.dot(b_A_ab_inv_t, b_dA_ab_inv)
    b_dA_ab_inv = tl.dot(b_dA_ab_inv, b_A_ab_inv_t)###BT*r BT*r
    da_mask = iB > jB
    b_dA_ab_inv = tl.where(da_mask, b_dA_ab_inv, 0) ###BT*r BT*r
    b_dA_ab_inv = tl.permute(tl.reshape(b_dA_ab_inv, (BT,r,BT,r)),(0,2,1,3))
    tl.store(p_dAab, b_dA_ab_inv, boundary_check=(0, 1))###BT*r BT*r


def mask_chunk_dplr_bwd_wy(
    A_ab_inv: torch.Tensor,
    A_ak: torch.Tensor,
    mask: torch.Tensor,
    v: torch.Tensor,
    ag: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    dv0: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A_ab_inv, A_ak, v, ag, dw, du = map(lambda x: x.contiguous(), [A_ab_inv, A_ak, v, ag, dw, du])
    B, T, r,H, K, V = *dw.shape, du.shape[-1]
    r = mask.shape[-1]


    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(triton.next_power_of_2(K//r), 64)
    BV = min(triton.next_power_of_2(V), 64) if check_shared_mem() else min(triton.next_power_of_2(V), 32)

    dA_ab = torch.empty(B,T,H,BT,r,r,dtype=torch.float,device=A_ab_inv.device)
    dA_ak = torch.empty_like(A_ak, dtype=torch.float)
    dv = torch.empty_like(v)
    dag = torch.empty_like(ag)
    dmask = torch.empty_like(mask)
    
    mask_prepare_wy_repr_bwd_kernel[(NT, B * H)](
        A_ab_inv=A_ab_inv,
        A_ak=A_ak,
        mask=mask,
        dmask=dmask,
        ag=ag,
        v=v,
        dw=dw,
        du=du,
        dv=dv,
        dv0=dv0,
        dag=dag,
        dAak=dA_ak,
        dAab=dA_ab,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        r=r,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dA_ab, dA_ak, dv, dag,dmask
