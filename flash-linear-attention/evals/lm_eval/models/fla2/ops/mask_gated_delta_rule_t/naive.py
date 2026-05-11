# -*- coding: utf-8 -*-

import torch
from einops import rearrange
from typing import Optional

import time
import torch
import triton
import triton.language as tl
from einops import rearrange
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
from fla.ops.utils import chunk_local_cumsum

from fla.ops import chunk_gated_delta_rule
@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float('-inf')))



@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def gated_fwd_recompute_w_u_kernel(
    k,
    v,
    beta,
    mask_ij,
    w,
    u,
    Aw,
    Au,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    T,
    K,
    V,
    r: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    dk = K//r
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    p_Aw = tl.make_block_ptr(Aw + i_bh*T*BT*r*r ,(T*r,BT*r), (BT*r,1), (i_t*BT*r,0), (BT*r,BT*r),(1,0))
    b_Aw = tl.load(p_Aw, boundary_check=(0, 1)).to(k.dtype.element_ty)
    for i_r in range(r):
        p_mask = mask_ij + tl.arange(0,r)*r+i_r#读取第ir列
        b_mask = tl.load(p_mask)
        for i_k in range(tl.cdiv(dk, BK)):
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*dk + i_k * BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)[:,None,:]*b_mask[None,:,None].to(b_k.dtype)#BT*r*d
            b_kb = tl.reshape(b_kb,(BT*r,BK))
            b_w = tl.dot(b_Aw, b_kb, allow_tf32=False)#get BT*r *BK
            p_w = tl.make_block_ptr(w + i_bh * s_qk_h*r, (T*r, K), (s_qk_t, s_qk_d), (i_t * BT * r, i_r*dk + i_k * BK), (BT*r, BK), (1, 0))
            tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))
    tl.debug_barrier()
    b_Aw = None
    p_Au = tl.make_block_ptr(Au + i_bh*T*BT*r*r ,(T*r,BT*r), (BT*r,1), (i_t*BT*r,0), (BT*r,BT*r),(1,0))
    b_Au = tl.load(p_Au, boundary_check=(0, 1)).to(k.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):#no need for 任意mask不使用 #无需for 循环 ，这里也不存在mask
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)[:,None,:]*tl.full([r],1, dtype=b_v.dtype)[None,:,None]
        b_vb = tl.reshape(b_vb,(BT*r,BV))
        b_u = tl.dot(b_Au, b_vb, allow_tf32=False)
        p_u = tl.make_block_ptr(u + i_bh * s_vo_h*r, (T*r, V), (s_vo_t, s_vo_d), (i_t * BT*r, i_v * BV), (BT*r, BV), (1, 0))
        tl.store(p_u, (b_u).to(p_u.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16)
    ],
    key=["BT", "BK","r"],
)
@triton.jit
def gated_chunk_scaled_dot_kkt_fwd_kernel(        
    k,
    beta,
    g_cumsum,
    mask_ij,
    A,
    Ag,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    T,
    K,
    r:  tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    b_A = tl.zeros([BT,BT,r,r], dtype=tl.float32)#r*BT r*BT
    dk = K//r
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    for i_r in range(r):
        r_mask = tl.arange(0, r) == i_r 
        p_mask = mask_ij + tl.arange(0,r)* r + i_r#列读，因而是行数目
        b_mask = tl.load(p_mask)
        ij_mask = b_mask[:,None]*r_mask[None,:]#行数

        for i_k in range(tl.cdiv(dk, BK)):#分块k读取计算
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r * dk + i_k * BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
            dot = tl.dot(b_kb, tl.trans(b_k), allow_tf32=False)
            b_A += dot[:,:,None,None]*ij_mask[None,None,:,:]
    b_A = tl.where((tl.arange(0, BT)[:,None] > tl.arange(0, BT)[None,:])[:,:,None,None], b_A, 0)
    p_A = tl.make_block_ptr(A + (i_bh*T//BT+i_t)*BT*BT*r*r ,(BT,BT,r,r), (BT*r*r,r*r,r,1), (0,0,0,0), (BT,BT,r,r),(3,2,1,0))
    tl.store(p_A, (b_A).to(p_A.dtype.element_ty),boundary_check=(0,1,2,3))

    p_g = tl.make_block_ptr(g_cumsum + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_g_diff = b_g[:, None] - b_g[None, :]
    b_g_diff = safe_exp(b_g_diff)

    b_Ag = b_A * ((b_g_diff)[:,:,None,None])#BT BT
    p_Ag = tl.make_block_ptr(Ag + (i_bh*T//BT+i_t)*BT*BT*r*r ,(BT,BT,r,r), (BT*r*r,r*r,r,1), (0,0,0,0), (BT,BT,r,r),(3,2,1,0))
    tl.store(p_Ag, (b_Ag).to(p_Ag.dtype.element_ty),boundary_check=(0,1,2,3))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16)
    ],
    key=["BT", "r"],
)
@triton.jit
def solve_tril_16x16_kernel(
    A,
    Ad,
    s_A_bh,
    s_Ad_bh,
    T,
    r:  tl.constexpr,
    BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    offset = (i_t * 16) % BT 

    p_A = tl.make_block_ptr(A + (i_bh)*s_A_bh, (T,BT,r,r),(BT*r*r,r*r,r,1) ,(i_t * 16, offset, 0, 0), (16, 16,r,r), (3,2,1,0))
    b_A = tl.load(p_A, boundary_check=(0,1,2,3)).to(tl.float32)
    b_A = -tl.where((tl.arange(0, 16)[:,None] > tl.arange(0, 16)[None,:])[:,:,None,None], b_A, 0)

    for i in range(1, 16):
        mask = tl.arange(0, 16) == i 
        b_a = tl.sum(tl.where(mask[:,None,None,None], b_A, 0), 0)
        q = (tl.sum(b_a[:,None,:,:,None]*b_A[:,:,None,:,:],-2))
        b_a = b_a + tl.sum(q,0)*((tl.arange(0, 16) < i)[:,None,None])
        b_A = tl.where(mask[:,None,None,None],b_a,b_A)#按行计算 ，逐步交换结果
    b_A += ((tl.arange(0, 16)[:, None, None, None] == tl.arange(0, 16)[None, :, None, None])&(tl.arange(0, r)[None, None, :, None] == tl.arange(0, r)[None, None, None, :]))
    
    b_A = tl.permute(b_A,(0,2,1,3))
    b_A = tl.reshape(b_A,(16*r,16*r))#BT*r BT*r
    p_Ad = tl.make_block_ptr(Ad + (i_bh)*s_Ad_bh,(T*r,16*r),(16*r,1), (i_t * 16 * r, 0), (16*r,16*r), (1,0))
    tl.store(p_Ad, (b_A).to(p_Ad.dtype.element_ty),boundary_check=(0,1))

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16)
    ],
    key=["r"],
)
@triton.jit
def merge_16x16_to_32x32_inverse_kernel(
        A,
        Ad,
        Ai,
        s_A_bh,
        s_Ad_bh,
        T,
        r: tl.constexpr,
        BT: tl.constexpr 
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    p_A21 = tl.make_block_ptr(A + (i_bh)*s_A_bh, (T*r,32*r),(32*r,1) ,((i_t * 32 + 16) *r, 0), (16*r, 16*r), (1,0))
    b_A21 = tl.load(p_A21, boundary_check=(0,1)).to(tl.float32)

    p_Ad11  = tl.make_block_ptr(Ad + (i_bh)*s_Ad_bh,(T*r,16*r),(16*r,1), (i_t * 32 * r, 0), (16*r,16*r), (1,0))
    p_Ad22  = tl.make_block_ptr(Ad + (i_bh)*s_Ad_bh,(T*r,16*r),(16*r,1), ((i_t *32 +16) * r, 0), (16*r,16*r), (1,0))

    p_Ai11 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,32*r), (32*r, 1), (i_t * 32 * r , 0), (16*r, 16*r), (1, 0))
    p_Ai22 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,32*r), (32*r, 1), ((i_t * 32 + 16) * r , 16*r), (16*r, 16*r), (1, 0))
    p_Ai21 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,32*r), (32*r, 1), ((i_t * 32 + 16) * r, 0), (16*r, 16*r), (1, 0))

    Ai11 = tl.load(p_Ad11, boundary_check=(0, 1)).to(tl.float32)
    Ai22 = tl.load(p_Ad22, boundary_check=(0, 1)).to(tl.float32)
    Ai21 = -tl.dot(tl.dot(Ai22,b_A21, input_precision='ieee'),Ai11,input_precision='ieee')
    tl.store(p_Ai11,Ai11.to(p_Ai11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai22,Ai22.to(p_Ai22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai21,Ai21.to(p_Ai21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16)
    ],
    key=["r"],
)
@triton.jit
def merge_16x16_to_64x64_inverse_kernel(
        A,
        Ad,
        Ai,
        s_A_bh,
        s_Ad_bh,
        T,
        r: tl.constexpr,
        BT: tl.constexpr 
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    p_A21 = tl.make_block_ptr(A + (i_bh)*s_A_bh, (T*r,64*r),(64*r,1) ,((i_t * 64 + 16) *r, 0), (16*r, 16*r), (1,0))
    p_A31 = tl.make_block_ptr(A + (i_bh)*s_A_bh, (T*r,64*r),(64*r,1) ,((i_t * 64 + 32) *r, 0), (16*r, 16*r), (1,0))
    p_A32 = tl.make_block_ptr(A + (i_bh)*s_A_bh, (T*r,64*r),(64*r,1) ,((i_t * 64 + 32) *r, 16*r), (16*r, 16*r), (1,0))
    p_A41 = tl.make_block_ptr(A + (i_bh)*s_A_bh, (T*r,64*r),(64*r,1) ,((i_t * 64 + 48) *r, 0), (16*r, 16*r), (1,0))
    p_A42 = tl.make_block_ptr(A + (i_bh)*s_A_bh, (T*r,64*r),(64*r,1) ,((i_t * 64 + 48) *r, 16*r), (16*r, 16*r), (1,0))
    p_A43 = tl.make_block_ptr(A + (i_bh)*s_A_bh, (T*r,64*r),(64*r,1) ,((i_t * 64 + 48) *r, 32*r), (16*r, 16*r), (1,0))
    
    b_A21 = tl.load(p_A21, boundary_check=(0,1)).to(tl.float32)
    b_A31 = tl.load(p_A31, boundary_check=(0,1)).to(tl.float32)
    b_A32 = tl.load(p_A32, boundary_check=(0,1)).to(tl.float32)
    b_A41 = tl.load(p_A41, boundary_check=(0,1)).to(tl.float32)
    b_A42 = tl.load(p_A42, boundary_check=(0,1)).to(tl.float32)
    b_A43 = tl.load(p_A43, boundary_check=(0,1)).to(tl.float32)


    p_Ad11  = tl.make_block_ptr(Ad + (i_bh)*s_Ad_bh,(T*r,16*r),(16*r,1), (i_t * 64 * r, 0), (16*r,16*r), (1,0))
    p_Ad22  = tl.make_block_ptr(Ad + (i_bh)*s_Ad_bh,(T*r,16*r),(16*r,1), ((i_t * 64 + 16) * r, 0), (16*r,16*r), (1,0))
    p_Ad33  = tl.make_block_ptr(Ad + (i_bh)*s_Ad_bh,(T*r,16*r),(16*r,1), ((i_t * 64 + 32) * r, 0), (16*r,16*r), (1,0))
    p_Ad44  = tl.make_block_ptr(Ad + (i_bh)*s_Ad_bh,(T*r,16*r),(16*r,1), ((i_t * 64 + 48) * r, 0), (16*r,16*r), (1,0))


    p_Ai11 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,64*r), (64*r, 1), ((i_t * 64 ) *r, 0), (16*r, 16*r), (1, 0))
    p_Ai22 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,64*r), (64*r, 1), ((i_t * 64 + 16) *r, 16*r), (16*r, 16*r), (1, 0))
    p_Ai33 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,64*r), (64*r, 1), ((i_t * 64 + 32) *r, 32*r), (16*r, 16*r), (1, 0))
    p_Ai44 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,64*r), (64*r, 1), ((i_t * 64 + 48) *r, 48*r), (16*r, 16*r), (1, 0))
    p_Ai21 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,64*r), (64*r, 1), ((i_t * 64 + 16) *r, 0), (16*r, 16*r), (1, 0))
    p_Ai31 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,64*r), (64*r, 1), ((i_t * 64 + 32) *r, 0), (16*r, 16*r), (1, 0))
    p_Ai32 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,64*r), (64*r, 1), ((i_t * 64 + 32) *r, 16*r), (16*r, 16*r), (1, 0))
    p_Ai41 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,64*r), (64*r, 1), ((i_t * 64 + 48) *r ,0), (16*r, 16*r), (1, 0))
    p_Ai42 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,64*r), (64*r, 1), ((i_t * 64 + 48) *r, 16*r), (16*r, 16*r), (1, 0))
    p_Ai43 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,64*r), (64*r, 1), ((i_t * 64 + 48) *r, 32*r), (16*r, 16*r), (1, 0))


    Ai11 = tl.load(p_Ad11, boundary_check=(0, 1)).to(tl.float32)
    Ai22 = tl.load(p_Ad22, boundary_check=(0, 1)).to(tl.float32)
    Ai33 = tl.load(p_Ad33, boundary_check=(0, 1)).to(tl.float32)
    Ai44 = tl.load(p_Ad44, boundary_check=(0, 1)).to(tl.float32)
    
    Ai21 = -tl.dot(tl.dot(Ai22,b_A21, input_precision='ieee'),Ai11,input_precision='ieee')
    Ai32 = -tl.dot(tl.dot(Ai33,b_A32, input_precision='ieee'),Ai11,input_precision='ieee')
    Ai43 = -tl.dot(tl.dot(Ai44,b_A43, input_precision='ieee'),Ai11,input_precision='ieee')

    Ai31 = -tl.dot(
            Ai33,
            tl.dot(b_A31,Ai11, input_precision='ieee')+
            tl.dot(b_A32,Ai21, input_precision='ieee'),
            input_precision='ieee')

    Ai42 = -tl.dot(
            Ai44,
            tl.dot(b_A42,Ai22, input_precision='ieee')+
            tl.dot(b_A43,Ai32, input_precision='ieee'),
            input_precision='ieee')

    Ai41 = -tl.dot(
        Ai44,
        tl.dot(b_A41, Ai11, input_precision='ieee') +
        tl.dot(b_A42, Ai21, input_precision='ieee') +
        tl.dot(b_A43, Ai31, input_precision='ieee'),
        input_precision='ieee'
    )

    tl.store(p_Ai11,Ai11.to(p_Ai11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai22,Ai22.to(p_Ai22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai33,Ai33.to(p_Ai33.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai44,Ai44.to(p_Ai44.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai21,Ai21.to(p_Ai21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai31,Ai31.to(p_Ai31.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai32,Ai32.to(p_Ai32.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai41,Ai41.to(p_Ai41.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai42,Ai42.to(p_Ai42.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai43,Ai43.to(p_Ai43.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))



def gated_chunk_scaled_dot_kkt_fwd(k: torch.Tensor,
                                   beta: torch.Tensor,
                                   mask: torch.Tensor,
                                   g_cumsum:Optional[torch.Tensor] = None,
                                   BT:int = 32,
                                   output_dtype: torch.dtype=torch.float32):
    # gated_chunk_scaled_dot_kkt_fwd(k=k,beta=beta,g_cumsum=g,mask=mask,BT=BT,output_dtype=torch.float32)
    B, H, T, K = k.shape
    r = mask.shape[-1]
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K//r), 64)
    A = torch.empty(B*H*NT,BT*BT,r*r,device=k.device, dtype=output_dtype).contiguous()   
    Ag = torch.empty(B*H*NT,BT*BT,r*r,device=k.device, dtype=output_dtype).contiguous()                                                                                                                      
    gated_chunk_scaled_dot_kkt_fwd_kernel[(NT, B*H)](
        k, beta, g_cumsum, mask, A,Ag,
        T*K, K, 1,
        T, K, r, BT, BK
    )
    return A,Ag

def solve_tril(A,mask,k,BT,output_dtype=torch.float32):
    B, H, T, K = k.shape
    r = mask.shape[-1]
    NT = triton.cdiv(T, 16)
    Ad = torch.empty(B,H,NT*16*r,16*r,device=A.device, dtype=torch.float if BT != 16 else output_dtype)
    solve_tril_16x16_kernel[(NT, B*H)](
            A,Ad,
            T*BT*r*r,#s_abh
            T*16*r*r,#s_adbh
            T,
            r, BT
    )
    if BT == 16:                                                                                                                       
        return Ad

    A = rearrange(A,'b (t l) (c r)->b (t c) (l r)',t=BT,c=r).contiguous()#BT*r BT*r  
    if BT == 32:
        NT = triton.cdiv(T, BT)
        Ai = torch.zeros(B,H,NT*BT*r,BT*r,device=A.device, dtype=output_dtype)
        merge_16x16_to_32x32_inverse_kernel[(NT, B*H)](
            A,Ad,Ai,
            T*BT*r*r,#s_a_bh and s_ai_bh
            T*16*r*r,#s_ad_bh
            T,r,BT
        )
        return Ai

    if BT == 64:
        NT = triton.cdiv(T, BT)
        Ai = torch.zeros(B,H,NT*BT*r,BT*r,device=A.device, dtype=output_dtype)
        merge_16x16_to_64x64_inverse_kernel[(NT, B*H)](
            A,Ad,Ai,
            T*BT*r*r,#s_a_bh and s_ai_bh
            T*16*r*r,#s_ad_bh
            T,r,BT
        )
        return Ai


def gated_fwd_recompute_w_u(k, v, beta,mask, Aw,Au,BT):
    B, H, T, K, V = *k.shape, v.shape[-1]
    r = mask.shape[-1]
    u = torch.empty(B,H,r*T,V,device=k.device, dtype=k.dtype)
    w = torch.empty(B,H,r*T,K,device=k.device, dtype=k.dtype)
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K//r), 64)#32
    BV = min(triton.next_power_of_2(V), 64)
    gated_fwd_recompute_w_u_kernel[(NT, B*H)](
        k, v, beta,mask, w, u, Aw,Au,
        T*K, K, 1,
        T*V, V, 1,
        T, K, V, r,BT, BK, BV
    )
    return w, u




#finish
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def gated_chunk_delta_rule_fwd_kernel_h(
    k,
    v,#u
    d,#w
    v_new,
    g,
    h,
    initial_state,  # initial state of the chunk [B, H, D_head_K, D_head_V]
    final_state,  # final state of the chunk [B, H, D_head_K, D_head_V]
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    r: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)#读取一横行
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(initial_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_h = tl.make_block_ptr(h + i_bh * NT * K * V + i_t * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        #这里save是对的
        b_h_cumsum = tl.zeros([r, BK//r, BV], dtype=tl.float32)
        for i_r in range(r):
            for i_c in range(tl.cdiv(BT, BC)):#BK 大，通过BC 分块
                r_mask = tl.arange(0,r) == i_r
                p_k = tl.make_block_ptr(k + i_bh * K * T, (T, K), (K, 1),
                                        (i_t * BT + i_c * BC, i_k * BK + i_r * BK//r), (BC,BK//r), (1, 0))#读取对应
                p_d = tl.make_block_ptr((d + i_bh * T * r * K),(T, r, K ),(r * K, K, 1),
                                        (i_t * BT + i_c * BC, i_r, i_k * BK), (BC,1,BK),(2,1,0))
                p_v = tl.make_block_ptr((v + i_bh * T * r * V),(T, r, V ),(r * V, V, 1),
                                        (i_t * BT + i_c * BC, i_r, i_v * BV), (BC,1,BV),(2,1,0))
                p_v_new = tl.make_block_ptr(v_new + (i_bh * r + i_r)* T *  V, (T , V), (V, 1),
                                            (i_t * BT  + i_c * BC, i_v * BV), (BC , BV), (1, 0)) 
                b_k = tl.load(p_k, boundary_check=(0, 1))#BK//r,BC
                b_d = tl.load(p_d, boundary_check=(0, 1, 2))#BK
                b_v = tl.load(p_v, boundary_check=(0, 1, 2))
                b_v = tl.reshape(b_v,(BC,BV))
                b_d = tl.reshape(b_d,(BC,BK))
                b_v -= tl.dot(b_d, b_h.to(tl.bfloat16), allow_tf32=False)#ok #到这相等的 这里BC
                tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))#至少到这里第一步结果相同
                
                last_idx = min((i_t + 1) * BT, T) - 1
                b_g_last = tl.load(g + i_bh*T + last_idx)
                b_g_last = tl.exp(b_g_last)
                b_h = b_g_last * b_h

                bkv = tl.where(r_mask[:,None,None],tl.dot(tl.trans(b_k),b_v.to(b_k.dtype),allow_tf32=False)[None,:,:],0)
                b_h_cumsum += bkv.to(b_h_cumsum.dtype)
        b_h += tl.reshape(b_h_cumsum,(BK,BV))

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(final_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

#finish
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def gated_chunk_linear_attn_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    scale,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    r : tl.constexpr
):
    i_v, i_t, i_bhr = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bh = i_bhr//r
    i_r = i_bhr % r
    rk = K//r
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K//r, BK)):#这里需要注意拆分#这里K//BK = r
        #问题是不同r_block读取了同一份qk，有影响吗
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r * rk + i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r * rk + i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (V, 1), (i_r * rk + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.trans(tl.load(p_k, boundary_check=(0, 1)))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h)#, allow_tf32=False)
        b_s += tl.dot(b_q, b_k)#, allow_tf32=False)

    p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_o = b_o * tl.exp(b_g)[:,None]

    b_g_diff = b_g[:, None] - b_g[None, :]
    b_s = b_s * safe_exp(b_g_diff)#BT BT

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_s = tl.where(m_s, b_s, 0)#置为0 Bs = 0
    p_v = tl.make_block_ptr(v + i_bhr * T * V, (T, V), (V, 1), (i_t * BT , i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + (tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale
    p_o = tl.make_block_ptr(o + i_bhr * T * V, (T, V), (V,1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))




@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16)
    ],
    key=["BT", "BK"],
)
@triton.jit
def preprocess_qkw(q,
        k,
        w,
        g,
        q_new,
        k_new,
        w_new,
        T,
        H,
        K,
        r:tl.constexpr,
        BT:tl.constexpr,
        BK:tl.constexpr,
        USE_Q:tl.constexpr,
        ):
    i_k,i_bh,i_t = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_k = tl.make_block_ptr(k + i_bh*T*K, (T, K), (K, 1),     (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_w = tl.make_block_ptr(w + i_bh*T*K*r,(T,r*K),(r * K, 1),(i_t * BT, i_k * r * BK) ,(BT,r*BK),(1,0))

    p_g = tl.make_block_ptr(g+i_bh*T,(T,),(1,),(i_t*BT,),(BT,),(0,))
    p_k_new = tl.make_block_ptr(k_new + i_bh*T*K, (T, K), (K, 1),    (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_w_new = tl.make_block_ptr(w_new +i_bh*T*K*r,(T,r*K),(r * K, 1),(i_t * BT, i_k * r * BK) ,(BT,r*BK),(1,0))
    
    last_idx = min((i_t + 1) * BT, T) - 1
    b_g_last = tl.load(g + i_bh*T + last_idx).to(tl.float32) #read BT 位置

    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_w = tl.load(p_w, boundary_check=(0, 1)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    b_d_last = tl.exp((b_g_last - b_g))
    b_d_begin = tl.exp(b_g)
    b_k = b_k * b_d_last[:, None]
    b_w = b_w * b_d_begin[:, None]
    tl.store(p_k_new, b_k.to(p_k_new.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_w_new, b_w.to(p_w_new.dtype.element_ty), boundary_check=(0, 1))


    if USE_Q:
        p_q = tl.make_block_ptr(q + i_bh*T*K, (T, K), (K, 1),    (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_q_new = tl.make_block_ptr(q_new + i_bh*T*K, (T, K), (K, 1),    (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        b_q = b_q * b_d_begin[:, None]
        tl.store(p_q_new, b_q.to(p_q_new.dtype.element_ty), boundary_check=(0, 1))


#finish
def gated_chunk_fwd_h_fn(k, w, u, g, BT, initial_state, final_state):
    # k, w, u, g, BT, initial_state, final_state
    B, H, T, K, V = *k.shape,u.shape[-1]
    _,_,rT,_ = w.shape
    r = rT//T
    BK = triton.next_power_of_2(K)#直接划分好
    assert BK <= 256, "current kernel does not support head dimension larger than 256."
    BV = 16 if BK > 128 else 32
    BV = 64 if BK <= 64 else BV
    BC = 16 if BK > 128 else 32
    BC = 64 if BK <= 64 else BC
    BC = min(BT, BC)
    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1
    h = k.new_empty(B, H, NT * K, V)

    grid = (NK,B*H,NT)
    k_new = torch.empty_like(k)
    w_new = torch.empty_like(w)
    preprocess_qkw[grid](
        q=None,
        k=k,
        w=w,
        g=g,
        q_new=None,
        k_new=k_new,
        w_new=w_new,
        T=T,
        H=H,
        K=K,
        r=r,
        BT=BT,
        BK=BK,
        USE_Q=False,
    )
    grid = (NK, NV, B * H)
    v_new = torch.empty(B,H,r,T,V,dtype=u.dtype,device=u.device)#做了v_new的r_first

    gated_chunk_delta_rule_fwd_kernel_h[grid](#r没有for循环
        k_new,u,w_new, 
        v_new,g,h,
        initial_state,
        final_state,
        H=H, T=T, K=K, V=V, BT=BT, BC=BC, BK=BK, BV=BV, NT=NT,r=r,      
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
    )
    return h, v_new


#finish
def gated_chunk_fwd_o_fn(q, k, v_new,h,g,BT):
    B,H,r,T,V,K = *v_new.shape,q.shape[-1]
    BK = triton.next_power_of_2(K//r)
    o = torch.empty_like(v_new)#there_fore,bhr nT,bv
    BK = min(triton.next_power_of_2(K//r), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)
    grid = (NV, NT, B * H * r)
    #h shape b h nk v
    gated_chunk_linear_attn_fwd_kernel_o[grid](
        q, k, v_new, h, g, o,
        T*K, K, 1 ,
        r*T*V,T*V,V,
        NT*K*V,V,
        scale=K**-0.5,
        H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,r = r,
    )
    o = o.sum(dim=2)#沿着r维度求和
    return o


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def gated_fwd_prepare_dv_kernel(
    q,
    k,
    g,
    do,
    dv,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    T,
    K,
    V,
    scale,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    r: tl.constexpr,
):
    i_t, i_bhr = tl.program_id(0), tl.program_id(1)#或许也可以r并行
    i_bh = i_bhr//r
    i_r = i_bhr % r
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    block_r = K//r
    for i_k in range(tl.cdiv(block_r, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r * block_r + i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r * block_r + i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q = tl.trans(tl.load(p_q, boundary_check=(0, 1)))
        b_A += tl.dot(b_k, b_q, allow_tf32=False)
    
    p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))

    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], b_A* safe_exp(b_g[None, :] - b_g[:, None]) * scale, 0).to(do.dtype.element_ty)
    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        p_dv = tl.make_block_ptr(dv + i_bhr * s_vo_h , (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_dv = tl.dot(b_A, b_do, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

#finish
def gated_fwd_prepare_dv(q, k, g, do, r,BT):
    B, H, T, K, V = *k.shape, do.shape[-1]
    dv = torch.empty(B,H,r,T,V,device = do.device, dtype= do.dtype)#没法like
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K//r),64)
    BV = min(triton.next_power_of_2(V), 64)
    gated_fwd_prepare_dv_kernel[(NT, B*H*r)](
        q, k, g , do, dv,
        T*K, K, 1,
        T*V, V, 1,
        T, K, V, K**-0.5, BT, BK, BV, r
    )
    return dv



#finish
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def gated_chunk_delta_rule_bwd_kernel_dhu(
    q,
    k,
    d,
    g,
    do,
    dh,
    dv,
    dv2,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_h_h,
    scale,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    r: tl.constexpr,
    KR: tl.constexpr,
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)#这个不变 读取所有
    for i_t in range(NT - 1, -1, -1):# 向前偏移了一位,计算流程是对的
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V , (K, V), (V, 1), (i_k * BK , i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1)) 

        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
                                (i_k * BK, i_t * BT), (BK, BT), (0, 1))#全读取
        p_d = tl.make_block_ptr(d + i_bh * (T * K * r), (K,T*r), (1, K),
                                (i_k * BK, i_t * BT * r), (BK, BT * r), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1),
                                (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        
        last_idx = min((i_t + 1) * BT, T) - 1
        b_glast = tl.load(g + i_bh * T + last_idx)
        b_glast = tl.exp(b_glast)
        
        b_q = (tl.load(p_q, boundary_check=(0, 1)))
        b_q = (b_q * scale).to(b_q.dtype)

        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_d = (tl.load(p_d,boundary_check=(0, 1))) 
        p_dv = tl.make_block_ptr(dv + i_bh * r * T * V, (T*r, V), (V , 1),
                                (i_t * BT * r, i_v * BV), (BT*r, BV), (1, 0))#load r
        b_dv = tl.load(p_dv, boundary_check=(0, 1))#BT*r Bv 
        b_dhtrans = tl.reshape(b_dh,(r,KR,BV))
        for i_r in range(r):
            rmask = tl.arange(0, r) == i_r #第ir列
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
                                (i_t * BT , i_r*KR + i_k * BK), (BT, KR), (1, 0))#  
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dhr = tl.sum(tl.where(rmask[:,None,None],b_dhtrans,0), 0)
            dv_sum = tl.dot(b_k,b_dhr.to(b_k.dtype),allow_tf32=False)
            b_dv += tl.reshape((dv_sum[:,None,:]*rmask[None,:,None]).to(b_dv.dtype),(BT*r,BV))

        p_dv2 = tl.make_block_ptr(dv2 + i_bh * r * T * V, (T*r, V), (V , 1),
                                (i_t * BT * r, i_v * BV), (BT*r, BV), (1, 0))
        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

        b_dh *= b_glast
        b_dh += tl.dot(b_q, b_do.to(b_q.dtype), allow_tf32=False)-tl.dot(b_d,b_dv.to(b_q.dtype),allow_tf32=False)



def gated_chunk_bwd_dhu_fn(q, k, w, g,h0, do, dv, BT):
    B,H,r,T,V,K = *dv.shape,q.shape[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension being larger than 256."
    BV = 16 if BK > 128 else 32
    BV = 64 if BK <= 64 else BV

    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)#感觉可以放并行度
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'

    dh = q.new_empty(B, H, NT * K,V)#一样的#need 求和 得一起算
    q_new = torch.empty_like(q)
    k_new = torch.empty_like(k)
    w_new = torch.empty_like(w)
    # grid = (NK,)
    grid = (NK,B*H,NT)
    preprocess_qkw[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        q_new=q_new,
        k_new=k_new,
        w_new=w_new,
        T=T,
        H=H,
        K=K,
        r=r,
        BT=BT,
        BK=BK,
        USE_Q=True,
    )


    grid = (NK, NV, B * H)
    dv = rearrange(dv,'b h r t v-> b h (t r) v').contiguous()
    dv2 = torch.empty_like(dv)#一样的 #bhr T V
    gated_chunk_delta_rule_bwd_kernel_dhu[grid](
        q_new, k_new, w_new, g, do, dh, dv, dv2,
        T*K,K,1,
        NT*K*V,
        K**-0.5,
        H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,r=r,KR = K//r,
    )
    return dh, dv2

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def gated_chunk_delta_rule_bwd_kernel_dqkw(
    q,
    k,
    v,
    w,
    g,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    dw,
    dg,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    s_g_r,
    s_g_k,
    scale,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    r: tl.constexpr,
):
    i_k, i_t, i_bhr = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_r = i_bhr%r
    i_bh = i_bhr//r
    o_i = tl.arange(0, BT)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (1, K), (i_r*K//r + i_k * BK, i_t * BT), (BK, BT), (0, 1))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*K//r + i_k * BK), (BT, BK), (1, 0))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT*r,BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dg_last = tl.zeros([1,],dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bhr * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h*r, (T*r, V), (s_vo_t, s_vo_d), (i_t * BT * r, i_v * BV), (BT * r, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, (V,NT * K), (1,s_h_t), (i_v * BV,i_t * K +  i_r * K // r + i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, (V,NT * K), (1,s_h_t), (i_v * BV,i_t * K +  i_r * K // r + i_k * BK), (BV, BK), (0, 1))

        b_v  = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h  = (tl.load(p_h, boundary_check=(0, 1)))#BV BK
        b_dh = (tl.load(p_dh, boundary_check=(0, 1)))#需要额外添加r维度
        
        b_dg_last += tl.sum(b_h * b_dh) #这里是存在r求和的

        b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)#ok 
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)#d_do 全， bh应该包含 i_Kbufen
        b_dk += tl.dot(b_v, b_dh, allow_tf32=False)#用来计算dk,yes 行独立没问题
        b_dv = (tl.load(p_dv, boundary_check=(0, 1)))#BT*r BV
        b_dw += (tl.dot(b_dv.to(b_v.dtype),b_h.to(b_v.dtype))) #get BT*r BK
    
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    b_dg = tl.zeros([BT,], dtype=tl.float32)
    p_g = tl.make_block_ptr(g + i_bh * T ,(T,),(1,),(i_t*BT,),(BT,),(0,))
    b_g = tl.load(p_g,boundary_check=(0,))
    b_glast = tl.load(g +i_bh*T + (min(i_t * BT + BT, T) - 1))
    b_dg_last *= tl.exp(b_glast)


    p_w = tl.make_block_ptr(w + i_bh * T*r*K, (T*r, K), (K,1), (i_t * BT * r,i_r*K//r + i_k * BK), (BT*r ,BK), (1, 0))
    b_w = tl.load(p_w,boundary_check=(0,1))#BT * r ,BK
    b_dw = b_dw * tl.reshape(tl.broadcast_to(tl.reshape(tl.exp(b_g),(BT,1)),(BT,r)),(BT*r))[:,None]
    b_dg -= tl.sum(tl.reshape(b_w*b_dw,(BT,r*BK)),-1)

    b_dq = b_dq*scale*tl.exp(b_g)[:,None] 
    b_dg += tl.sum(b_dq*tl.trans(b_q),1)#BT*BK

    b_dk = b_dk * safe_exp(b_glast-b_g)[:,None]
    b_dg -= tl.sum(b_dk*b_k,1)#BT*BK
    b_dg_last += tl.sum(b_dk*b_k)

    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds* safe_exp(b_g[:, None] - b_g[None, :]) * scale, 0)
    b_ds2 = b_ds*(tl.dot(tl.trans(b_q),tl.trans(b_k)))

    b_dg += tl.sum(b_ds2,axis=1)
    b_dg -= tl.sum(b_ds2,axis=0)
    b_ds = b_ds.to(b_k.dtype)

    b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
    b_dk += tl.trans(tl.dot(b_q, b_ds, allow_tf32=False)) #这些应该没啥问题


    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*K//r + i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*K//r + i_k * BK), (BT, BK), (1, 0))
    p_dw = tl.make_block_ptr(dw + i_bh * T*r*K, (T*r, K), (K,1), (i_t * BT * r,i_r*K//r + i_k * BK), (BT*r ,BK), (1, 0))
    p_dg = tl.make_block_ptr(dg + i_r * s_g_r + i_k * s_g_k + i_bh * T,(T,),(1,),(i_t*BT,),(BT,),(0,))
    b_dg = tl.where(o_i<min(BT, T-i_t*BT) - 1, b_dg, b_dg + b_dg_last)
    
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dw, ((-b_dw.to(p_dw.dtype.element_ty))), boundary_check=(0, 1))
    tl.store(p_dg,b_dg.to(p_dg.dtype.element_ty),boundary_check=(0,))



def gated_chunk_bwd_dqkw_fn(q, k, v_new, w, g, h, du, do, dh, BT):
    B, H, T, K, V = *q.shape, v_new.shape[-1]
    _,_,RT,_ = w.shape
    r = RT // T
    #最后一个函数，计算dw,dq,dk
    BK = triton.next_power_of_2(K//r)#需要更细粒度的划分，确保不会使得 不同位置的划到一起
    BK = min(triton.next_power_of_2(K//r), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NK = triton.cdiv(K//r, BK)
    NT = triton.cdiv(T, BT)
    grid = (NK, NT, B * H * r)#通过NK控制位置
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)#k_org
    dw = torch.empty_like(w)#bh nt k
    dg = torch.empty(r*NK,*g.shape,dtype=torch.float32,device=g.device)

    gated_chunk_delta_rule_bwd_kernel_dqkw[grid](
        q, k, v_new, w, g, h, do, dh, dq, dk, du, dw,dg,
        T*K,K,1,
        T*V, V, 1,
        NT*K*V,V,
        B*H*T*NK,
        B*H*T,
        scale=K ** -0.5,
        H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,r = r,
    )
    dg = dg.sum(0)
    return dq.to(q.dtype), dk.to(k.dtype), dw.to(w.dtype),dg

#compute this 
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def gated_bwd_prepare_wy_repr_kernel(           
    k, v, beta,mask_ij,g_cumsum,Aw,Au,
    dw, du,
    dk, dv, dbeta,dmask,dg,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    T,
    K,
    V,
    r: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_A = tl.make_block_ptr(Aw + i_bh*T*BT*r*r ,(T*r,BT*r), (BT*r,1), (i_t * BT * r,0), (BT*r,BT*r),(1,0))
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(k.dtype.element_ty)
    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dA = tl.zeros([BT*r,BT*r], dtype=tl.float32)
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_dmask = tl.zeros([r,r],dtype=tl.float32)
    block_k = K//r
    for i_r in range(r):
        p_mask = mask_ij + tl.arange(0,r)*r + i_r#读取第ir列
        b_mask = tl.load(p_mask)#第r列
        rmask = tl.arange(0, r) == i_r #第r列
        for i_k in range(tl.cdiv(block_k, BK)):
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*block_k + i_k * BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            p_dw = tl.make_block_ptr(dw + i_bh * s_qk_h*r, (T*r, K), (s_qk_t, s_qk_d), (i_t * BT * r, i_r*block_k + i_k * BK), (BT * r, BK), (1, 0))
            b_k_beta = ((b_k * b_beta[:, None])[:,None,:]*b_mask[None,:,None]).to(b_k.dtype)#BT*r*d
            b_k_beta = tl.reshape(b_k_beta,(BT*r,BK))
            b_dw = tl.load(p_dw, boundary_check=(0, 1))
            b_dA += tl.dot(b_dw, tl.trans(b_k_beta), allow_tf32=False)
            b_dk_beta = tl.dot(tl.trans(b_A), b_dw, allow_tf32=False)
            b_dk_beta = tl.reshape(b_dk_beta,(BT,r,BK))
            sum_dk = tl.sum(b_dk_beta * b_mask[None,:,None],1)
            b_dk = sum_dk* b_beta[:, None]
            b_dbeta += tl.sum(sum_dk * b_k, 1)
            b_ss = (tl.sum(tl.sum(b_dk_beta * b_beta[:,None,None] * b_k[:,None,:],0),-1))
            b_dmask += (b_ss[:,None]*rmask[None,:]).to(tl.float32)
            p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*block_k + i_k * BK), (BT, BK), (1, 0))
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    
    i = tl.arange(0, BT * r)[:, None]
    j = tl.arange(0, BT * r)[None, :]
    iB = i // r
    jB = j // r
    da_mask = iB > jB
    b_dA = tl.where(da_mask, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), tl.trans(b_A), allow_tf32=False)
    b_dA = tl.dot(tl.trans(b_A), b_dA.to(b_A.dtype), allow_tf32=False)
    b_dA = tl.where(da_mask, -b_dA, 0) #等价于 kkt的 dA 很多0，对角处
    b_dA  = tl.reshape(b_dA,(BT,r,BT,r))


    p_A = tl.make_block_ptr(Au + i_bh*T*BT*r*r ,(T*r,BT*r), (BT*r,1), (i_t * BT * r,0), (BT*r,BT*r),(1,0))
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(k.dtype.element_ty)
    b_dA2 = tl.zeros([BT*r,BT*r], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):#分块r 
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(du + i_bh * s_vo_h * r, (T * r, V), (s_vo_t, s_vo_d), (i_t * BT * r, i_v * BV), (BT * r, BV), (1, 0))#r*BT BV
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v_beta = ((b_v * b_beta[:, None])[:,None,:]*tl.full([r],1, dtype=b_v.dtype)[None,:,None]).to(b_v.dtype)##BT*r*BV
        b_v_beta = tl.reshape(b_v_beta,(BT*r,BV))
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA2 += tl.dot(b_du, tl.trans(b_v_beta), allow_tf32=False)#BT*r,BT*r
        b_dv_beta = tl.dot(tl.trans(b_A), b_du, allow_tf32=False)#BT*r,BV
        b_dv_beta = tl.reshape(b_dv_beta,(BT,r,BV))#
        sum_dv = tl.sum(b_dv_beta,-2)#这里不一样，结果
        b_dv = (sum_dv * b_beta[:, None])#？哪一步结果不一样呢
        b_dbeta += tl.sum(sum_dv * b_v, 1)
        p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    
    b_dA2 = tl.where(da_mask, b_dA2, 0)
    b_dA2 = tl.dot(b_dA2.to(b_A.dtype), tl.trans(b_A), allow_tf32=False)
    b_dA2 = tl.dot(tl.trans(b_A), b_dA2.to(b_A.dtype), allow_tf32=False)
    b_dA2 = tl.where(da_mask, -b_dA2, 0) #等价于 kkt的 dA 很多0，对角处
    b_dA2 = tl.reshape(b_dA2,(BT,r,BT,r))


    p_g = tl.make_block_ptr(g_cumsum + i_bh*T,(T,),(1,),(i_t*BT,),(BT,),(0,))
    b_g = tl.load(p_g,boundary_check=(0,))
    b_dA2 *= safe_exp(b_g[:,None]-b_g[None,:])[:,None,:,None]
    b_dA += b_dA2
    b_dA2 = tl.permute(b_dA2,(0,2,1,3))#Bt bt r r

    b_A = tl.zeros([BT,BT,r,r], dtype=tl.float32)
    
    for i_r in range(r):#只取ir项 
        p_mask = mask_ij + tl.arange(0,r)*r+i_r#读取第ir列
        b_mask = tl.load(p_mask)#第ir列
        rmask = tl.arange(0, r) == i_r #第ir列
        g = tl.sum(tl.where(rmask[None,None,None,:], b_dA, 0), -1)#BT r BT #取出第ir列
        ir_A = tl.sum(g * b_mask[None,:,None],1).to(k.dtype.element_ty)#BT BT

        for i_k in range(tl.cdiv(block_k, BK)):#ik = 1
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*block_k + i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*block_k + i_k * BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dk = tl.load(p_dk, boundary_check=(0, 1))
            b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)#BT*BK

            b_dk_beta = tl.dot(ir_A, b_k, allow_tf32=False)
            b_dbeta += tl.sum(b_dk_beta * b_k, 1)
            b_dk += tl.dot(tl.trans(ir_A), b_k_beta, allow_tf32=False)
            b_dk += b_dk_beta * b_beta[:, None]
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

            beta_kkt = (tl.dot(b_k_beta,tl.trans(b_k), allow_tf32=False))#BT BT
            b_A += beta_kkt[:,:,None,None] * ((rmask[None,:] * b_mask[:,None])[None,None,:,:])#这列全广播了不对

            betas = tl.sum(tl.sum(beta_kkt[:,None,:]*g,-1),0)
            b_dmask +=  (betas[:,None]*rmask[None,:]).to(tl.float32)


    p_dbeta = tl.make_block_ptr(dbeta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty), boundary_check=(0,))
    
    p_dmask = tl.make_block_ptr(dmask + (i_bh * (T//BT) + i_t)* r * r , (r,r), (r,1), (0,0), (r,r), (1,0))
    tl.store(p_dmask, b_dmask.to(p_dmask.dtype.element_ty), boundary_check=(0,1))

    b_dA2 *= b_A #BT BT r r
    b_dA2 = tl.sum(tl.reshape(b_dA2,(BT,BT,r*r)),-1)

    b_dg = tl.sum(b_dA2,1)-tl.sum(b_dA2,0)
    p_dg = tl.make_block_ptr(dg+i_bh*T,(T,),(1,),(i_t*BT,),(BT,),(0,))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))



def gated_bwd_prepare_wy_repr(k, v, beta, mask,g, Aw,Au, dw, du, BT):
    B, H, T, K, V = *k.shape, v.shape[-1]
    r = mask.shape[-1]
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K//r), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT = triton.cdiv(T, BT)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v).contiguous()
    dbeta = torch.zeros_like(beta)
    dg = torch.empty_like(g)
    dmask = torch.zeros([B*H*NT,r,r],device=k.device,dtype=k.dtype).contiguous()
    assert BK <= K//r
    gated_bwd_prepare_wy_repr_kernel[(NT, B*H)](
        k, v, beta, mask, g, Aw,Au,
        dw, du,
        dk, dv, dbeta,dmask,dg,
        T*K, K, 1,
        T*V, V, 1,
        T, K, V, r, BT, BK, BV
    )
    dmask = dmask.sum(0)
    return dk, dv, dbeta, dmask,dg


class gated_ChunkDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, beta,g,mask,BT, initial_state, output_final_state=False, checkpoint_level=1):
        B,H,L,K = q.shape
        g = chunk_local_cumsum(g,BT,head_first=True,output_dtype=torch.float)
        Aw,Au = gated_chunk_scaled_dot_kkt_fwd(k=k,beta=beta,g_cumsum=g,mask=mask,BT=BT,output_dtype=torch.float32)
        
        Aw = solve_tril(A=Aw,mask=mask,k=k,BT=BT,output_dtype=k.dtype)
        Au = solve_tril(A=Au,mask=mask,k=k,BT=BT,output_dtype=k.dtype)
        #到这里应该没啥问题
        r = mask.shape[-1]
        w, u = gated_fwd_recompute_w_u(k, v, beta, mask,Aw,Au,BT)#

        final_state = None
        if output_final_state:
            final_state = q.new_empty(q.shape[0], q.shape[1], q.shape[-1], v.shape[-1],
                                      dtype=torch.float32, requires_grad=False)#这部分不需要修正
        h, v_new = gated_chunk_fwd_h_fn(k, w, u, g, BT, initial_state, final_state)#need change'
        #final_state almost 一致
        o = gated_chunk_fwd_o_fn(q, k, v_new, h, g, BT)#need change
        if checkpoint_level == 1:
            h, v_new = None, None #这里重新计算了？
        ctx.save_for_backward(q, k, v, beta,g, mask, Aw, Au , h, v_new, initial_state)
        ctx.BT = BT
        return o.to(q.dtype), final_state
    
    
    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, d_ht=None):
        q, k, v, beta, g, mask , Aw,Au, h, v_new, initial_state = ctx.saved_tensors
        BT = ctx.BT
        r = mask.shape[-1]
        start = time.time()
        w, u = gated_fwd_recompute_w_u(k, v, beta, mask, Aw,Au,BT)#跳过
        end = time.time()
        print('recompute_wu:',end-start)
        if h is None:
            h, v_new = gated_chunk_fwd_h_fn(k, w, u, g, BT, initial_state, None)
        start = time.time() 

        #从这里开始重新书写计算代码
        dv = gated_fwd_prepare_dv(q, k, g, do, r, BT)#qk do v_new#因此这个dv应该是一个w的shape finish
        end = time.time()
        print('pre:',end-start)
        #dv BHR T V
        
        start = time.time()
        dh, dv = gated_chunk_bwd_dhu_fn(q, k, w, g,initial_state,do, dv, BT)#new_dv dh #final for wyper dv
        end = time.time()
        print('chunk_bwd_dhu_fn:',end-start)
        
        start = time.time()
        dq, dk, dw , dg = gated_chunk_bwd_dqkw_fn(q, k, v_new, w, g, h, dv, do, dh, BT)#这一步也巨慢
        end = time.time()
        print('chunk_bwd_dqkw_fn:',end-start)
        #仅仅两个dg位置可能出错，别的不会

        start = time.time()
        dk2, dv, dbeta,dmask,dg2 = gated_bwd_prepare_wy_repr(k, v, beta, mask,g, Aw,Au, dw, dv, BT)#只有这里带mask
        dk.add_(dk2)
        dg.add_(dg2)
        end = time.time()
        print('bwd_prepare_wy_repr:',end-start)
        #仅仅两个dg位置可能出错，别的不会
        dg = chunk_local_cumsum(dg, BT, reverse=True,head_first=True,output_dtype=torch.float)
        
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dbeta.to(beta.dtype),dg,dmask.to(mask.dtype),None, None, None


def mask_gated_chunk_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    mask: torch.Tensor,#use for mask org_tensor 
    BT: int,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
):
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "FusedChunkDeltaRuleFunction does not support float32. Please use bfloat16."
    o, final_state = gated_ChunkDeltaRuleFunction.apply(q, k, v, beta,g,mask, BT, initial_state, output_final_state)
    return o, final_state


def delta_rule_recurrence(q, k, v, beta,g, mask,initial_state=None):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    r = mask.shape[-1]
    o = torch.zeros_like(v)
    if initial_state == None:
        S = torch.zeros(b, h, d_k, d_v,device=k.device,dtype=torch.float32)
    else:
        S = initial_state
    q = q * (d_k ** -0.5)
    if beta.ndim < v.ndim:
        beta = beta[..., None]
    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].clone()
        beta_i = beta[:, :, i]
        _v = _v * beta_i
        kkt = torch.einsum('b h d,b h v->b h d v',_k*beta_i,_k)
        kkt = rearrange(kkt,' b h (r d) (l v)-> b h r d l v',r= r,l=r)
        kkt = torch.einsum('b h r d l v,r l->b h r d l v',kkt,mask.to(kkt))
        kkt = rearrange(kkt,'b h r d l v-> b h (r d) (l v)')
        iplr = torch.eye(d_k).to(q)-kkt
        iplr = torch.einsum(' b h q k ,b h->b h q k',iplr,g[:,:,i])
        S = torch.einsum(' b h q k ,b h k v->b h q v',iplr.float(),S.clone()) + _k.unsqueeze(-1).float() * _v.unsqueeze(-2).float()
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q.float(), S).to(k.dtype)
    return o,S


if __name__ =="__main__":
    import sys
    import time
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(42)  

    # for i in range(200):
    B = 16
    H = 4
    L = 128
    DK = 256
    DV = 256
    r = 4
    q = (torch.randn(B, H, L, DK)).cuda().requires_grad_(True)
    k = (torch.randn(B, H, L, DK)).cuda()
    k = torch.nn.functional.normalize(k, dim=-1, p=2).requires_grad_(True)
    v = (torch.randn(B, H, L, DV)).cuda().requires_grad_(True)
    beta = torch.randn(B, H, L).cuda().sigmoid().requires_grad_(True)
    mask = torch.randn([r,r])
    mask = mask.cuda().requires_grad_(True).contiguous()

    # mask = torch.ones([2,2])
    # mask = mask.cuda().requires_grad_(True).contiguous()

    g = torch.nn.functional.logsigmoid(torch.randn(B, H, L).cuda()).requires_grad_(True)
    g_exp = (torch.exp(g))

    do = torch.randn(B, H, L, DV).cuda()
    o1,ss = delta_rule_recurrence(q,k,v,beta,g_exp,mask)
    o1.backward(do, retain_graph=True)
    q_grad, q.grad = q.grad, None
    k_grad, k.grad = k.grad, None
    v_grad, v.grad = v.grad, None
    mask_grad, mask.grad = mask.grad, None
    beta_grad, beta.grad = beta.grad, None
    g_grad, g.grad = g.grad, None
    # end = time.time()
    # print(end-start)
    # start = time.time()
    # w, u, A = fwd_prepare_wy_repr(k, v,beta, mask, 64)
    # o,f_state = mask_gated_chunk_delta_rule(q, k, v,beta,g,mask,BT=32,output_final_state=True)
    # o2,f_state = mask_chunk_delta_rule(q, k, v,beta,mask,BT=32)

    # qh,kh,vh,betah,gh = map(lambda x: rearrange(x, 'b h l ... -> b l h ...'), (q, k, v, beta, g))
    # o,f_state = chunk_gated_delta_rule(qh,kh,vh,gh,(betah*rearrange(mask,'c r-> (c r)')).contiguous(),use_qk_l2norm_in_kernel=False,output_final_state=True)
    # o = rearrange(o,'b l h d->b h l d')
    o,f_state = mask_gated_chunk_delta_rule(q, k, v,beta,g,mask,BT=32,output_final_state=True)
    o.backward(do,retain_graph=True)
    q_grad0, q.grad = q.grad, None
    k_grad0, k.grad = k.grad, None
    v_grad0, v.grad = v.grad, None
    beta_grad0, beta.grad = beta.grad, None
    mask_grad0, mask.grad = mask.grad, None
    g_grad0, g.grad = g.grad, None
    print((o1-o).abs().max())
    print((f_state-ss).abs().max())
    print((q_grad-q_grad0).abs().max())
    print((k_grad-k_grad0).abs().max())#计算结果差距大 差距到1
    print((v_grad-v_grad0).abs().max())
    print((beta_grad-beta_grad0).abs().max())
    print((mask_grad-mask_grad0).abs().max())
    
    print((g_grad-g_grad0).abs().max())
    print(mask_grad)
    print(mask_grad0)
    

    # o2,f_state2 = mask_gated_chunk_delta_rule(q, k, v,beta,g,mask,BT=32,output_final_state=True)
    # o2.backward(do,retain_graph=True)
    # q_grad2, q.grad = q.grad, None
    # k_grad2, k.grad = k.grad, None
    # v_grad2, v.grad = v.grad, None
    # beta_grad2, beta.grad = beta.grad, None
    # mask_grad2, mask.grad = mask.grad, None

    # print((o-o2).abs().max())
    # print((f_state-f_state2).abs().max())

    # print((q_grad2-q_grad0).abs().max())
    # print((k_grad2-k_grad0).abs().max())#计算结果差距大 差距到1
    # print((v_grad2-v_grad0).abs().max())
    # print((beta_grad2-beta_grad0).abs().max())
    # print((mask_grad2-mask_grad0).abs().max())
    # print('naive:',mask_grad2)
    # print('triton:',mask_grad0)
    # print(k_grad2)
    # print(k_grad0)
    

    # BT = 16
    # w, u, A = fwd_prepare_wy_repr(k, v,beta, mask, BT)#compute for A matrix #compute all
    # print('finish0')
    # h, v_new = chunk_fwd_h_fn(k, w, u, BT, None, None)#need change'
    # print('finish1')
    # o = chunk_fwd_o_fn(q, k, v_new, h, BT)#need change
    # print('finish2')
    # w, u = fwd_recompute_w_u(k, v, beta, mask, A, BT)#跳过
    # print('finish3')
    # dv = fwd_prepare_dv(q, k, do, r, BT)#qk do v_new#因此这个dv应该是一个w的shape finish
    # print('finish4')
    # dh, dv = chunk_bwd_dhu_fn(q, k, w, do, dv, BT)#new_dv dh #final for wyper dv
    # print('finish5')
    # dq, dk, dw = chunk_bwd_dqkw_fn(q, k, v_new, w, h, dv, do, dh, BT)#这一步也巨慢
    # print('finish6')    
    
    # Ass = rearrange(A,'b h (n t) l->b h n t l',n = L//BT)
    # dwss = rearrange(dw,'b h (n t) k->b h n t k',n = L//BT)
    # dvss = rearrange(dv,'b h (n t) k->b h n t k',n = L//BT)
    # dk2, dv, dbeta,dmask = bwd_prepare_wy_repr(k, v, beta, mask, A, dw, dv, BT)
    # print('triton:',dmask) #几乎完全相等

    # vbeta = v*beta[...,None]
    # vbeta = rearrange(vbeta,'b h (n T) d->b h n T d',T=BT)
    # vbeta = vbeta.unsqueeze(-2).expand(-1,-1,-1,-1,r,-1)
    # vbeta = rearrange(vbeta,'b h n t r d-> b h n (t r) d')

    # kbeta = k*beta[...,None]
    # kbeta = rearrange(kbeta,'b h (n T) (r d)->b h n T r d',T=BT,r=r)
    # kbeta = torch.einsum('b h n T r d,c r-> b h n T c r d',kbeta,mask)
    # kbeta = rearrange(kbeta,'b h n t c r d-> b h n (t c) (r d)')
    # dA = dvss@vbeta.transpose(-1,-2)+dwss@kbeta.transpose(-1,-2) 

    
    # dorg = Ass.transpose(-1,-2)@dwss#bhn bt*r k
    # dorg = rearrange(dorg,'b h n (t r) (c k)->b h n t r c k',r=r,c=r)
    # betan = rearrange(beta,'b h (n t)->b h n t',n=L//BT)
    # kn = rearrange(k,'b h (n t) (r d)->b h n t r d ',n = L//BT,r=r)

    # dmask = torch.einsum('b h n t r c k,b h n t->b h n t r c k',dorg,betan)
    # dmask = torch.einsum('b h n t r c k,b h n t c k->b h n t r c k',dmask,kn)
    # dmask = rearrange(dmask,'b h n t r c k-> (b h n) (t k) r c')
    # dmaskss = dmask.sum(0).sum(0)

    # i = torch.arange(0, BT * r)[:, None]
    # j = torch.arange(0, BT * r)[None, :]
    # iB = i // r
    # jB = j // r
    # da_mask = iB > jB
    # da_mask = da_mask.cuda()
    # b_dA = torch.where(da_mask, dA, 0)

    # b_dA = b_dA @ Ass.transpose(-1,-2)
    # b_dA = Ass.transpose(-1,-2)@b_dA

    # b_dA = torch.where(da_mask, -b_dA, 0) #等价于 kkt的 dA 很多0，对角处
    # b_dA = rearrange(b_dA,'b h n (t r) (l c)-> b h n t r l c',c=r,r=r)
    # # print((dAss-b_dA).abs())#到这里也完全相等


    # # betakkt = k*beta[...,None]
    # kbeta = k*beta[...,None]
    # kbeta = rearrange(kbeta,'b h (n T) (r d)->b h n T r d',T=BT,r=r)
    # kbeta2 = rearrange(k,'b h (n T) (r d)->b h n T r d',T=BT,r=r)
    # betakkt = torch.einsum('b h n T r d,b h n s r d->b h n r T s',kbeta,kbeta2)#r  Bt bt
    # betakkt = rearrange(betakkt,'b h n r T s->b h n T s r')#BT r BT###横向
    # # print((dAss-b_dA).abs())

    # #证明是下面的计算出错了
    # dmask = torch.einsum('b h n t r l c,b h n t l c-> b h n t r l c',b_dA,betakkt)
    # # print((dAss-dmask).abs().max())#意味着这个计算结果也是对的
    # # print((dAss-dmask))

    # dmask = rearrange(dmask,'b h n t r l c->b h n (t l) r c')
    # dmask = dmask.sum(-3)
    # dmask = dmask.sum(0).sum(0).sum(0)
    # print('matrix:',dmask)



    




