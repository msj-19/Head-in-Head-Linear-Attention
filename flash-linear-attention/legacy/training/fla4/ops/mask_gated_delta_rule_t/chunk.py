# -*- coding: utf-8 -*-
# Copyright (c) 2023, Yu Zhang, Songlin Yang
import time
import torch
import triton
import triton.language as tl
from einops import rearrange
from fla.utils import autocast_custom_bwd, autocast_custom_fwd,contiguous
from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.op import exp
import torch.nn.functional as F
from typing import Optional
import warnings
from typing import Optional
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla4.utils import autotune_cache_kwargs, is_nvidia_hopper, use_cuda_graph

@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float('-inf')))

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V','r','BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def gated_fwd_recompute_w_u_kernel(
    k,
    v,
    beta,
    mask,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    r: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
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
    dk = K//r
    p_beta = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    p_A = tl.make_block_ptr(A + (bos*r*H + i_h) * BT * r, (T*r, BT*r), (H*BT*r, 1), (i_t * BT * r, 0), (BT * r, BT * r), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos*r*H + i_h) * V, (T*r, V), (H*V, 1), (i_t * BT * r, i_v * BV), (BT*r, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)[:,None,:]*tl.full([r],1, dtype=b_v.dtype)[None,:,None]
        b_vb = tl.reshape(b_vb,(BT*r,BV))
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, (b_u).to(p_u.dtype.element_ty), boundary_check=(0, 1))

    if USE_G:
        p_g = tl.make_block_ptr(g + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = exp(tl.load(p_g, boundary_check=(0,)))
    
    for i_r in range(r):
        p_mask = tl.make_block_ptr(mask + (bos*H + i_h)*r*r, (T,r,r),(H*r*r,r,1), (i_t*BT,0,i_r),(BT,r,1),(2,1,0))
        b_mask = tl.load(p_mask,boundary_check=(0,1,2))#BT,r,1
        for i_k in range(tl.cdiv(dk, BK)):
            p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T,K), (H*K,1), (i_t * BT, i_r*dk + i_k * BK), (BT,BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_kbm = b_k*b_beta[:, None] ###BT BK
            if USE_G:
                b_kbm *= b_g[:, None] ###get BT r BK
            b_kbm = (b_kbm[:,None,:]*b_mask).to(b_k.dtype)#BT r BK
            b_kbm = tl.reshape(b_kbm,(BT*r,BK)) 
            b_w = tl.dot(b_A,b_kbm).to(b_k.dtype)#r BT*r BK
            p_w = tl.make_block_ptr(w + (bos*r*H + i_h) * K, (T*r,K), (H*K,1), (i_t * BT * r,i_r*dk + i_k * BK), (BT*r,BK), (1, 0))
            tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'BT', 'IS_VARLEN','r'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def gated_chunk_scaled_dot_kkt_fwd_kernel(        
    k,
    beta,
    g,
    mask,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    K: tl.constexpr,
    H: tl.constexpr,
    r:  tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_b = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))
    p_mask = tl.make_block_ptr(mask + (bos*H + i_h)*r*r, (T,r,r),(H*r*r,r,1), (i_t*BT,0,0),(BT,r,r),(2,1,0))
    b_mask = tl.load(p_mask,boundary_check=(0,1,2))#BT,r,r
    dk = K//r
    b_A0 = tl.zeros([r,BT,BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(dk, BK)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, r,dk), (H*K,dk,1), (i_t * BT, 0, i_k * BK), (BT,r,BK), (2, 1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1, 2))#load BT r BK
        b_k = tl.permute(b_k,(1,0,2))#r,BT,BK
        b_ktrans =  tl.permute(b_k,(0,2,1))#r,BK,BT
        b_A0 += tl.dot(b_k, b_ktrans)###get r BT BT
    b_A0 = tl.permute(b_A0,(1,2,0))#bt bt r    
    b_A = b_A0[:,:,None,:]*b_mask[:,None,:,:]#BT BT r r
    b_A = tl.where((tl.arange(0, BT)[:,None] > tl.arange(0, BT)[None,:])[:,:,None,None], b_A, 0)

    if USE_G:
        p_g = tl.make_block_ptr(g + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A *= exp(b_g_diff)[:,:,None,None]
    b_A *= b_b[:, None][:,:,None,None]

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A[:,:,None,None], b_A, 0)
    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT*r*r, (T, BT,r,r),(BT*H*r*r,r*r,r,1), (i_t * BT,0,0,0), (BT, BT,r,r), (3,2,1,0))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1,2,3))



@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'BT', 'IS_VARLEN','r'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def solve_tril_16x16_kernel_org(
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H:  tl.constexpr,
    r:  tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)###等价放长了i-t 此时原始句子已经看成 B T*r H BT*r的结果
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T * r, i_b * T * r + T * r

    A = A + (bos * H + i_h) * BT * r 
    Ad = Ad + (bos * H + i_h) * 16 #return B T*r H 16

    offset = (i_t * 16) % (BT * r)
    p_A = tl.make_block_ptr(A, (T*r,BT*r),(H*BT*r,1), (i_t * 16, offset), (16, 16), (1,0))
    p_Ad = tl.make_block_ptr(Ad, (T*r,16),(H*16,1),(i_t*16,0),(16,16),(1,0))
    ####Ad位置需要调整
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    b_A = -tl.where((tl.arange(0, 16)[:, None] > tl.arange(0, 16)[None, :]), b_A, 0)
    o_i = tl.arange(0, 16)
    for i in range(r, min(16, T * r - i_t * 16)):#避免超出范围
        b_a = -tl.load(A + (i_t * 16 + i) * H * BT * r + offset + o_i)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        mask = o_i == i
        b_A = tl.where(mask[:, None], b_a, b_A)
    b_A += o_i[:, None] == o_i[None, :]
    tl.store(p_Ad, b_A.to(p_Ad.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'BT', 'IS_VARLEN','r'],
)
@triton.jit(do_not_specialize=['T'])
def merge_r1_to_r2_inverse_kernel(
        A,
        Ad,
        Ai,
        cu_seqlens,
        chunk_indices,
        T,
        r: tl.constexpr,
        H: tl.constexpr,
        BT: tl.constexpr,
        IS_VARLEN: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    offset = ((i_t*16) % BT) *r


    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos*r*H + i_h) * BT * r
    Ad += (bos*r*H + i_h) * 16
    Ai += (bos*r*H + i_h) * 16 * r

    p_A21 = tl.make_block_ptr(A,  (T*r,BT*r),(H*BT*r,1) ,(i_t * 16 * r + 16, offset), (16, 16), (1,0))
    b_A21 = tl.load(p_A21, boundary_check=(0,1)).to(tl.float32)

    p_Ad11  = tl.make_block_ptr(Ad,(T*r,16),(H*16,1), (i_t * 16 * r, 0), (16,16), (1,0))
    p_Ad22  = tl.make_block_ptr(Ad,(T*r,16),(H*16,1), (i_t * 16 * r +16 , 0), (16,16), (1,0))

    p_Ai11 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 * r,     0), (16, 16), (1, 0))
    p_Ai22 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 * r +16,16), (16, 16), (1, 0))
    p_Ai21 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 * r +16, 0), (16, 16), (1, 0))

    Ai11 = tl.load(p_Ad11, boundary_check=(0, 1)).to(tl.float32)
    Ai22 = tl.load(p_Ad22, boundary_check=(0, 1)).to(tl.float32)
    Ai21 = -tl.dot(tl.dot(Ai22,b_A21, input_precision='ieee'),Ai11,input_precision='ieee')
    tl.store(p_Ai11,Ai11.to(p_Ai11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai22,Ai22.to(p_Ai22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai21,Ai21.to(p_Ai21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))



@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'BT', 'IS_VARLEN','r'],
)
@triton.jit(do_not_specialize=['T'])
def merge_r1_to_r4_inverse_kernel(
        A,
        Ad,
        Ai,
        cu_seqlens,
        chunk_indices,
        T,
        r: tl.constexpr,
        H: tl.constexpr,
        BT: tl.constexpr,
        IS_VARLEN: tl.constexpr 
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    offset = ((i_t*16*r) % (BT*r)) 
    A  +=  (bos * r * H +i_h) * BT * r
    Ai += (bos * r * H +i_h) * 16 * r
    Ad += (bos * r * H +i_h) * 16


    p_A21 = tl.make_block_ptr(A, (T*r,BT*r),(H*BT*r,1) ,(i_t * 16 *r +16, offset),    (16, 16), (1,0))
    p_A31 = tl.make_block_ptr(A, (T*r,BT*r),(H*BT*r,1) ,(i_t * 16 *r +32, offset),    (16, 16), (1,0))
    p_A32 = tl.make_block_ptr(A, (T*r,BT*r),(H*BT*r,1) ,(i_t * 16 *r +32, offset+16), (16, 16), (1,0))
    p_A41 = tl.make_block_ptr(A, (T*r,BT*r),(H*BT*r,1) ,(i_t * 16 *r +48, offset),    (16, 16), (1,0))
    p_A42 = tl.make_block_ptr(A, (T*r,BT*r),(H*BT*r,1) ,(i_t * 16 *r +48, offset+16), (16, 16), (1,0))
    p_A43 = tl.make_block_ptr(A, (T*r,BT*r),(H*BT*r,1) ,(i_t * 16 *r +48, offset+32), (16, 16), (1,0))
    
    b_A21 = tl.load(p_A21, boundary_check=(0,1)).to(tl.float32)
    b_A31 = tl.load(p_A31, boundary_check=(0,1)).to(tl.float32)
    b_A32 = tl.load(p_A32, boundary_check=(0,1)).to(tl.float32)
    b_A41 = tl.load(p_A41, boundary_check=(0,1)).to(tl.float32)
    b_A42 = tl.load(p_A42, boundary_check=(0,1)).to(tl.float32)
    b_A43 = tl.load(p_A43, boundary_check=(0,1)).to(tl.float32)


    p_Ad11  = tl.make_block_ptr(Ad ,(T*r,16),(H*16,1), (i_t * 16 *r    , 0), (16,16), (1,0))
    p_Ad22  = tl.make_block_ptr(Ad ,(T*r,16),(H*16,1), (i_t * 16 *r +16, 0), (16,16), (1,0))
    p_Ad33  = tl.make_block_ptr(Ad ,(T*r,16),(H*16,1), (i_t * 16 *r +32, 0), (16,16), (1,0))
    p_Ad44  = tl.make_block_ptr(Ad ,(T*r,16),(H*16,1), (i_t * 16 *r +48, 0), (16,16), (1,0))
    ###这里是对的


    p_Ai11 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r, 0),     (16, 16), (1, 0))
    p_Ai22 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+16, 16), (16, 16), (1, 0))
    p_Ai33 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+32, 32), (16, 16), (1, 0))
    p_Ai44 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+48, 48), (16, 16), (1, 0))
    
    p_Ai21 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+16, 0),  (16, 16), (1, 0))
    p_Ai31 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+32, 0),  (16, 16), (1, 0))
    p_Ai32 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+32, 16), (16, 16), (1, 0))
    p_Ai41 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+48 ,0),  (16, 16), (1, 0))
    p_Ai42 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+48, 16), (16, 16), (1, 0))
    p_Ai43 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+48, 32), (16, 16), (1, 0))


    Ai11 = tl.load(p_Ad11, boundary_check=(0, 1)).to(tl.float32)
    Ai22 = tl.load(p_Ad22, boundary_check=(0, 1)).to(tl.float32)
    Ai33 = tl.load(p_Ad33, boundary_check=(0, 1)).to(tl.float32)
    Ai44 = tl.load(p_Ad44, boundary_check=(0, 1)).to(tl.float32)####这里计算应该是对的


    Ai21 = -tl.dot(tl.dot(Ai22,b_A21, input_precision='ieee'),Ai11,input_precision='ieee')
    Ai32 = -tl.dot(tl.dot(Ai33,b_A32, input_precision='ieee'),Ai22,input_precision='ieee')
    Ai43 = -tl.dot(tl.dot(Ai44,b_A43, input_precision='ieee'),Ai33,input_precision='ieee')

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

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'BT', 'IS_VARLEN','r'],
)
@triton.jit
def merge_r4_to_r8_inverse_kernel(
        A,###B H T 8 BT 8
        Ad,###B H T 8 16 8
        Ai,###B H T 8 16 4
        cu_seqlens,
        chunk_indices,
        T,
        r: tl.constexpr,
        H: tl.constexpr,
        BT: tl.constexpr,
        IS_VARLEN: tl.constexpr 
):

    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    offset = ((i_t*16*8) % (BT*8)) 
    A  +=  (bos * r * H +i_h) * BT * r
    Ai += (bos * r * H +i_h) * 16 * 8
    Ad += (bos * r * H +i_h) * 16 * 4

    p_A21 = tl.make_block_ptr(A,   (T*r,BT*r),(H*BT*r,1) ,    (i_t * 16 * 8 + 64, offset), (64, 64), (1,0))
    b_A21 = tl.load(p_A21, boundary_check=(0,1)).to(tl.float32)

    p_Ad11  = tl.make_block_ptr(Ad ,(T*r,16*4),(H*16*4,1),  (i_t * 16 * 8,  0),      (64,64), (1,0))
    p_Ad22  = tl.make_block_ptr(Ad ,(T*r,16*4),(H*16*4,1),  (i_t * 16 * 8 + 64 , 0), (64,64), (1,0))

    p_Ai11 = tl.make_block_ptr(Ai, (T*r,16*8), (H*16*8, 1), (i_t * 16 * 8,     0),   (64, 64), (1, 0))
    p_Ai22 = tl.make_block_ptr(Ai, (T*r,16*8), (H*16*8, 1), (i_t * 16 * 8 +64,64),   (64, 64), (1, 0))
    p_Ai21 = tl.make_block_ptr(Ai, (T*r,16*8), (H*16*8, 1), (i_t * 16 * 8 +64, 0),   (64, 64), (1, 0))

    Ai11 = tl.load(p_Ad11, boundary_check=(0, 1)).to(tl.float32)
    Ai22 = tl.load(p_Ad22, boundary_check=(0, 1)).to(tl.float32)
    Ai21 = -tl.dot(tl.dot(Ai22,b_A21, input_precision='ieee'),Ai11,input_precision='ieee')
    tl.store(p_Ai11,Ai11.to(p_Ai11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai22,Ai22.to(p_Ai22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai21,Ai21.to(p_Ai21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))




@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'BT', 'IS_VARLEN','r'],
)
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_32x32_inverse_kernel(
        A,
        Ad,
        Ai,
        cu_seqlens,
        chunk_indices,
        T,
        r: tl.constexpr,
        H: tl.constexpr,
        BT: tl.constexpr,
        IS_VARLEN: tl.constexpr 
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos * r* H + i_h) * BT * r
    Ai += (bos *r * H + i_h) * BT * r
    Ad += (bos *r * H + i_h) * 16 * r

    p_A21 = tl.make_block_ptr(A, (T*r,32*r),(H*32*r,1) ,((i_t * 32 + 16) *r, 0), (16*r, 16*r), (1,0))
    b_A21 = tl.load(p_A21, boundary_check=(0,1)).to(tl.float32)

    p_Ad11  = tl.make_block_ptr(Ad,(T*r,16*r),(H*16*r,1), (i_t * 32 * r, 0), (16*r,16*r), (1,0))
    p_Ad22  = tl.make_block_ptr(Ad,(T*r,16*r),(H*16*r,1), ((i_t *32 +16) * r, 0), (16*r,16*r), (1,0))

    p_Ai11 = tl.make_block_ptr(Ai, (T*r,32*r), (H*32*r, 1), (i_t * 32 * r , 0), (16*r, 16*r), (1, 0))
    p_Ai22 = tl.make_block_ptr(Ai, (T*r,32*r), (H*32*r, 1), ((i_t * 32 + 16) * r , 16*r), (16*r, 16*r), (1, 0))
    p_Ai21 = tl.make_block_ptr(Ai, (T*r,32*r), (H*32*r, 1), ((i_t * 32 + 16) * r, 0), (16*r, 16*r), (1, 0))

    Ai11 = tl.load(p_Ad11, boundary_check=(0, 1)).to(tl.float32)
    Ai22 = tl.load(p_Ad22, boundary_check=(0, 1)).to(tl.float32)
    Ai21 = -tl.dot(tl.dot(Ai22,b_A21, input_precision='ieee'),Ai11,input_precision='ieee')
    tl.store(p_Ai11,Ai11.to(p_Ai11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai22,Ai22.to(p_Ai22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai21,Ai21.to(p_Ai21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'BT', 'IS_VARLEN','r'],
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_64x64_inverse_kernel(
        A,
        Ad,
        Ai,
        cu_seqlens,
        chunk_indices,
        T,
        r: tl.constexpr,
        H: tl.constexpr,
        BT: tl.constexpr,
        IS_VARLEN: tl.constexpr 
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos * r* H + i_h) * BT * r
    Ai += (bos *r * H + i_h) * BT * r
    Ad += (bos *r * H + i_h) * 16 * r

    p_A21 = tl.make_block_ptr(A , (T*r,64*r),(H*64*r,1) ,((i_t * 64 + 16) *r, 0), (16*r, 16*r), (1,0))
    p_A31 = tl.make_block_ptr(A , (T*r,64*r),(H*64*r,1) ,((i_t * 64 + 32) *r, 0), (16*r, 16*r), (1,0))
    p_A32 = tl.make_block_ptr(A , (T*r,64*r),(H*64*r,1) ,((i_t * 64 + 32) *r, 16*r), (16*r, 16*r), (1,0))
    p_A41 = tl.make_block_ptr(A , (T*r,64*r),(H*64*r,1) ,((i_t * 64 + 48) *r, 0), (16*r, 16*r), (1,0))
    p_A42 = tl.make_block_ptr(A , (T*r,64*r),(H*64*r,1) ,((i_t * 64 + 48) *r, 16*r), (16*r, 16*r), (1,0))
    p_A43 = tl.make_block_ptr(A , (T*r,64*r),(H*64*r,1) ,((i_t * 64 + 48) *r, 32*r), (16*r, 16*r), (1,0))
    
    b_A21 = tl.load(p_A21, boundary_check=(0,1)).to(tl.float32)
    b_A31 = tl.load(p_A31, boundary_check=(0,1)).to(tl.float32)
    b_A32 = tl.load(p_A32, boundary_check=(0,1)).to(tl.float32)
    b_A41 = tl.load(p_A41, boundary_check=(0,1)).to(tl.float32)
    b_A42 = tl.load(p_A42, boundary_check=(0,1)).to(tl.float32)
    b_A43 = tl.load(p_A43, boundary_check=(0,1)).to(tl.float32)


    p_Ad11  = tl.make_block_ptr(Ad ,(T*r,16*r),(H*16*r,1), (i_t * 64 * r, 0), (16*r,16*r), (1,0))
    p_Ad22  = tl.make_block_ptr(Ad ,(T*r,16*r),(H*16*r,1), ((i_t * 64 + 16) * r, 0), (16*r,16*r), (1,0))
    p_Ad33  = tl.make_block_ptr(Ad ,(T*r,16*r),(H*16*r,1), ((i_t * 64 + 32) * r, 0), (16*r,16*r), (1,0))
    p_Ad44  = tl.make_block_ptr(Ad ,(T*r,16*r),(H*16*r,1), ((i_t * 64 + 48) * r, 0), (16*r,16*r), (1,0))


    p_Ai11 = tl.make_block_ptr(Ai, (T*r,64*r), (H*64*r, 1), ((i_t * 64 ) *r, 0), (16*r, 16*r), (1, 0))
    p_Ai22 = tl.make_block_ptr(Ai, (T*r,64*r), (H*64*r, 1), ((i_t * 64 + 16) *r, 16*r), (16*r, 16*r), (1, 0))
    p_Ai33 = tl.make_block_ptr(Ai, (T*r,64*r), (H*64*r, 1), ((i_t * 64 + 32) *r, 32*r), (16*r, 16*r), (1, 0))
    p_Ai44 = tl.make_block_ptr(Ai, (T*r,64*r), (H*64*r, 1), ((i_t * 64 + 48) *r, 48*r), (16*r, 16*r), (1, 0))
    p_Ai21 = tl.make_block_ptr(Ai, (T*r,64*r), (H*64*r, 1), ((i_t * 64 + 16) *r, 0), (16*r, 16*r), (1, 0))
    p_Ai31 = tl.make_block_ptr(Ai, (T*r,64*r), (H*64*r, 1), ((i_t * 64 + 32) *r, 0), (16*r, 16*r), (1, 0))
    p_Ai32 = tl.make_block_ptr(Ai, (T*r,64*r), (H*64*r, 1), ((i_t * 64 + 32) *r, 16*r), (16*r, 16*r), (1, 0))
    p_Ai41 = tl.make_block_ptr(Ai, (T*r,64*r), (H*64*r, 1), ((i_t * 64 + 48) *r ,0), (16*r, 16*r), (1, 0))
    p_Ai42 = tl.make_block_ptr(Ai, (T*r,64*r), (H*64*r, 1), ((i_t * 64 + 48) *r, 16*r), (16*r, 16*r), (1, 0))
    p_Ai43 = tl.make_block_ptr(Ai, (T*r,64*r), (H*64*r, 1), ((i_t * 64 + 48) *r, 32*r), (16*r, 16*r), (1, 0))


    Ai11 = tl.load(p_Ad11, boundary_check=(0, 1)).to(tl.float32)
    Ai22 = tl.load(p_Ad22, boundary_check=(0, 1)).to(tl.float32)
    Ai33 = tl.load(p_Ad33, boundary_check=(0, 1)).to(tl.float32)
    Ai44 = tl.load(p_Ad44, boundary_check=(0, 1)).to(tl.float32)
    
    Ai21 = -tl.dot(tl.dot(Ai22,b_A21, input_precision='ieee'),Ai11,input_precision='ieee')
    Ai32 = -tl.dot(tl.dot(Ai33,b_A32, input_precision='ieee'),Ai22,input_precision='ieee')
    Ai43 = -tl.dot(tl.dot(Ai44,b_A43, input_precision='ieee'),Ai33,input_precision='ieee')

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




@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['BT','r'],
)
@triton.jit(do_not_specialize=['T'])
def solve_tril_16x16_kernel(
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H:  tl.constexpr,
    r:  tl.constexpr,
    BT: tl.constexpr,
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

    A = A + (bos*H + i_h) * BT * r * r 
    Ad = Ad + (bos * r * H + i_h) * 16 * r 

    offset = (i_t * 16) % BT
    p_A = tl.make_block_ptr(A, (T,BT,r,r), (H*BT*r*r,r*r,r,1), (i_t * 16, offset,0,0), (16, 16,r,r), (3,2,1,0))
    ####Ad位置需要调整
    b_A = tl.load(p_A, boundary_check=(0, 1,2,3)).to(tl.float32)
    b_A = -tl.where((tl.arange(0, 16)[:, None] > tl.arange(0, 16)[None, :])[:,:,None,None], b_A, 0)
    o_i = tl.arange(0, 16)

    for i in range(1, min(16, T-i_t*16)):#避免超出范围
        mask = tl.arange(0, 16) == i 
        b_a = tl.sum(tl.where(mask[:,None,None,None], b_A, 0), 0)
        q = (tl.sum(b_a[:,None,:,:,None]*b_A[:,:,None,:,:],-2))
        b_a = b_a + tl.sum(q,0)*((tl.arange(0, 16) < i)[:,None,None])
        b_A = tl.where(mask[:,None,None,None],b_a,b_A)#按行计算 ，逐步交换结果
    b_A += ((tl.arange(0, 16)[:, None, None, None] == tl.arange(0, 16)[None, :, None, None])&(tl.arange(0, r)[None, None, :, None] == tl.arange(0, r)[None, None, None, :]))
    ####get BT BT r*r=> BT*r BT*r
    b_A = tl.permute(b_A,(0,2,1,3))
    b_A = tl.reshape(b_A,(16*r,16*r))#BT*r BT*r
    p_Ai = tl.make_block_ptr(Ad, (T*r, 16*r), (H*16*r, 1), (i_t * 16 * r , 0), (16*r, 16*r), (1, 0))
    tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


#done
def gated_chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    mask: torch.Tensor| None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 32,
    output_dtype: torch.dtype = torch.float32,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K = k.shape
    r = mask.shape[-1] #B T H r r
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B,T,H, BT, r*r,device=k.device, dtype=output_dtype)                                                                                                                    
    gated_chunk_scaled_dot_kkt_fwd_kernel[(NT, B*H)](
        k=k, beta=beta, g=g, mask=mask, A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T, K=K, H=H, BT=BT,r=r,
    )
    return A



@input_guard
def solve_tril(
    A: torch.Tensor,
    mask: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float):
    B,T,H,BT,_= A.shape
    r = mask.shape[-1]

    A = rearrange(A,'b t h l (c r)->b (t c) h (l r)',c=r).contiguous()#BT*r BT*r  

    chunk_indices = prepare_chunk_indices(r*cu_seqlens, 16) if cu_seqlens is not None else None
    N_64 = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(r*T,16)

    Ad_1 = torch.empty(B,r*T,H,16,device=A.device, dtype=torch.float)
    solve_tril_16x16_kernel_org[(N_64, B*H)](
            A=A,Ad=Ad_1,
            cu_seqlens=r*cu_seqlens if cu_seqlens is not None else None,
            chunk_indices=chunk_indices,
            T=T,
            r=r, BT=BT,H=H,
    )
    chunk_indices = prepare_chunk_indices(cu_seqlens, 16) if cu_seqlens is not None else None
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T,16)

    Ad = torch.zeros(B,T*r,H,16*r,device=A.device, dtype=torch.float if BT != 16 else output_dtype)
    if r==1:
        Ad = Ad_1    
    if r==2:
        merge_r1_to_r2_inverse_kernel[(NT, B*H)](
            A=A,Ad=Ad_1,Ai=Ad,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,r=r,BT=BT,H=H
        )
    if r==4:
        merge_r1_to_r4_inverse_kernel[(NT, B*H)](
            A=A,Ad=Ad_1,Ai=Ad,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,r=r,BT=BT,H=H
        )
    if r==8:
        Ad = torch.zeros(B,T*r,H,16*4,device=A.device, dtype=torch.float if BT != 16 else output_dtype)###根据r考虑如何merge
        merge_r1_to_r4_inverse_kernel[(2*NT, B*H)](
            A=A,Ad=Ad_1,Ai=Ad,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=2*T,r=4,BT=2*BT,H=H
        )
        Ad1 = torch.zeros(B,H,NT*r*16,16*r,device=A.device, dtype=torch.float if BT != 16 else output_dtype)###根据r考虑如何merge
        merge_r4_to_r8_inverse_kernel[(NT,B*H)](
            # A,Ad,Ad1,
            A=A,Ad=Ad,Ai=Ad1,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,r=8,BT=BT,H=H,
        )
    # print('down2')
    if BT == 16: 
        if r==8: 
            return Ad1                                                                                                                         
        return Ad

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    Ai = torch.zeros(B,T*r,H,BT*r,device=A.device, dtype=output_dtype)

    if BT == 32:
        merge_16x16_to_32x32_inverse_kernel[(NT, B*H)](
            A=A,Ad=Ad,Ai=Ai,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,r=r,BT=BT,H=H
        )
        return Ai

    if BT == 64:
        merge_16x16_to_64x64_inverse_kernel[(NT, B*H)](
            A=A,Ad=Ad,Ai=Ai,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,r=r,BT=BT,H=H
        )
        return Ai


def gated_fwd_recompute_w_u(k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    mask:torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None):
    B, T, H, K, V = *k.shape, v.shape[-1]
    r = mask.shape[-1]
    BT = A.shape[-1]//r
    u = torch.empty(B,r*T,H,V,device=k.device, dtype=k.dtype)
    w = torch.empty(B,r*T,H,K,device=k.device, dtype=k.dtype)
    NT = triton.cdiv(T, BT)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BK = min(triton.next_power_of_2(K//r), 64)#32
    BV = min(triton.next_power_of_2(V), 64)
    gated_fwd_recompute_w_u_kernel[(NT, B*H)](
        k=k, v=v, beta=beta,mask=mask, w=w, u=u, A=A,
        g = g, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,H=H,K=K,V=V,r=r,BT=BT,BV=BV,BK=BK,
    )
    return w, u


def ceildiv(a, b):
    return -(a // -b)

def pad(x, chunk_size=16):
    seq_len = x.shape[-2]
    #b n l d
    padded_seq_len = ceildiv(seq_len, chunk_size) * chunk_size
    if x.shape[-2] % chunk_size != 0:
        x = F.pad(x, (0, 0, 0, padded_seq_len - seq_len))
    return x

def pad_b(x,val, chunk_size=16):
    seq_len = x.shape[-1]  # 获取序列长度 l
    padded_seq_len = ceildiv(seq_len, chunk_size) * chunk_size  # 计算填充后的长度
    # 如果序列长度不是 chunk_size 的倍数，则进行填充
    if seq_len % chunk_size != 0:
        x = F.pad(x, (0, padded_seq_len - seq_len),value=val)  # 只在最后一个维度（l）进行填充
    return x

def pad_m(x,val, chunk_size=16):
    seq_len = x.shape[-3]  # 获取序列长度 b h l r r
    padded_seq_len = ceildiv(seq_len, chunk_size) * chunk_size  # 计算填充后的长度
    # 如果序列长度不是 chunk_size 的倍数，则进行填充
    if seq_len % chunk_size != 0:
        x = F.pad(x, (0,0,0,0,0,padded_seq_len - seq_len),value=val)  # 只在最后一个维度（l）进行填充
    return x


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=['H', 'K', 'V', 'BT','r'],
    use_cuda_graph=use_cuda_graph,
    **autotune_cache_kwargs
)    
@triton.jit(do_not_specialize=['T'])
def gated_chunk_delta_rule_fwd_kernel_h(
    k,
    v,#u
    w,#w
    v_new,
    g,
    h,
    h0, 
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    r: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_h1 = tl.zeros([K, BV], dtype=tl.float32)

    h += (boh * H + i_h) * K*V
    k += (bos * H + i_h) * K
    w += (bos * r * H + i_h) * K
    v += (bos * r * H + i_h) * V ####这个需要调整

    if SAVE_NEW_VALUE:
        v_new += (bos * r * H + i_h) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
    # B,H,NV,NT
    stride_h = H*K*V
    stride_k = H*K
    stride_v = H*V
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T*r, K), (stride_k, 1), (i_t * BT * r, 0), (BT*r, K), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))

        p_v = tl.make_block_ptr(v, (T*r, V), (stride_v, 1), (i_t * BT* r, i_v*BV), (BT* r, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v_new = tl.make_block_ptr(v_new, (T*r, V), (stride_v, 1), (i_t * BT* r, i_v*BV), (BT* r, BV), (1, 0))
            tl.store(p_v_new,b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        b_v = tl.reshape(b_v,BT,r,BV)
        if USE_G:##这部分也完全改写了，可能存在更快的方案
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None,None] ####BT, r ,BV ###此处bv应当被reshape看成
            b_g_last = exp(b_g_last)
            b_h1 *= b_g_last

        b_v = b_v.to(k.dtype.element_ty)###actully 等价r个 ###BT r BV
        b_v = tl.permute(b_v,(1,0,2))####r BT BV

        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (K, BT), (0, 1))
        b_k = tl.load(p_k,boundary_check=(0,1)) #### K BT
        b_k = tl.reshape(b_k,(r,K//r,BT))

        h_sum = tl.dot(b_k,b_v)###get r K//r BV
        h_sum = tl.reshape(h_sum,(K,BV))####if K>64 怎么加呢，shape不对
        b_h1 += h_sum

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
#finish
@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': 128}, num_warps=8, num_stages=3),
        triton.Config({'BV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BV': 32}, num_warps=2, num_stages=3),
    ],
    key=['H', 'K', 'V', 'BT','BK','r'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def gated_chunk_linear_attn_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    r : tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bhr = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bh = i_bhr//r
    i_r = i_bhr % r
    i_b = i_bh//H
    i_h = i_bh%H
    rk = K//r

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += ((bos * r + i_r )* H + i_h) * V
    o += ((bos * r + i_r )* H + i_h) * V
    
    h += (i_tg * H + i_h).to(tl.int64) * K*V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K//r, BK)):#这里需要注意拆分#这里K//BK = r
        #问题是不同r_block读取了同一份qk，有影响吗
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_r * rk + i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1,H*K), (i_r * rk + i_k * BK,i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_r * rk + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1)) 
        b_o += tl.dot(b_q, b_h)
        b_s += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        b_s = b_s * exp(b_g[:, None] - b_g[None, :])


    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_s = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_s = tl.where(m_s, b_s, 0)

    p_v = tl.make_block_ptr(v, (T, V), (H*V*r, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (H*V*r, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + tl.dot(b_s.to(b_v.dtype), b_v) * scale
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
def gated_chunk_fwd_h_fn(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    chunk_size:int=32,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None):
    # k, w, u, g, BT, initial_state, final_state
    B, T, H , K, V = *k.shape,u.shape[-1]
    _,rT,_,_ = w.shape
    r = rT//T
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = torch.empty(B, NT,H, K, V,device=k.device,dtype=k.dtype)
    final_state = torch.empty(B, H, K, V,device=k.device,dtype=torch.float32)
    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    v_new = torch.empty(B,T,r,H,V,dtype=u.dtype,device=u.device)#做了v_new的r_first
    gated_chunk_delta_rule_fwd_kernel_h[grid](#r没有for循环,不考虑增加gk的使用
        k=k,v=u,w=w, 
        v_new=v_new,
        g=g,h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        H=H, T=T, K=K, V=V, BT=BT,r=r,      
    )    
    return h, v_new,final_state

#finish
def gated_chunk_fwd_o_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 32,
    chunk_indices: torch.LongTensor | None = None,
):
    BT =chunk_size
    B,T,r,H,V,K = *v.shape,q.shape[-1]
    BK = triton.next_power_of_2(K//r)
    o = torch.empty_like(v)#there_fore,bTr Hv
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    BK = min(triton.next_power_of_2(K//r), 64)

    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H * r)
    gated_chunk_linear_attn_fwd_kernel_o[grid](
        q=q, k=k, v=v, h=h, g=g, o=o,
        cu_seqlens = cu_seqlens,
        chunk_indices = chunk_indices,
        scale=scale,
        H=H, T=T, K=K, V=V, BT=BT, BK=BK,r = r,
    )
    o = o.sum(dim=2)#沿着r维度求和
    return o

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps  in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G','r'],
    **autotune_cache_kwargs
)
@triton.jit(do_not_specialize=['T'])
def gated_fwd_prepare_dv_kernel(
    q,
    k,
    g,
    do,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    r: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):

    i_t, i_bhr = tl.program_id(0), tl.program_id(1)
    i_bh = i_bhr//r
    i_r = i_bhr % r
    i_b = i_bh//H
    i_h = i_bh%H
    block_r = K//r

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    do += (bos * H + i_h) * V
    dv += ((bos * r + i_r )* H + i_h) * V

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(block_r, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_r * block_r + i_k * BK), (BT, BK), (1, 0))
        p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (i_r * block_r + i_k * BK, i_t * BT), (BK, BT), (0, 1))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_A += tl.dot(b_k, b_q) * scale
    if USE_G:
        b_A *= exp(b_g[None, :] - b_g[:, None]) 

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))        
        p_dv = tl.make_block_ptr(dv, (T, V), (H*V*r, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

#finish
def gated_fwd_prepare_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    r: int,
    g: torch.Tensor | None = None,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    BT: int = 64,
    chunk_indices: torch.LongTensor | None = None,):
    B, T, H, K, V = *k.shape, do.shape[-1]
    dv = torch.empty(B,T,r,H,V,device = do.device, dtype= do.dtype)#没法like
    
    BT = min(BT, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None

    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(triton.next_power_of_2(K//r),64)
    BV = min(triton.next_power_of_2(V), 64)
    gated_fwd_prepare_dv_kernel[(NT, B*H*r)](
        q=q, k=k, g=g , do=do, dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        r=r,
        T=T,H=H,K=K,V=V,BT=BT,BK=BK,BV=BV,
    )
    return dv


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [4, 3, 2]
        for BV in [64, 32]
    ],
    key=['H', 'K', 'V', 'BT', 'BV', 'BK','USE_G'],
    use_cuda_graph=use_cuda_graph,
    **autotune_cache_kwargs
)
@triton.jit(do_not_specialize=['T'])
def gated_chunk_delta_rule_bwd_kernel_dhu(
    q,
    k,
    w,
    g,
    do,
    dh,
    dht,
    dh0,
    dv,
    dv2,
    scale,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    r: tl.constexpr,
    KR: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)#这个不变 读取所有

    dh += (boh * H + i_h) * K*V
    dv += (bos *r * H + i_h) * V
    dv2 += (bos *r * H + i_h) * V
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K

    w += (bos * r*  H + i_h) * K
    do += (bos * H + i_h) * V
    stride_v = H*V
    stride_h = H*K*V
    stride_k = H*K
    if USE_INITIAL_STATE:
        dh0 += i_nh * K*V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K*V

    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):# 向前偏移了一位,计算流程是对的
        p_dh = tl.make_block_ptr(dh + i_t * stride_h , (K, V), (V, 1), (0 , i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1)) 

        if USE_G:
            last_idx = min((i_t + 1) * BT, T) - 1
            bg_last = tl.load(g + (bos + last_idx) * H + i_h)
            bg_last_exp = exp(bg_last)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_g_exp = exp(b_g)
        else:
            bg_last = None
            last_idx = None
            b_g = None
            b_g_exp = None

        p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1),
                                (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T*r, V), (stride_v, 1),
                                (i_t * BT * r, i_v * BV), (BT*r, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv , (T*r, V), (stride_v , 1),
                                (i_t * BT * r, i_v * BV), (BT*r, BV), (1, 0))#load r
        ###read
        b_dv = tl.load(p_dv, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, BK), (1, 0))        
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k = tl.permute(tl.reshape(b_k,(BT,r,KR)),(1,0,2))
        b_h_trans = tl.reshape(b_dh,(r,KR,BV))
        b_ss = tl.dot(b_k,b_h_trans.to(b_k.dtype))####r BT BV
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_ss *= tl.where(m_t, exp(bg_last - b_g), 0)[None,:, None]
        b_dv += tl.reshape(tl.permute(b_ss,(1,0,2)),(BT*r,BV)) ####BT*r BV
        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


        p_w = tl.make_block_ptr(w, (K, T*r), (1, stride_k), (0, i_t * BT*r), (BK, BT*r), (0, 1))
        p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (0, i_t * BT), (BK, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        if USE_G:
            b_dh *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        b_q = (b_q * scale).to(b_q.dtype)
        b_dh += tl.dot(b_q, b_do.to(b_q.dtype))-tl.dot(b_w, b_dv.to(b_w.dtype))
    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))



def gated_chunk_bwd_dhu_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    dht: Optional[torch.Tensor],
    do: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    BT: int = 32, 
):
    B,T,r,H,V,K = *dv.shape,q.shape[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension being larger than 256."
    
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    dh = q.new_empty(B, NT , H, K,V)#一样的#need 求和 得一起算
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)####B T r H V

    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    gated_chunk_delta_rule_bwd_kernel_dhu[grid](
        q=q, k=k, w=w, g=g, do=do, dh=dh,dht=dht,dh0=dh0, dv=dv, dv2=dv2,
        cu_seqlens=cu_seqlens,chunk_offsets=chunk_offsets,
        scale=scale,
        H=H, T=T, K=K, V=V, BT=BT, BK=BK,r=r,KR = K//r,
    )
    return dh,dh0,dv2

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2,4]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G', 'r'],
    **autotune_cache_kwargs
)
@triton.jit(do_not_specialize=['T'])
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
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    NK: tl.constexpr,
    BV: tl.constexpr,
    r: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    
    i_k, i_t, i_bhr = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_r = i_bhr%r
    i_bh = i_bhr//r
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    v += ((bos * r + i_r) * H + i_h) * V

    do += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K*V
    dh += (i_tg * H + i_h).to(tl.int64) * K*V
    q += (bos * H + i_h) * K 
    k += (bos * H + i_h) * K 
    dq += (bos * H + i_h) * K 
    dk += (bos * H + i_h) * K 

    dv += (bos * H * r + i_h) * V
    dw += (bos * H * r + i_h) * K
    w += (bos * H * r + i_h) * K


    dg += (i_r * NK + i_k) * all * H
    b_dg_last = tl.zeros([1,], dtype=tl.float32)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT*r,BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    ####dq save位置正常

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v , (T, V), (H*V*r, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))### 对的
        p_do = tl.make_block_ptr(do , (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h, (V, K), (1,V), (i_v * BV,  i_r * K // r + i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh, (V, K), (1,V), (i_v * BV, i_r * K // r + i_k * BK), (BV, BK), (0, 1))
        b_v  = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h  = (tl.load(p_h, boundary_check=(0, 1)))#BV BK
        b_dh = (tl.load(p_dh, boundary_check=(0, 1)))#需要额外添加r维度

        b_dg_last += (tl.sum(b_h * b_dh))
        
        b_ds += tl.dot(b_do, tl.trans(b_v))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))   

        p_dv = tl.make_block_ptr(dv , (T*r, V), (H*V, 1), (i_t * BT * r, i_v * BV), (BT * r, BV), (1, 0))
        b_dv = tl.load(p_dv, boundary_check=(0, 1))
        b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))

    p_dw = tl.make_block_ptr(dw, (T*r, K), (H*K,1), (i_t * BT * r, i_r * K//r + i_k * BK), (BT*r ,BK), (1, 0))
    tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))
    tl.debug_barrier()
    
    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_r*K//r + i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_r*K//r + i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_t * BT, i_r*K//r + i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_t * BT, i_r*K//r + i_k * BK), (BT, BK), (1, 0))
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    

    b_dg = tl.zeros([BT,], dtype=tl.float32)
    g += bos * H + i_h
    dg += bos * H + i_h
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)
    b_dg_last *= exp(b_g_last)

    b_dq = b_dq * exp(b_g)[:, None] * scale

    b_dk = b_dk * tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
    b_dg_last += tl.sum(b_dk * b_k)

    b_ds = tl.where(m_A, b_ds * exp(b_g[:, None] - b_g[None, :]), 0) * scale
    b_ds = b_ds.to(b_k.dtype)
    # [BT, BK]
    b_dq += tl.dot(b_ds, b_k)
    b_dk += tl.dot(tl.trans(b_ds), b_q)
    b_dg = tl.sum(b_dq * b_q, axis=1) - tl.sum(b_dk * b_k, axis=1)

    p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))



def gated_chunk_bwd_dqkw_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    BT: int = 32,
):
    B,T,H, K, V = *q.shape, v.shape[-1]
    _,RT,_,_ = w.shape
    r = RT // T
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    #最后一个函数，计算dw,dq,dk
    BK = triton.next_power_of_2(K//r)#需要更细粒度的划分，确保不会使得 不同位置的划到一起
    BK = min(triton.next_power_of_2(K//r), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NK = triton.cdiv(K//r, BK)

    grid = (NK, NT, B * H * r)#通过NK控制位置
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)#k_org
    dw = torch.empty_like(w)#b t r h k
    dg = torch.empty(r*NK,*g.shape,dtype=torch.float32,device=g.device)

    gated_chunk_delta_rule_bwd_kernel_dqkw[grid](
        q=q, k=k, v=v, w=w, g=g, h=h, do=do, dh=dh, dq=dq, dk=dk, dv=dv, dw=dw,dg=dg,
        cu_seqlens=cu_seqlens,chunk_indices=chunk_indices,
        scale=scale,
        B=B,
        H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,r = r,NK=NK,
    )
    dg = dg.sum(0)
    return dq.to(q.dtype), dk.to(k.dtype), dw.to(w.dtype),dg

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV','r','IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def gated_bwd_prepare_wy_repr_kernel(           
    k, v, beta,mask,g,A,
    dw, du,
    dk, dv, dbeta,dmask,dg,
    cu_seqlens,
    chunk_indices,
    T,
    K:tl.constexpr,
    H:tl.constexpr,
    V:tl.constexpr,
    r: tl.constexpr,
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

    p_b = tl.make_block_ptr(beta + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(A + (bos * H * r + i_h) * BT * r, (BT * r, T * r), (1, H*BT*r), (0, i_t * BT * r), (BT*r, BT*r), (0, 1))


    b_b = tl.load(p_b, boundary_check=(0,))
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dA = tl.zeros([BT*r, BT*r], dtype=tl.float32)
    b_dmask = tl.zeros([BT,r,r],dtype=tl.float32)
    block_k = K//r
    
    p_g = tl.make_block_ptr(g + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_g_exp = tl.exp(b_g)
    b_dg = tl.zeros([BT], dtype=tl.float32)
    

    for i_r in range(r):
        p_r_mask = tl.make_block_ptr(mask + (bos*H + i_h)*r*r,(T,r,r),(H*r*r,r,1),(i_t*BT,0,i_r),(BT,r,1),(2,1,0))
        b_rmask = tl.reshape(tl.load(p_r_mask),(BT,r))
        rmask = tl.arange(0, r) == i_r #第ir列
        for i_k in range(tl.cdiv(block_k, BK)):
            p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T,K), (H*K, 1), (i_t * BT, i_r*block_k + i_k * BK), (BT,BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + (bos*H + i_h) * K, (T,K), (H*K,1), (i_t * BT, i_r*block_k + i_k * BK), (BT,BK), (1, 0))
            p_dw = tl.make_block_ptr(dw + (bos*H*r + i_h) * K, (T*r, K), (H*K,1), (i_t * BT * r,i_r*block_k + i_k * BK), (BT*r,BK), (1,0))
            # [BT, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_kb = b_k * (b_b * b_g_exp)[:, None] ###BT BK

            b_kbg = b_kb[:,None,:] * b_rmask[:,:,None]####get BT r BK ####reshape this
            b_dw = tl.load(p_dw, boundary_check=(0, 1))

            b_dA += tl.dot(b_dw, tl.trans(tl.reshape(b_kbg,(BT*r,BK))).to(b_dw.dtype))####需要获得BT*r BT*r的计算结果
            b_dkbg = tl.dot(b_A, b_dw)###get BT*r BK

            b_dkbg_re = tl.reshape(b_dkbg,(BT,r,BK))

            ###BT BK
            b_dk = tl.sum(b_dkbg_re * (b_g_exp * b_b)[:, None,None] * b_rmask[:,:,None],1)
            b_db += tl.sum(tl.reshape(b_dkbg_re * b_k[:,None,:] * b_g_exp[:, None,None] * b_rmask[:,:,None],(BT,BK*r)),1)
            b_dg += tl.sum(tl.reshape(b_dkbg_re * b_kbg,(BT,BK*r)),1)
            b_ss = tl.sum(b_dkbg_re * ((b_k[:,None,:] * (b_g_exp * b_b)[:, None,None])) ,axis = -1)###BT r
            b_dmask += (b_ss[:,:,None].to(tl.float32)*rmask[None,None,:].to(tl.float32))
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(du + (bos*H*r + i_h) * V, (T*r, V), (H*V, 1), (i_t * BT * r, i_v * BV), (BT*r, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = ((b_v * b_b[:, None])[:,None,:]*tl.full([r],1, dtype=b_v.dtype)[None,:,None]).to(b_v.dtype)##BT*r*BV
        b_vb = tl.reshape(b_vb,(BT*r,BV))

        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA += tl.dot(b_du, tl.trans(b_vb))####BT*r BT*r

        b_dvb = tl.dot(b_A, b_du)####BT*r BV
        b_dvb = tl.reshape(b_dvb,(BT,r,BV))
        sum_dv = tl.sum(b_dvb,axis=1)###BT BV
        
        b_dv = sum_dv * b_b[:, None]
        b_db += tl.sum(sum_dv * b_v, 1)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    i = tl.arange(0, BT * r)[:, None]
    j = tl.arange(0, BT * r)[None, :]
    iB = i // r
    jB = j // r
    m_t = i_t * BT * r + tl.arange(0, BT * r) < T * r
    m_A = (iB > jB) & (m_t[:, None] & m_t)

    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))#####BT*r BT*r

    b_dA = tl.where(m_A, -b_dA, 0)
    b_dA = tl.reshape(b_dA,(BT,r,BT,r))
    b_dA *= exp(b_g[:, None] - b_g[None, :])[:,None,:,None]

    b_dA = b_dA.to(k.dtype.element_ty)
    b_dA = tl.permute(b_dA,(0,2,1,3))#Bt bt r r
    b_A = tl.zeros([BT, BT,r,r], dtype=tl.float32)

    tl.debug_barrier()
    for i_r in range(r):
        p_r_mask = tl.make_block_ptr(mask + (bos*H + i_h)*r*r,(T,r,r),(H*r*r,r,1),(i_t*BT,0,i_r),(BT,r,1),(2,1,0))
        b_rmask = tl.reshape(tl.load(p_r_mask),(BT,r))
        rmask = tl.arange(0, r) == i_r #第ir列
        sum_da = tl.sum(tl.where(rmask[None,None,None,:], b_dA, 0), -1)#BT BT r #ir
        ir_A = tl.sum(sum_da * b_rmask[:,None,:],-1).to(k.dtype.element_ty)#BT BT

        for i_k in range(tl.cdiv(block_k, BK)):
            p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T,K), (H*K, 1), (i_t * BT, i_r * block_k + i_k * BK), (BT,  BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + (bos*H + i_h) * K, (T,K), (H*K,1), (i_t * BT, i_r * block_k + i_k * BK), (BT,  BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))####BT BK
            b_dk = tl.load(p_dk, boundary_check=(0, 1))
            b_kt = tl.trans(b_k)###BK BT
            b_kb = b_k * b_b[:,None]###BT BK

            beta_kkt =  (tl.dot(b_kb, b_kt))####BT BT
            b_A += beta_kkt[:,:,None,None] * ((rmask[None,None,:] * b_rmask[:,:,None])[:,None,:,:])#####BT BT 
            b_dkb = tl.dot(ir_A, b_k)#######BT BK
            b_db += tl.sum(b_dkb*b_k,1)

            sss = tl.dot((b_kt* b_b[None,:]), ir_A)###BT BT
            b_dk +=  tl.trans(sss)####BT BK

            b_dk += b_dkb * b_b[:, None]
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

            betas = tl.sum(beta_kkt[:,:,None]*sum_da,1)
            b_dmask +=  betas[:,:,None]*rmask[None,None,:]

    p_db = tl.make_block_ptr(dbeta + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))
    ###应该是对的

    p_dmask = tl.make_block_ptr(dmask + (bos*H + i_h) * r * r , (T,r,r), (H*r*r,r,1), (i_t * BT,0,0), (BT,r,r), (2,1,0))
    tl.store(p_dmask, b_dmask.to(p_dmask.dtype.element_ty), boundary_check=(0,1,2))

    b_AdA = b_dA * b_A
    # b_AdA_reshaped = tl.sum(tl.reshape(b_AdA,(BT,BT,r*r)),-1) ###BUG maybe because after permute,leading to un-contiguous
    b_AdA_reshaped = tl.sum(tl.sum(b_AdA,axis=-1),axis=-1)

    p_dg = tl.make_block_ptr(dg + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_dg += tl.sum(b_AdA_reshaped, axis=1) - tl.sum(b_AdA_reshaped, axis=0)
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

@input_guard
def gated_bwd_prepare_wy_repr(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    mask: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    BT: int = 32,
    cu_seqlens: torch.LongTensor | None = None,
):
    B, T,H, K, V = *k.shape, v.shape[-1]
    r = mask.shape[-1]
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BK = min(triton.next_power_of_2(K//r), 64)
    BV = min(triton.next_power_of_2(V), 64)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v).contiguous()
    dbeta = torch.zeros_like(beta)
    dg = torch.empty(*g.shape,dtype=torch.float32,device=g.device)
    dmask = torch.zeros([B,T,H,r,r],device=k.device,dtype=k.dtype).contiguous()
    assert BK <= K//r
    gated_bwd_prepare_wy_repr_kernel[(NT, B*H)](
        k=k, v=v, beta=beta, mask=mask, g=g, A=A,
        dw=dw, du=du,
        dk=dk, dv=dv, dbeta=dbeta,dmask=dmask,dg=dg,
        cu_seqlens=cu_seqlens,chunk_indices=chunk_indices,
        T=T, K=K, V=V, r=r, BT=BT, BK=BK, BV=BV,H=H,
    )
    return dk, dv, dbeta, dmask,dg

def mask_chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    mask: torch.Tensor,
    BT: int,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None
):   
    
    B,L,h,dk = q.shape
    r = mask.shape[-1]
    g = chunk_local_cumsum(g, chunk_size=BT, cu_seqlens=cu_seqlens)##no-need-change
    A = gated_chunk_scaled_dot_kkt_fwd(k=k,beta=beta,g=g,
        mask=mask,cu_seqlens=cu_seqlens,output_dtype=torch.float32,chunk_size=BT)###需要对照修改

    A = solve_tril(A=A,mask=mask,cu_seqlens=cu_seqlens,output_dtype=k.dtype)#bh

    w, u = gated_fwd_recompute_w_u(k=k, v=v, beta=beta, mask=mask,A=A,g=g,cu_seqlens=cu_seqlens)#

    h, v_new,final_state = gated_chunk_fwd_h_fn(k=k, w=w, u=u, g=g, initial_state=initial_state,chunk_size=BT,
                                                output_final_state=output_final_state,
                                                cu_seqlens=cu_seqlens,) 

    o = gated_chunk_fwd_o_fn(q=q, k=k, v=v_new, h=h, g=g,chunk_size=BT,scale=scale,cu_seqlens=cu_seqlens)

    return g, o, A, final_state

def mask_chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    mask: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    BT:int,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None
):
    B,L,h,dk = q.shape
    r = mask.shape[-1]
    

    w, u = gated_fwd_recompute_w_u(k=k, v=v, beta=beta, mask=mask,A=A,g=g,cu_seqlens=cu_seqlens)#
        

    
    
    h, v_new,_ = gated_chunk_fwd_h_fn(k=k, w=w, u=u, g=g, initial_state=initial_state,chunk_size=BT,
                                                output_final_state=False,
                                                cu_seqlens=cu_seqlens,) 

    
    dv = gated_fwd_prepare_dv(q=q, k=k, g=g, do=do,scale=scale,r=r,BT=BT,cu_seqlens=cu_seqlens)
    

    #############B T r H V 
    dh, dh0, dv = gated_chunk_bwd_dhu_fn(q=q, k=k, w=w, g=g,h0=initial_state,dht=dht,do=do, dv=dv, BT=BT,
                                         scale=scale,cu_seqlens=cu_seqlens)



    dq, dk, dw, dg = gated_chunk_bwd_dqkw_fn(q=q, k=k, v=v_new, w=w, g=g, h=h, dv=dv, do=do, dh=dh,scale=scale, BT=BT,
                                              cu_seqlens=cu_seqlens)



    dk2, dv, dbeta,dmask,dg2 = gated_bwd_prepare_wy_repr(k=k, v=v, beta=beta, mask=mask,g=g, A=A, dw=dw, du=dv, BT=BT,cu_seqlens=cu_seqlens)#只有这里带mask
    dk.add_(dk2)
    dg.add_(dg2)
    
    dg = chunk_local_cumsum(dg, chunk_size=BT, reverse=True,cu_seqlens=cu_seqlens,output_dtype=torch.float)##no-need-change
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dbeta.to(beta.dtype),dg,dmask.to(mask.dtype),dh0


class mask_gated_ChunkDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(        
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
        BT: int,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g, o, A, final_state = mask_chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            mask=mask,
            scale=scale,
            BT=BT,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta,mask, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.BT = BT
        return o.to(q.dtype), final_state
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        q, q_rstd, k, k_rstd, v, g, beta, mask, A, initial_state, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, dbeta, dg, dmask, dh0 = mask_chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            mask=mask,
            A=A,
            BT=ctx.BT,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dbeta.to(beta),dg,dmask.to(mask),None, None,dh0,None, None,None


def delta_rule_recurrence(q, k, v, beta, g, mask,initial_state=None,output_final_state=True):
    g_exp = torch.exp(g).float()
    BT = 32
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    r = mask.shape[-1]
    o = torch.zeros_like(v)
    if l%BT==0:
        S_t = torch.zeros(b, h, l//BT, d_k, d_v,device=k.device,dtype=torch.float32)
    else:
        S_t = torch.zeros(b, h, l//BT + 1, d_k, d_v,device=k.device,dtype=torch.float32)
    if initial_state == None:
        S = torch.zeros(b, h, d_k, d_v,device=k.device,dtype=torch.float32)
    else:
        S = initial_state
    if beta.ndim < v.ndim:
        beta = beta[..., None]
    for i in range(l):
        if i%BT==0:
            S_t[:,:,i//BT,:,:] = S
        _k = k[:, :, i].float()
        _q = q[:, :, i].float()*(d_k ** -0.5)
        _v = v[:, :, i].float()
        beta_i = beta[:, :, i].float()
        _v = _v * beta_i
        kkt = torch.einsum('b h d,b h v->b h d v',_k*beta_i,_k)
        kkt = rearrange(kkt,' b h (r d) (l v)-> b h r d l v',r= r,l=r)
        kkt = torch.einsum('b h r d l v,b h r l->b h r d l v',kkt,mask[:,:,i,:,:].float())#16d参数，几乎可以忽略
        kkt = rearrange(kkt,'b h r d l v-> b h (r d) (l v)')
        iplr = torch.eye(d_k).to(q)-kkt
        iplr = torch.einsum(' b h q k ,b h->b h q k',iplr,g_exp[:,:,i])
        S = torch.einsum('b h q k ,b h k v->b h q v',iplr.float(),S) + _k.unsqueeze(-1).float() * _v.unsqueeze(-2).float()
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q.float(), S).to(k.dtype)
    return o,S_t,S


def mask_gated_chunk_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    mask: torch.Tensor,#use for mask org_tensor 
    BT: int,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
):
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "FusedChunkDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."
    if head_first:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, final_state = mask_gated_ChunkDeltaRuleFunction.apply(
        q, k, v, beta,g,mask, scale,BT, initial_state, output_final_state,cu_seqlens,use_qk_l2norm_in_kernel)
    return o, final_state


if __name__ =="__main__":
    import sys
    import time
    from fla.modules.l2norm import l2_norm as l2_norm_fn 
    torch.set_default_dtype(torch.bfloat16)
    import torch
    import random
    import numpy as np
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    B = 2
    H = 8
    L = 2048
    DK = 256
    DV = 256
    q = (torch.randn(B, H, L, DK)).cuda().requires_grad_(True)
    k = (torch.randn(B, H, L, DK)).cuda()
    k = torch.nn.functional.normalize(k, dim=-1, p=2).requires_grad_(True)
    v = (torch.randn(B, H, L, DV)).cuda().requires_grad_(True)
    do = (torch.randn(B, H, L, DV)).cuda()
    beta = torch.randn(B, H, L).cuda().sigmoid().requires_grad_(True)
    r=4
    mask = (torch.randn(B,H,L,r,r)).cuda().requires_grad_(True)

    g = (torch.nn.functional.logsigmoid(torch.randn(B, H, L).cuda())).requires_grad_(True)


    B,H,L,DV = v.shape
    q, k, v, beta, g,mask,do = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (q, k, v, beta, g,mask,do))
    # g_exp = torch.exp(g)
    # o11,h_11,ss = delta_rule_recurrence(q,k,v,beta,g,target_matrix)
    # do = torch.randn(B, H, L, DV).cuda()
    # # o11.backward(do, retain_graph=True)
    # q_grad, q.grad = q.grad, None
    # k_grad, k.grad = k.grad, None
    # v_grad, v.grad = v.grad, None
    # beta_grad, beta.grad = beta.grad, None
    # g_grad, g.grad = g.grad, None
    # mask_grad, mask.grad = mask.grad, None
    o22,f_state = mask_gated_chunk_delta_rule(q, k, v, beta, g,mask,BT=32,output_final_state=True)#10s嘛 额
    o22.backward(do,retain_graph=True)



