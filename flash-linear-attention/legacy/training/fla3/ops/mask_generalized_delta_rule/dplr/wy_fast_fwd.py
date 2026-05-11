# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple
from einops import rearrange
import torch
import triton
import triton.language as tl
import torch.nn.functional as F

from ....ops.utils import prepare_chunk_indices
from ....ops.utils.op import gather
from ....utils import is_gather_supported, use_cuda_graph
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
# from fla.utils import autotune_cache_kwargs
from fla.utils import is_nvidia_hopper, use_cuda_graph



@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BT'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_fwd_kernel_chunk32(
    A_ab,
    A_ab_inv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,  # placeholder, do not delete
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
    p_Aab = tl.make_block_ptr(A_ab + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_Aab_inv = tl.make_block_ptr(A_ab_inv + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A_ab = tl.load(p_Aab, boundary_check=(0, 1))
    for i in range(1, BT):
        mask = tl.arange(0, BT) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A_ab, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A_ab, 0) * (tl.arange(0, BT) < i)
        b_A_ab = tl.where(mask[:, None], b_a, b_A_ab)
    b_A_ab += tl.arange(0, BT)[:, None] == tl.arange(0, BT)[None, :]
    tl.store(p_Aab_inv, b_A_ab.to(p_Aab_inv.dtype.element_ty), boundary_check=(0, 1))


# @triton.heuristics({
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
# })
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [2, 4, 8]
#         for num_stages in [2, 3, 4]
#     ],
#     key=['H', 'BT', 'IS_VARLEN','r'],
#     # **autotune_cache_kwargs,
# )
# @triton.jit(do_not_specialize=['T'])
# def solve_tril_16x16_kernel_org(
#     A,
#     Ad,
#     cu_seqlens,
#     chunk_indices,
#     T,
#     H:  tl.constexpr,
#     r:  tl.constexpr,
#     BT: tl.constexpr,
#     IS_VARLEN: tl.constexpr,
# ):
#     i_t, i_bh = tl.program_id(0), tl.program_id(1)###等价放长了i-t 此时原始句子已经看成 B T*r H BT*r的结果
#     i_b, i_h = i_bh // H, i_bh % H
#     if IS_VARLEN:
#         i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
#         bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
#         T = eos - bos
#     else:
#         bos, eos = i_b * T * r, i_b * T * r + T * r

#     A = A + (bos * H + i_h) * BT * r 
#     Ad = Ad + (bos * H + i_h) * 16 #return B T*r H 16

#     offset = (i_t * 16) % (BT * r)
#     p_A = tl.make_block_ptr(A, (T*r,BT*r),(H*BT*r,1), (i_t * 16, offset), (16, 16), (1,0))
#     p_Ad = tl.make_block_ptr(Ad, (T*r,16),(H*16,1),(i_t*16,0),(16,16),(1,0))

#     b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
#     b_A = -tl.where((tl.arange(0, 16)[:, None] > tl.arange(0, 16)[None, :]), b_A, 0)
#     o_i = tl.arange(0, 16)
#     for i in range(r, min(16, T*r-i_t*16)):#避免超出范围
#         b_a = -tl.load(A + (i_t * 16 + i) * H * BT * r + offset + o_i)
#         b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
#         mask = o_i == i
#         b_A = tl.where(mask[:, None], b_a, b_A)
#     b_A += o_i[:, None] == o_i[None, :]
#     tl.store(p_Ad, b_A.to(p_Ad.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))

# @triton.heuristics({
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
# })
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [1, 2, 4, 8]
#         for num_stages in [2, 3, 4, 5]
#     ],
#     key=['H', 'BT', 'IS_VARLEN','r'],
# )
# @triton.jit(do_not_specialize=['T'])
# def merge_r1_to_r2_inverse_kernel(
#         A,
#         Ad,
#         Ai,
#         cu_seqlens,
#         chunk_indices,
#         T,
#         r: tl.constexpr,
#         H: tl.constexpr,
#         BT: tl.constexpr,
#         IS_VARLEN: tl.constexpr
# ):
#     i_t, i_bh = tl.program_id(0), tl.program_id(1)
#     offset = ((i_t*16) % BT) *r
#     i_b, i_h = i_bh // H, i_bh % H
#     if IS_VARLEN:
#         i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
#         bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
#         T = eos - bos
#     else:
#         bos, eos = i_b * T, i_b * T + T

#     A += (bos*r*H + i_h) * BT * r
#     Ad += (bos*r*H + i_h) * 16
#     Ai += (bos*r*H + i_h) * 16 * r

#     p_A21 = tl.make_block_ptr(A,  (T*r,BT*r),(H*BT*r,1) ,(i_t * 16 * r + 16, offset), (16, 16), (1,0))
#     b_A21 = tl.load(p_A21, boundary_check=(0,1)).to(tl.float32)

#     p_Ad11  = tl.make_block_ptr(Ad,(T*r,16),(H*16,1), (i_t * 16 * r, 0), (16,16), (1,0))
#     p_Ad22  = tl.make_block_ptr(Ad,(T*r,16),(H*16,1), (i_t * 16 * r +16 , 0), (16,16), (1,0))

#     p_Ai11 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 * r,     0), (16, 16), (1, 0))
#     p_Ai22 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 * r +16,16), (16, 16), (1, 0))
#     p_Ai21 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 * r +16, 0), (16, 16), (1, 0))

#     Ai11 = tl.load(p_Ad11, boundary_check=(0, 1)).to(tl.float32)
#     Ai22 = tl.load(p_Ad22, boundary_check=(0, 1)).to(tl.float32)
#     Ai21 = -tl.dot(tl.dot(Ai22,b_A21, input_precision='ieee'),Ai11,input_precision='ieee')
#     tl.store(p_Ai11,Ai11.to(p_Ai11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#     tl.store(p_Ai22,Ai22.to(p_Ai22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#     tl.store(p_Ai21,Ai21.to(p_Ai21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))



# @triton.heuristics({
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
# })
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [1, 2, 4, 8]
#         for num_stages in [2, 3, 4, 5]
#     ],
#     key=['H', 'BT', 'IS_VARLEN','r'],
# )
# @triton.jit(do_not_specialize=['T'])
# def merge_r1_to_r4_inverse_kernel(
#         A,
#         Ad,
#         Ai,
#         cu_seqlens,
#         chunk_indices,
#         T,
#         r: tl.constexpr,
#         H: tl.constexpr,
#         BT: tl.constexpr,
#         IS_VARLEN: tl.constexpr 
# ):
#     i_t, i_bh = tl.program_id(0), tl.program_id(1)
#     i_b, i_h = i_bh // H, i_bh % H
#     if IS_VARLEN:
#         i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
#         bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
#         T = eos - bos
#     else:
#         bos, eos = i_b * T, i_b * T + T
#     offset = ((i_t*16*r) % (BT*r)) 
#     A  +=  (bos * r * H +i_h) * BT * r
#     Ai += (bos * r * H +i_h) * 16 * r
#     Ad += (bos * r * H +i_h) * 16


#     p_A21 = tl.make_block_ptr(A, (T*r,BT*r),(H*BT*r,1) ,(i_t * 16 *r +16, offset),    (16, 16), (1,0))
#     p_A31 = tl.make_block_ptr(A, (T*r,BT*r),(H*BT*r,1) ,(i_t * 16 *r +32, offset),    (16, 16), (1,0))
#     p_A32 = tl.make_block_ptr(A, (T*r,BT*r),(H*BT*r,1) ,(i_t * 16 *r +32, offset+16), (16, 16), (1,0))
#     p_A41 = tl.make_block_ptr(A, (T*r,BT*r),(H*BT*r,1) ,(i_t * 16 *r +48, offset),    (16, 16), (1,0))
#     p_A42 = tl.make_block_ptr(A, (T*r,BT*r),(H*BT*r,1) ,(i_t * 16 *r +48, offset+16), (16, 16), (1,0))
#     p_A43 = tl.make_block_ptr(A, (T*r,BT*r),(H*BT*r,1) ,(i_t * 16 *r +48, offset+32), (16, 16), (1,0))
    
#     b_A21 = tl.load(p_A21, boundary_check=(0,1)).to(tl.float32)
#     b_A31 = tl.load(p_A31, boundary_check=(0,1)).to(tl.float32)
#     b_A32 = tl.load(p_A32, boundary_check=(0,1)).to(tl.float32)
#     b_A41 = tl.load(p_A41, boundary_check=(0,1)).to(tl.float32)
#     b_A42 = tl.load(p_A42, boundary_check=(0,1)).to(tl.float32)
#     b_A43 = tl.load(p_A43, boundary_check=(0,1)).to(tl.float32)


#     p_Ad11  = tl.make_block_ptr(Ad ,(T*r,16),(H*16,1), (i_t * 16 *r    , 0), (16,16), (1,0))
#     p_Ad22  = tl.make_block_ptr(Ad ,(T*r,16),(H*16,1), (i_t * 16 *r +16, 0), (16,16), (1,0))
#     p_Ad33  = tl.make_block_ptr(Ad ,(T*r,16),(H*16,1), (i_t * 16 *r +32, 0), (16,16), (1,0))
#     p_Ad44  = tl.make_block_ptr(Ad ,(T*r,16),(H*16,1), (i_t * 16 *r +48, 0), (16,16), (1,0))
#     ###这里是对的


#     p_Ai11 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r, 0),     (16, 16), (1, 0))
#     p_Ai22 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+16, 16), (16, 16), (1, 0))
#     p_Ai33 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+32, 32), (16, 16), (1, 0))
#     p_Ai44 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+48, 48), (16, 16), (1, 0))
    
#     p_Ai21 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+16, 0),  (16, 16), (1, 0))
#     p_Ai31 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+32, 0),  (16, 16), (1, 0))
#     p_Ai32 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+32, 16), (16, 16), (1, 0))
#     p_Ai41 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+48 ,0),  (16, 16), (1, 0))
#     p_Ai42 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+48, 16), (16, 16), (1, 0))
#     p_Ai43 = tl.make_block_ptr(Ai, (T*r,16*r), (H*16*r, 1), (i_t * 16 *r+48, 32), (16, 16), (1, 0))


#     Ai11 = tl.load(p_Ad11, boundary_check=(0, 1)).to(tl.float32)
#     Ai22 = tl.load(p_Ad22, boundary_check=(0, 1)).to(tl.float32)
#     Ai33 = tl.load(p_Ad33, boundary_check=(0, 1)).to(tl.float32)
#     Ai44 = tl.load(p_Ad44, boundary_check=(0, 1)).to(tl.float32)####这里计算应该是对的


#     Ai21 = -tl.dot(tl.dot(Ai22,b_A21, input_precision='ieee'),Ai11,input_precision='ieee')
#     Ai32 = -tl.dot(tl.dot(Ai33,b_A32, input_precision='ieee'),Ai22,input_precision='ieee')
#     Ai43 = -tl.dot(tl.dot(Ai44,b_A43, input_precision='ieee'),Ai33,input_precision='ieee')

#     Ai31 = -tl.dot(
#             Ai33,
#             tl.dot(b_A31,Ai11, input_precision='ieee')+
#             tl.dot(b_A32,Ai21, input_precision='ieee'),
#             input_precision='ieee')

#     Ai42 = -tl.dot(
#             Ai44,
#             tl.dot(b_A42,Ai22, input_precision='ieee')+
#             tl.dot(b_A43,Ai32, input_precision='ieee'),
#             input_precision='ieee')

#     Ai41 = -tl.dot(
#         Ai44,
#         tl.dot(b_A41, Ai11, input_precision='ieee') +
#         tl.dot(b_A42, Ai21, input_precision='ieee') +
#         tl.dot(b_A43, Ai31, input_precision='ieee'),
#         input_precision='ieee'
#     )

#     tl.store(p_Ai11,Ai11.to(p_Ai11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#     tl.store(p_Ai22,Ai22.to(p_Ai22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#     tl.store(p_Ai33,Ai33.to(p_Ai33.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#     tl.store(p_Ai44,Ai44.to(p_Ai44.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#     tl.store(p_Ai21,Ai21.to(p_Ai21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#     tl.store(p_Ai31,Ai31.to(p_Ai31.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#     tl.store(p_Ai32,Ai32.to(p_Ai32.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#     tl.store(p_Ai41,Ai41.to(p_Ai41.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#     tl.store(p_Ai42,Ai42.to(p_Ai42.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#     tl.store(p_Ai43,Ai43.to(p_Ai43.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))

# @triton.heuristics({
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
# })
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [1, 2, 4, 8]
#         for num_stages in [2, 3, 4, 5]
#     ],
#     key=['H', 'BT', 'IS_VARLEN','r'],
# )
# @triton.jit
# def merge_r4_to_r8_inverse_kernel(
#         A,###B H T 8 BT 8
#         Ad,###B H T 8 16 8
#         Ai,###B H T 8 16 4
#         cu_seqlens,
#         chunk_indices,
#         T,
#         r: tl.constexpr,
#         H: tl.constexpr,
#         BT: tl.constexpr,
#         IS_VARLEN: tl.constexpr 
# ):

#     i_t, i_bh = tl.program_id(0), tl.program_id(1)
#     i_b, i_h = i_bh // H, i_bh % H
#     if IS_VARLEN:
#         i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
#         bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
#         T = eos - bos
#     else:
#         bos, eos = i_b * T, i_b * T + T
#     offset = ((i_t*16*8) % (BT*8)) 
#     A  +=  (bos * r * H +i_h) * BT * r
#     Ai += (bos * r * H +i_h) * 16 * 8
#     Ad += (bos * r * H +i_h) * 16 * 4

#     p_A21 = tl.make_block_ptr(A,   (T*r,BT*r),(H*BT*r,1) ,    (i_t * 16 * 8 + 64, offset), (64, 64), (1,0))
#     b_A21 = tl.load(p_A21, boundary_check=(0,1)).to(tl.float32)

#     p_Ad11  = tl.make_block_ptr(Ad ,(T*r,16*4),(H*16*4,1),  (i_t * 16 * 8,  0),      (64,64), (1,0))
#     p_Ad22  = tl.make_block_ptr(Ad ,(T*r,16*4),(H*16*4,1),  (i_t * 16 * 8 + 64 , 0), (64,64), (1,0))

#     p_Ai11 = tl.make_block_ptr(Ai, (T*r,16*8), (H*16*8, 1), (i_t * 16 * 8,     0),   (64, 64), (1, 0))
#     p_Ai22 = tl.make_block_ptr(Ai, (T*r,16*8), (H*16*8, 1), (i_t * 16 * 8 +64,64),   (64, 64), (1, 0))
#     p_Ai21 = tl.make_block_ptr(Ai, (T*r,16*8), (H*16*8, 1), (i_t * 16 * 8 +64, 0),   (64, 64), (1, 0))

#     Ai11 = tl.load(p_Ad11, boundary_check=(0, 1)).to(tl.float32)
#     Ai22 = tl.load(p_Ad22, boundary_check=(0, 1)).to(tl.float32)
#     Ai21 = -tl.dot(tl.dot(Ai22,b_A21, input_precision='ieee'),Ai11,input_precision='ieee')
#     tl.store(p_Ai11,Ai11.to(p_Ai11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#     tl.store(p_Ai22,Ai22.to(p_Ai22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#     tl.store(p_Ai21,Ai21.to(p_Ai21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))




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
    # o_i = tl.arange(0, 16)

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




# @input_guard
# def solve_tril(
#     A: torch.Tensor,
#     mask: torch.Tensor,
#     cu_seqlens: Optional[torch.Tensor] = None,
#     output_dtype: torch.dtype = torch.float):
#     B,T,H,BT,_,_= A.shape
#     r = mask.shape[-1]
#     A = rearrange(A,'b t h l c r->b (t c) h (l r)').contiguous()#BT*r BT*r  

#     chunk_indices = prepare_chunk_indices(r*cu_seqlens, 16) if cu_seqlens is not None else None
#     N_64 = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(r*T,16)

#     Ad_1 = torch.empty(B,T,r,H,16,device=A.device, dtype=torch.float)
#     solve_tril_16x16_kernel_org[(N_64, B*H)](
#             A=A,Ad=Ad_1,
#             cu_seqlens=r*cu_seqlens if cu_seqlens is not None else None,
#             chunk_indices=chunk_indices,
#             T=T,
#             r=r, BT=BT,H=H,
#     )
#     chunk_indices = prepare_chunk_indices(cu_seqlens, 16) if cu_seqlens is not None else None
#     NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T,16)

#     Ad = torch.zeros(B,T,r,H,16,r,device=A.device, dtype=torch.float if BT != 16 else output_dtype)
#     if r==1:
#         Ad = Ad_1    
#     if r==2:
#         merge_r1_to_r2_inverse_kernel[(NT, B*H)](
#             A=A,Ad=Ad_1,Ai=Ad,
#             cu_seqlens=cu_seqlens,
#             chunk_indices=chunk_indices,
#             T=T,r=r,BT=BT,H=H
#         )
#     if r==4:
#         merge_r1_to_r4_inverse_kernel[(NT, B*H)](
#             A=A,Ad=Ad_1,Ai=Ad,
#             cu_seqlens=cu_seqlens,
#             chunk_indices=chunk_indices,
#             T=T,r=r,BT=BT,H=H
#         )
#     if r==8:
#         Ad = torch.zeros(B,T*r,H,16*4,device=A.device, dtype=torch.float if BT != 16 else output_dtype)###根据r考虑如何merge
#         merge_r1_to_r4_inverse_kernel[(2*NT, B*H)](
#             A=A,Ad=Ad_1,Ai=Ad,
#             cu_seqlens=cu_seqlens,
#             chunk_indices=chunk_indices,
#             T=2*T,r=4,BT=2*BT,H=H
#         )
#         Ad1 = torch.zeros(B,H,NT*r*16,16*r,device=A.device, dtype=torch.float if BT != 16 else output_dtype)###根据r考虑如何merge
#         merge_r4_to_r8_inverse_kernel[(NT,B*H)](
#             # A,Ad,Ad1,
#             A=A,Ad=Ad,Ai=Ad1,
#             cu_seqlens=cu_seqlens,
#             chunk_indices=chunk_indices,
#             T=T,r=8,BT=BT,H=H,
#         )

#     if BT == 16: 
#         if r==8: 
#             return Ad1                                                                                                                         
#         return Ad

#     if chunk_indices is None and cu_seqlens is not None:
#         chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
#     NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
#     Ai = torch.zeros(B,T*r,H,BT*r,device=A.device, dtype=output_dtype)

#     if BT == 32:
#         merge_16x16_to_32x32_inverse_kernel[(NT, B*H)](
#             A=A,Ad=Ad,Ai=Ai,
#             cu_seqlens=cu_seqlens,
#             chunk_indices=chunk_indices,
#             T=T,r=r,BT=BT,H=H
#         )
#         return Ai

#     if BT == 64:
#         merge_16x16_to_64x64_inverse_kernel[(NT, B*H)](
#             A=A,Ad=Ad,Ai=Ai,
#             cu_seqlens=cu_seqlens,
#             chunk_indices=chunk_indices,
#             T=T,r=r,BT=BT,H=H
#         )
#         return Ai



@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN','r'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def mask_wu_fwd_kernel(
    w,
    u,
    ag,
    v,
    mask,
    A_ab_inv,
    A_ak,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    r: tl.constexpr,
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
    o_s = tl.arange(0, BT)

    p_A_ab_inv = tl.make_block_ptr(A_ab_inv + (bos*H*r + i_h) * BT*r, (T*r, BT*r), (H*BT*r, 1), (i_t * BT*r, 0), (BT*r, BT*r), (1, 0))
    p_A_ak = tl.make_block_ptr(A_ak + (bos*H + i_h) * BT*r*r, (T, BT,r,r), (H*BT*r*r, r*r, r, 1), (i_t * BT, 0, 0, 0), (BT, BT,r,r), (3,2, 1, 0))
    ####A ak

    b_Aab_inv = tl.load(p_A_ab_inv, boundary_check=(0, 1))
    b_Aab_inv = tl.reshape(b_Aab_inv,(BT,r,BT,r))
    b_Aab_inv = tl.where((o_s[:, None] >= o_s[None, :])[:,None,:,None], b_Aab_inv, 0)###如何mask
    b_Aab_inv = tl.reshape(b_Aab_inv,(BT*r,BT*r))

    b_Aak = tl.load(p_A_ak,boundary_check=(0,1,2,3))
    b_Aak = tl.where((o_s[:, None] > o_s[None, :])[:,:,None,None], b_Aak, 0)    ###如何mask
    b_Aak2 = tl.reshape(tl.permute(tl.sum(b_Aak, -1),(0,2,1)),(BT*r,BT)) ###BT BT r
    b_Aak2 = tl.dot(b_Aab_inv,b_Aak2)
    b_Aak2 = b_Aak2.to(v.dtype.element_ty, fp_downcast_rounding="rtne")###BT*r BT
    dk = K//r

    b_Aab_inv = b_Aab_inv.to(ag.dtype.element_ty, fp_downcast_rounding="rtne")
    for i_r in range(r):
        p_maskr = tl.make_block_ptr(mask + (bos*H + i_h)*r*r, (T,r,r),(H*r*r,r,1), (i_t*BT,0,i_r),(BT,r,1),(2,1,0))
        b_maskr = tl.load(p_maskr,boundary_check=(0,1,2))#BT,r,1
        for i_k in range(tl.cdiv(dk, BK)):
            p_ag = tl.make_block_ptr(ag + (bos*H + i_h) * K, (T,K), (H*K,1), (i_t * BT, i_r*dk + i_k * BK), (BT,BK), (1, 0))
            b_ag = tl.load(p_ag, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w + (bos*r*H + i_h) * K, (T*r,K), (H*K,1), (i_t * BT * r,i_r*dk + i_k * BK), (BT*r,BK), (1, 0))
            b_agbm = b_ag[:,None,:]*b_maskr.to(b_ag.dtype)#BT r BK
            b_agbm = tl.reshape(b_agbm,(BT*r,BK)) 
            b_w = tl.dot(b_Aab_inv, b_agbm)  # both bf16 or fp16
            tl.store(p_w, b_w.to(p_w.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos*r*H + i_h) * V, (T*r, V), (H*V, 1), (i_t * BT * r, i_v * BV), (BT*r, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_u = (tl.dot(b_Aak2, b_v))  # both bf16 or fp16
        tl.store(p_u, b_u.to(p_u.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


def mask_wu_fwd(
    ag: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    A_ak: torch.Tensor,
    A_ab_inv: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *ag.shape, v.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    r = mask.shape[-1]
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(triton.next_power_of_2(K//r), 64)
    BV = min(triton.next_power_of_2(V), 64)

    w = torch.empty(B,T,r,H,K,device = ag.device,dtype = ag.dtype)
    u = torch.empty(B,T,r,H,V,device = v.device,dtype = v.dtype)
    mask_wu_fwd_kernel[(NT, B * H)](
        ag=ag,
        v=v,
        A_ak=A_ak,
        A_ab_inv=A_ab_inv,
        w=w,
        u=u,
        mask=mask,
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
    return w, u

def ceildiv(a, b):
    return -(a // -b)

def mask_prepare_wy_repr_fwd(
    ag: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    A_ak: torch.Tensor,
    A_ab: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, _ = ag.shape
    BT = 16
    r = mask.shape[-1]
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    A_ab_inv = torch.zeros(B,T,r,H,BT,r,device=A_ab.device,dtype=torch.float)
    A_ab = -A_ab.contiguous()
    solve_tril_16x16_kernel[(triton.cdiv(T, 16), B * H)](
        A=A_ab,
        Ad=A_ab_inv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        r=r,
        BT=16,
        IS_VARLEN=cu_seqlens is not None,
    )
    w, u = mask_wu_fwd(
        ag=ag,
        mask=mask,
        v=v,
        A_ak=A_ak,
        A_ab_inv=A_ab_inv,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )
    return w, u, A_ab_inv

