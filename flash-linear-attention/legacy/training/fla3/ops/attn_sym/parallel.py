# # -*- coding: utf-8 -*-
# # Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# import warnings
# from typing import Optional

# import torch
# import triton
# import triton.language as tl
# from einops import rearrange, reduce

# from fla.ops.utils import prepare_chunk_indices
# from fla.ops.utils.cumsum import chunk_global_cumsum
# from ...ops.utils.op import exp, log, safe_exp
# from fla.utils import autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, contiguous


# @triton.heuristics({
#     'USE_G': lambda args: args['g_cumsum'] is not None,
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
#     'USE_ALPHA': lambda args: args['alpha'] is not None,
# })
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [1,2,4] + ([8] if check_shared_mem('hopper') else [])
#         for num_stages in [2,3,4,5]
#     ],
#     key=['B', 'H', 'HQ', 'G', 'K', 'V', 'BK', 'BV', 'USE_G', 'IS_VARLEN', 'USE_ALPHA'],
# )
# @triton.jit
# def parallel_attn_fwd_kernel(
#     q,
#     k,
#     v,
#     o,
#     g_cumsum,
#     lse,
#     alpha,
#     scale,
#     cu_seqlens,
#     chunk_indices,
#     T,
#     B: tl.constexpr,
#     H: tl.constexpr,
#     HQ: tl.constexpr,
#     G: tl.constexpr,
#     K: tl.constexpr,
#     V: tl.constexpr,
#     BT: tl.constexpr,
#     BS: tl.constexpr,
#     BK: tl.constexpr,
#     BV: tl.constexpr,
#     USE_G: tl.constexpr,
#     IS_VARLEN: tl.constexpr,
#     USE_ALPHA: tl.constexpr,
# ):
#     i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
#     i_b, i_hq = i_bh // HQ, i_bh % HQ
#     i_h = i_hq // G

#     if IS_VARLEN:
#         i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
#         bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
#         T = eos - bos
#     else:
#         i_n = i_b
#         bos, eos = i_n * T, i_n * T + T
#     # the Q block is kept in the shared memory throughout the whole kernel
#     if USE_ALPHA:
#         p_q_even = tl.make_block_ptr(q + (bos * HQ +      i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
#         p_q_odd = tl.make_block_ptr (q + (bos * HQ + HQ + i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
#         b_q_even = tl.load(p_q_even, boundary_check=(0, 1))
#         b_q_odd = tl.load(p_q_odd, boundary_check=(0, 1))
#         b_q_even = (b_q_even * scale).to(b_q_even.dtype)
#         b_q_odd = (b_q_odd * scale).to(b_q_odd.dtype)
#     else:
#         p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
#         b_q = tl.load(p_q, boundary_check=(0, 1))
#         b_q = (b_q * scale).to(b_q.dtype)
    
#     if USE_ALPHA:
#         p_alpha = alpha + i_h * K + tl.arange(0, BK)
#         m_alpha = tl.arange(0, BK) < K  
#         b_alpha = tl.load(p_alpha, mask=m_alpha, other=0).to(b_q_even.dtype)[:,None]#[BK],仅奇偶不同处load如此
#     p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
#     p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
#     # [BT, BV]
#     b_o = tl.zeros([BT, BV], dtype=tl.float32)

#     b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
#     b_acc = tl.zeros([BT], dtype=tl.float32)

#     if USE_G:
#         p_g = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
#         b_gq = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
#     else:
#         b_gq = None

#     for i_s in range(0, i_t * BT, BS):
#         if USE_ALPHA:
#             p_k_even = tl.make_block_ptr(k + (bos * H +     i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
#             p_k_odd =  tl.make_block_ptr(k + (bos * H + H + i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
#             b_k_even = tl.load(p_k_even, boundary_check=(0, 1))
#             b_k_odd = tl.load(p_k_odd, boundary_check=(0, 1))
#         else:
#             p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
#             b_k = tl.load(p_k, boundary_check=(0, 1))
#         p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
#         # [BS, BV]
#         b_v = tl.load(p_v, boundary_check=(0, 1))
#         # [BT, BS]
#         if USE_ALPHA:
#             b_s_ee = tl.dot(b_q_even, b_k_even)
#             b_s_eo = tl.dot(b_q_even, b_k_odd*b_alpha)
#             b_s_oe = tl.dot(b_q_odd, b_k_even*b_alpha)
#             b_s_oo = tl.dot(b_q_odd, b_k_odd)
#             b_s_e = tl.join(b_s_ee,b_s_oe)
#             b_s_o = tl.join(b_s_eo,b_s_oo)
#             b_s = tl.join(b_s_e,b_s_o) ###BT BS 2 2
#             b_s = tl.permute(b_s,0,2,1,3).reshape(BT,BS)
#         else:
#             b_s = tl.dot(b_q, b_k)

#         if USE_G:
#             p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
#             b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
#             b_s += b_gq[:, None] - b_gk[None, :]

#         # [BT, BS]
#         b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
#         b_r = exp(b_mp - b_m)
#         # [BT, BS]
#         b_p = safe_exp(b_s - b_m[:, None])
#         # [BT]
#         b_acc = b_acc * b_r + tl.sum(b_p, 1)
#         # [BT, BV]
#         if USE_ALPHA:
#             b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q_even.dtype), b_v)
#         else:
#             b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

#         b_mp = b_m

#     # [BT]
#     o_q = i_t * BT + tl.arange(0, BT)
#     for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
#         if USE_ALPHA:
#             p_k_even = tl.make_block_ptr(k + (bos * H +     i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
#             p_k_odd =  tl.make_block_ptr(k + (bos * H + H + i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
#             b_k_even = tl.load(p_k_even, boundary_check=(0, 1))
#             b_k_odd = tl.load(p_k_odd, boundary_check=(0, 1))
#         else:
#             p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
#             # [BK, BS]
#             b_k = tl.load(p_k, boundary_check=(0, 1))
#         p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

#         # [BS]
#         o_k = i_s + tl.arange(0, BS)
#         # [BS, BV]
#         b_v = tl.load(p_v, boundary_check=(0, 1))
#         # [BT, BS]
#         if USE_ALPHA:
#             b_s_ee = tl.dot(b_q_even, b_k_even)
#             b_s_eo = tl.dot(b_q_even, b_k_odd*b_alpha)
#             b_s_oe = tl.dot(b_q_odd, b_k_even*b_alpha)
#             b_s_oo = tl.dot(b_q_odd, b_k_odd)

#             b_s_e = tl.join(b_s_ee,b_s_oe)
#             b_s_o = tl.join(b_s_eo,b_s_oo)
#             b_s = tl.join(b_s_e,b_s_o) ###BT BS 2 2
#             b_s = tl.permute(b_s,0,2,1,3).reshape(BT,BS)
            
#             # b_s = tl.dot(b_q_even, b_k_even)[:,None,:,None]*b00[None,:,None,:] ##[BT//2, BS//2] BS//2*2*BS//2*2
#             # b_s += tl.dot(b_q_even, b_k_odd*b_alpha)[:,None,:,None]*b01[None,:,None,:] ##[BT//2, BS//2] BS//2*2*BS//2*2
#             # b_s += tl.dot(b_q_odd, b_k_even*b_alpha)[:,None,:,None]*b10[None,:,None,:] ##[BT//2, BS//2] BS//2*2*BS//2*2
#             # b_s += tl.dot(b_q_odd, b_k_odd)[:,None,:,None]*b11[None,:,None,:] ##[BT//2, BS//2] BS//2*2*BS//2*2
#             # b_s = b_s.reshape(BT,BS)
#         else:
#             b_s = tl.dot(b_q, b_k)
#         b_s = tl.where(o_q[:, None] >= o_k[None, :], b_s, float('-inf'))

#         if USE_G:
#             p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
#             b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
#             b_s += b_gq[:, None] - b_gk[None, :]

#         # [BT]
#         b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
#         b_r = exp(b_mp - b_m)
#         # [BT, BS]
#         b_p = safe_exp(b_s - b_m[:, None])
#         # [BT]
#         b_acc = b_acc * b_r + tl.sum(b_p, 1)
#         # [BT, BV]
#         if USE_ALPHA:
#             b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q_even.dtype), b_v)
#         else:
#             b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
#         b_mp = b_m

#     b_o = b_o / b_acc[:, None]
#     b_m += log(b_acc)
#     tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
#     tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))


# @triton.jit
# def parallel_attn_bwd_kernel_preprocess(
#     o,
#     do,
#     delta,
#     B: tl.constexpr,
#     V: tl.constexpr
# ):
#     i_n = tl.program_id(0)
#     o_d = tl.arange(0, B)
#     m_d = o_d < V

#     b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
#     b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
#     b_delta = tl.sum(b_o * b_do)

#     tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))

# #####################num_stages=2 仍然超
# ####等于2比较极限，或许可以尝试优化一下以使得可用
# @triton.heuristics({
#     'USE_G': lambda args: args['g_cumsum'] is not None,
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
#     'USE_ALPHA': lambda args: args['alpha'] is not None,
# })
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [1,2,4] + ([8] if check_shared_mem('hopper') else [])
#         # for num_stages in [1,2]
#         for num_stages in [2,3,4,5]
#     ],
#     key=['B', 'H', 'HQ', 'G', 'K', 'V', 'BK', 'BV', 'USE_G', 'IS_VARLEN', 'USE_ALPHA'],
# )
# @triton.jit(do_not_specialize=['T'])
# def parallel_attn_bwd_kernel_dq(
#     q,
#     k,
#     v,
#     alpha,
#     lse,
#     delta,
#     do,
#     dq,
#     dg_cumsum,
#     g_cumsum,
#     scale,
#     cu_seqlens,
#     chunk_indices,
#     T,
#     B: tl.constexpr,
#     H: tl.constexpr,
#     HQ: tl.constexpr,
#     G: tl.constexpr,
#     K: tl.constexpr,
#     V: tl.constexpr,
#     BT: tl.constexpr,
#     BS: tl.constexpr,
#     BK: tl.constexpr,
#     BV: tl.constexpr,
#     IS_VARLEN: tl.constexpr,
#     USE_G: tl.constexpr,
#     USE_ALPHA: tl.constexpr,
# ):
#     i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
#     i_b, i_hq = i_bh // HQ, i_bh % HQ
#     i_h = i_hq // G

#     if IS_VARLEN:
#         i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
#         bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
#         T = eos - bos
#     else:
#         i_n = i_b
#         bos, eos = i_n * T, i_n * T + T
#     if USE_ALPHA:###[BT BK]
#         p_q_even = tl.make_block_ptr (q + (bos * HQ +      i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
#         p_q_odd  = tl.make_block_ptr (q + (bos * HQ + HQ + i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
#         b_q_even = tl.load(p_q_even, boundary_check=(0, 1))
#         b_q_odd = tl.load(p_q_odd, boundary_check=(0, 1))
#         b_q_even = (b_q_even * scale).to(b_q_even.dtype)
#         b_q_odd = (b_q_odd * scale).to(b_q_odd.dtype)
#         p_dq_even = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K,      (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
#         p_dq_odd =  tl.make_block_ptr(dq + (bos * HQ + HQ + i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
#     else:
#         p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
#         b_q = tl.load(p_q, boundary_check=(0, 1))
#         b_q = (b_q * scale).to(b_q.dtype)
#         p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0)) ####这个思考一下比例
    
#     if USE_ALPHA:
#         p_alpha = alpha + i_h * K + tl.arange(0, BK)
#         m_alpha = tl.arange(0, BK) < K  
#         b_alpha = tl.load(p_alpha, mask=m_alpha, other=0).to(b_q_even.dtype)[:,None]#[BK,BT],仅奇偶不同处load如此
#         # a = tl.arange(0,2)
#         # b00 = (a<1)[:,None] & (a<1)[None,:]
#         # b01 = (a<1)[:,None] & (a>=1)[None,:]
#         # b10 = (a>=1)[:,None] & (a<1)[None,:]
#         # b11 = (a>=1)[:,None] & (a>=1)[None,:]
#         # b00 = b00.to(b_q_even.dtype)
#         # b01 = b01.to(b_q_even.dtype)
#         # b10 = b10.to(b_q_even.dtype)
#         # b11 = b11.to(b_q_even.dtype)
#     p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
#     p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
#     p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
#     # [BT, BV]
#     b_do = tl.load(p_do, boundary_check=(0, 1))
#     # [BT]
#     b_lse = tl.load(p_lse, boundary_check=(0,))
#     b_delta = tl.load(p_delta, boundary_check=(0,))

#     # [BT, BK]
#     if USE_ALPHA:
#         b_dq_even = tl.zeros([BT//2, BK], dtype=tl.float32)
#         b_dq_odd = tl.zeros([BT//2, BK], dtype=tl.float32)
#     else:
#         b_dq = tl.zeros([BT, BK], dtype=tl.float32)
#     if USE_G:
#         b_dg = tl.zeros([BT, ], dtype=tl.float32)
#         p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
#         b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
#     else:
#         b_gq = None
#         b_dg = None
#     for i_s in range(0, i_t * BT, BS):###分前后段运算，二者有所区别
#         if USE_ALPHA:
#             p_k_even = tl.make_block_ptr(k + (bos * H +     i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
#             p_k_odd =  tl.make_block_ptr(k + (bos * H + H + i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
#             b_k_even = tl.load(p_k_even, boundary_check=(0, 1))
#             b_k_odd = tl.load(p_k_odd, boundary_check=(0, 1))
#         else:
#             p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
#             b_k = tl.load(p_k, boundary_check=(0, 1))
#         p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
#         # [BV, BS]
#         b_v = tl.load(p_v, boundary_check=(0, 1))
#         # [BT, BS]
#         if USE_ALPHA:
#             #####考虑use join替换或许会快，不知道
#             b_s_ee = tl.dot(b_q_even, b_k_even)
#             b_s_eo = tl.dot(b_q_even, b_k_odd*b_alpha)
#             b_s_oe = tl.dot(b_q_odd, b_k_even*b_alpha)
#             b_s_oo = tl.dot(b_q_odd, b_k_odd)

#             b_s_e = tl.join(b_s_ee,b_s_oe)
#             b_s_o = tl.join(b_s_eo,b_s_oo)
#             b_s = tl.join(b_s_e,b_s_o) ###BT BS 2 2
#             b_s = tl.permute(b_s,0,2,1,3).reshape(BT,BS)


#             # b_s =  tl.dot(b_q_even, b_k_even)[:,None,:,None]*b00[None,:,None,:] ##[BT//2, BS//2] BS//2*2*BS//2*2
#             # b_s += tl.dot(b_q_even, b_k_odd*b_alpha)[:,None,:,None]*b01[None,:,None,:] ##[BT//2, BS//2] BS//2*2*BS//2*2
#             # b_s += tl.dot(b_q_odd, b_k_even*b_alpha)[:,None,:,None]*b10[None,:,None,:] ##[BT//2, BS//2] BS//2*2*BS//2*2
#             # b_s += tl.dot(b_q_odd, b_k_odd)[:,None,:,None]*b11[None,:,None,:] ##[BT//2, BS//2] BS//2*2*BS//2*2
#             # b_s = b_s.reshape(BT,BS)
#         else:
#             b_s = tl.dot(b_q, b_k)
#         if USE_G:
#             p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
#             b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
#             b_s += b_gq[:, None] - b_gk[None, :]

#         b_p = safe_exp(b_s - b_lse[:, None])
#         # [BT, BV] @ [BV, BS] -> [BT, BS]
#         b_dp = tl.dot(b_do, b_v)
#         b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])##[BT, BS]
#         # [BT, BS] @ [BS, BK] -> [BT, BK]
#         if USE_ALPHA:
#             b_ds = tl.permute(b_ds.reshape(BT//2,2,BS//2,2),(0,2,1,3))
#             b_ds_even, b_ds_odd = b_ds.split()
#             b_ds_even_even, b_ds_odd_even = b_ds_even.split()
#             b_ds_even_odd, b_ds_odd_odd = b_ds_odd.split()
#             # b_ds = tl.permute(b_ds.reshape(BT//2,2,BS//2,2),(0,2,1,3)).reshape(BT//2,BS//2,4)
#             # b_ds_even_even = tl.sum(tl.where((tl.arange(0,4)==0)[None,None,:], b_ds, 0), -1)
#             # b_ds_even_odd = tl.sum(tl.where((tl.arange(0,4)==1)[None,None,:], b_ds, 0), -1)###[BT//2, BS//2]
#             # b_ds_odd_even = tl.sum(tl.where((tl.arange(0,4)==2)[None,None,:], b_ds, 0), -1)###[BT//2, BS//2]
#             # b_ds_odd_odd = tl.sum(tl.where((tl.arange(0,4)==3)[None,None,:], b_ds, 0), -1)

#             b_dq_even += tl.dot(b_ds_even_even.to(b_k_even.dtype), tl.trans(b_k_even))
#             b_dq_even += tl.dot(b_ds_even_odd.to(b_k_odd.dtype), tl.trans(b_k_odd*b_alpha))
#             b_dq_odd += tl.dot(b_ds_odd_even.to(b_k_even.dtype), tl.trans(b_k_even*b_alpha))
#             b_dq_odd += tl.dot(b_ds_odd_odd.to(b_k_odd.dtype), tl.trans(b_k_odd))
#         else:
#             b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
#         if USE_G:
#             b_dg += tl.sum(b_ds, 1)

#     # [BT]
#     o_q = i_t * BT + tl.arange(0, BT)
#     for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
#         if USE_ALPHA:
#             p_k_even = tl.make_block_ptr(k + (bos * H +     i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
#             p_k_odd =  tl.make_block_ptr(k + (bos * H + H + i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
#             b_k_even = tl.load(p_k_even, boundary_check=(0, 1))
#             b_k_odd = tl.load(p_k_odd, boundary_check=(0, 1))
#         else:
#             p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
#             b_k = tl.load(p_k, boundary_check=(0, 1))
#         p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
#         # [BS]
#         o_k = i_s + tl.arange(0, BS)
#         # [BV, BS]
#         b_v = tl.load(p_v, boundary_check=(0, 1))
#         # [BT, BS]
#         if USE_ALPHA:
#             b_s_ee = tl.dot(b_q_even, b_k_even)
#             b_s_eo = tl.dot(b_q_even, b_k_odd*b_alpha)
#             b_s_oe = tl.dot(b_q_odd, b_k_even*b_alpha)
#             b_s_oo = tl.dot(b_q_odd, b_k_odd)

#             b_s_e = tl.join(b_s_ee,b_s_oe)
#             b_s_o = tl.join(b_s_eo,b_s_oo)
#             b_s = tl.join(b_s_e,b_s_o) ###BT BS 2 2
#             b_s = tl.permute(b_s,0,2,1,3).reshape(BT,BS)
#         else:
#             b_s = tl.dot(b_q, b_k)

#         if USE_G:
#             p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
#             b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
#             b_s += b_gq[:, None] - b_gk[None, :]
#             b_s = tl.where(o_q[:, None] >= o_k[None, :], b_s, -float('inf'))

#         b_p = safe_exp(b_s - b_lse[:, None])  # SY: important to use safe_exp here to avoid NaN.
#         b_p = tl.where(o_q[:, None] >= o_k[None, :], b_p, 0)

#         # [BT, BV] @ [BV, BS] -> [BT, BS]
#         b_dp = tl.dot(b_do, b_v)
#         b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
#         # [BT, BS] @ [BS, BK] -> [BT, BK]
#         # b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
#         if USE_ALPHA:
#             b_ds = tl.permute(b_ds.reshape(BT//2,2,BS//2,2),(0,2,1,3))
#             b_ds_even, b_ds_odd = b_ds.split()
#             b_ds_even_even, b_ds_odd_even = b_ds_even.split()
#             b_ds_even_odd, b_ds_odd_odd = b_ds_odd.split()############不存在梯度为0的情况
#             b_dq_even += tl.dot(b_ds_even_even.to(b_k_even.dtype), tl.trans(b_k_even))
#             b_dq_even += tl.dot(b_ds_even_odd.to(b_k_odd.dtype), tl.trans(b_k_odd*b_alpha))
#             b_dq_odd += tl.dot(b_ds_odd_even.to(b_k_even.dtype), tl.trans(b_k_even*b_alpha))
#             b_dq_odd += tl.dot(b_ds_odd_odd.to(b_k_odd.dtype), tl.trans(b_k_odd))
#         else:
#             b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
#         if USE_G:
#             b_dg += tl.sum(b_ds, 1)

#     if USE_ALPHA:
#         b_dq_even *= scale
#         b_dq_odd *= scale
#         tl.store(p_dq_even, b_dq_even.to(p_dq_even.dtype.element_ty), boundary_check=(0, 1))
#         tl.store(p_dq_odd, b_dq_odd.to(p_dq_odd.dtype.element_ty), boundary_check=(0, 1))
#     else:
#         b_dq *= scale
#         tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
#     if USE_G:
#         p_dg = tl.make_block_ptr(dg_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
#         tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


# @triton.heuristics({
#     'USE_G': lambda args: args['g_cumsum'] is not None,
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
#     'USE_ALPHA': lambda args: args['alpha'] is not None,
# })
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [1,2,4] + ([8] if check_shared_mem('hopper') else [])
#         # for num_stages in [1]
#         for num_stages in [2,3,4,5]
#     ],
#     key=['B', 'H', 'HQ', 'G', 'K', 'V', 'BK', 'BV', 'USE_G', 'IS_VARLEN', 'USE_ALPHA'],
# )
# @triton.jit(do_not_specialize=['T'])
# def parallel_attn_bwd_kernel_dkv(
#     q,
#     k,
#     v,
#     alpha,
#     g_cumsum,
#     lse,
#     delta,
#     dalpha,
#     do,
#     dk,
#     dv,
#     dg_cumsum,
#     cu_seqlens,
#     chunk_indices,
#     scale,
#     T,
#     NT,
#     B: tl.constexpr,
#     H: tl.constexpr,
#     HQ: tl.constexpr,
#     G: tl.constexpr,
#     K: tl.constexpr,
#     V: tl.constexpr,
#     BT: tl.constexpr,
#     BS: tl.constexpr,
#     BK: tl.constexpr,
#     BV: tl.constexpr,
#     USE_G: tl.constexpr,
#     IS_VARLEN: tl.constexpr,
#     USE_ALPHA:tl.constexpr,
# ):
#     i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
#     i_b, i_hq = i_bh // HQ, i_bh % HQ
#     i_h = i_hq // G

#     if IS_VARLEN:
#         i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
#         bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
#         T = eos - bos
#     else:
#         i_n = i_b
#         bos, eos = i_n * T, i_n * T + T

#     if USE_ALPHA:
#         p_k_even = tl.make_block_ptr(k + (bos * H +     i_h) * K, (T//2, K), (2*H*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
#         p_k_odd =  tl.make_block_ptr(k + (bos * H + H + i_h) * K, (T//2, K), (2*H*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
#         p_dk_even = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K,      (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
#         p_dk_odd =  tl.make_block_ptr(dk + (bos * HQ + HQ + i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
#         b_k_even = tl.load(p_k_even, boundary_check=(0, 1))
#         b_k_odd = tl.load(p_k_odd, boundary_check=(0, 1))
#     else:
#         p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
#         p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
#         b_k = tl.load(p_k, boundary_check=(0, 1))
#     p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
#     p_dv = tl.make_block_ptr(dv + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

#     if USE_ALPHA:
#         p_alpha = alpha + i_h * K + tl.arange(0, BK)
#         m_alpha = tl.arange(0, BK) < K  
#         b_alpha = tl.load(p_alpha, mask=m_alpha, other=0).to(b_k_even.dtype)[None,:]#[1,BK],仅奇偶不同处load如此
#     if USE_ALPHA:
#         b_dk_even = tl.zeros([BT//2, BK], dtype=tl.float32)###almost 130KB for stages=1
#         b_dk_odd = tl.zeros([BT//2, BK], dtype=tl.float32)
#         b_dalpha = tl.zeros([BK], dtype=tl.float32)
#     else:
#         b_dk = tl.zeros([BT, BK], dtype=tl.float32)
#     # [BT, BV]
#     b_v = tl.load(p_v, boundary_check=(0, 1))
#     b_dv = tl.zeros([BT, BV], dtype=tl.float32)####同样130KB OOM了
#     o_k = i_t * BT + tl.arange(0, BT)
#     if USE_G:
#         p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
#         b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
#         b_dg = tl.zeros([BT,], dtype=tl.float32)
#     else:
#         b_gk = None
#         b_dg = None

#     for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
#         if USE_ALPHA:
#             p_q_even = tl.make_block_ptr(q + (bos * HQ + i_hq) * K,    (T//2, K), (2*HQ*K, 1), (i_s//2, 0), (BS//2, BK), (1, 0))
#             p_q_odd =  tl.make_block_ptr(q + (bos * HQ + HQ + i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_s//2, 0), (BS//2, BK), (1, 0))
#             b_q_even = tl.load(p_q_even, boundary_check=(0, 1))
#             b_q_odd = tl.load(p_q_odd, boundary_check=(0, 1))
#             b_q_even = (b_q_even * scale).to(b_q_even.dtype)
#             b_q_odd = (b_q_odd * scale).to(b_q_odd.dtype)
#         else:
#             p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
#             b_q = tl.load(p_q, boundary_check=(0, 1))
#             b_q = (b_q * scale).to(b_q.dtype)
#         p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
#         p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
#         p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
#         # [BS]
#         o_q = i_s + tl.arange(0, BS)
#         # [BS, BV]
#         b_do = tl.load(p_do, boundary_check=(0, 1))
#         # [BS]
#         b_lse = tl.load(p_lse, boundary_check=(0,))
#         b_delta = tl.load(p_delta, boundary_check=(0,))
#         # [BT, BS]
#         if USE_ALPHA:
#             b_s_ee = tl.dot(b_k_even,          tl.trans(b_q_even))
#             b_s_eo = tl.dot(b_k_even*b_alpha, tl.trans(b_q_odd))
#             b_s_oe = tl.dot(b_k_odd*b_alpha,  tl.trans(b_q_even))
#             b_s_oo = tl.dot(b_k_odd,          tl.trans(b_q_odd))
#             b_s_e = tl.join(b_s_ee,b_s_oe)
#             b_s_o = tl.join(b_s_eo,b_s_oo)
#             b_s = tl.join(b_s_e,b_s_o) ###BT BS 2 2
#             b_s = tl.permute(b_s,0,2,1,3).reshape(BT,BS)
#         else:
#             b_s = tl.dot(b_k, tl.trans(b_q))###[BT, BS]

#         if USE_G:
#             p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
#             b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
#             b_s += b_gq[None, :] - b_gk[:, None]
#             b_s = tl.where(o_k[:, None] <= o_q[None, :], b_s, -float('inf'))
#         b_p = safe_exp(b_s - b_lse[None, :])
#         b_p = tl.where(o_k[:, None] <= o_q[None, :], b_p, 0)
#         # [BT, BS] @ [BS, BV] -> [BT, BV]
#         b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
#         # [BT, BV] @ [BV, BS] -> [BT, BS]
#         b_dp = tl.dot(b_v, tl.trans(b_do))
#         # [BT, BS]
#         b_ds = b_p * (b_dp - b_delta[None, :])
#         # [BT, BS] @ [BS, BK] -> [BT, BK]
#         if USE_ALPHA:
#             b_ds = tl.permute(b_ds.reshape(BT//2,2,BS//2,2),(0,2,1,3))
#             b_ds_even, b_ds_odd = b_ds.split()
#             b_ds_even_even, b_ds_odd_even = b_ds_even.split()
#             b_ds_even_odd, b_ds_odd_odd = b_ds_odd.split()

#             b_dk_even += tl.dot(b_ds_even_even.to(b_q_even.dtype), (b_q_even))
#             b_dk_odd += tl.dot(b_ds_odd_odd.to(b_q_odd.dtype), (b_q_odd))            
#             b_dk_even_alpha = tl.dot(b_ds_even_odd.to(b_q_odd.dtype), (b_q_odd))##[BT//2, BK]
#             b_dk_odd_alpha = tl.dot(b_ds_odd_even.to(b_q_even.dtype), (b_q_even))##[BT//2, B=K]
#             b_dk_even += b_dk_even_alpha * b_alpha
#             b_dk_odd += b_dk_odd_alpha * b_alpha
#             b_dalpha += tl.sum(b_dk_even_alpha * b_k_even + b_dk_odd_alpha * b_k_odd,0) ##[BT//2, BK]
#         else:
#             b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
#         if USE_G:
#             b_dg -= tl.sum(b_ds, 1)

#     for i_s in range((i_t + 1) * BT, tl.cdiv(T, BS) * BS, BS):
#         if USE_ALPHA:
#             p_q_even = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_s//2, 0), (BS//2, BK), (1, 0))
#             p_q_odd =  tl.make_block_ptr(q + (bos * HQ + HQ + i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_s//2, 0), (BS//2, BK), (1, 0))
#             b_q_even = tl.load(p_q_even, boundary_check=(0, 1))
#             b_q_odd = tl.load(p_q_odd, boundary_check=(0, 1))
#             b_q_even = (b_q_even * scale).to(b_q_even.dtype)
#             b_q_odd = (b_q_odd * scale).to(b_q_odd.dtype)
#         else:
#             p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
#             b_q = tl.load(p_q, boundary_check=(0, 1))
#             b_q = (b_q * scale).to(b_q.dtype)
#         p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
#         p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
#         p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
#         # [BS]
#         o_q = i_s + tl.arange(0, BS)
#         # [BS, BV]
#         b_do = tl.load(p_do, boundary_check=(0, 1))
#         # [BS]
#         b_lse = tl.load(p_lse, boundary_check=(0,))
#         b_delta = tl.load(p_delta, boundary_check=(0,))
#         # [BT, BS]
#         if USE_ALPHA:
#             b_s_ee = tl.dot(b_k_even,         tl.trans(b_q_even))
#             b_s_eo = tl.dot(b_k_even*b_alpha, tl.trans(b_q_odd))
#             b_s_oe = tl.dot(b_k_odd*b_alpha,  tl.trans(b_q_even))
#             b_s_oo = tl.dot(b_k_odd,          tl.trans(b_q_odd))

#             b_s_e = tl.join(b_s_ee,b_s_oe)
#             b_s_o = tl.join(b_s_eo,b_s_oo)
#             b_s = tl.join(b_s_e,b_s_o) ###BT BS 2 2
#             b_s = tl.permute(b_s,0,2,1,3).reshape(BT,BS)
#             # b_s =  tl.dot(b_k_even, tl.trans(b_q_even))[:,None,:,None]*b00[None,:,None,:] ##[BT//2, BS//2] BS//2*2*BS//2*2
#             # b_s += tl.dot(b_k_even*b_alpha, tl.trans(b_q_odd))[:,None,:,None]*b01[None,:,None,:] ##[BT//2, BS//2] BS//2*2*BS//2*2
#             # b_s += tl.dot(b_k_odd*b_alpha, tl.trans(b_q_even))[:,None,:,None]*b10[None,:,None,:] ##[BT//2, BS//2] BS//2*2*BS//2*2
#             # b_s += tl.dot(b_k_odd, tl.trans(b_q_odd))[:,None,:,None]*b11[None,:,None,:] ##[BT//2, BS//2] BS//2*2*BS//2*2
#             # b_s = b_s.reshape(BT,BS)
#         else:
#             b_s = tl.dot(b_k, tl.trans(b_q))###[BT, BS]

#         if USE_G:
#             p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
#             b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
#             b_s += b_gq[None, :] - b_gk[:, None]
#         b_p = safe_exp(b_s - b_lse[None, :])###BT BT
#         # [BT, BS] @ [BS, BV] -> [BT, BV]
#         b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
#         # [BT, BV] @ [BV, BS] -> [BT, BS]
#         b_dp = tl.dot(b_v, tl.trans(b_do))
#         # [BT, BS]
#         b_ds = b_p * (b_dp - b_delta[None, :])
#         # [BT, BS] @ [BS, BK] -> [BT, BK]

#         if USE_ALPHA:
#             b_ds = tl.permute(b_ds.reshape(BT//2,2,BS//2,2),(0,2,1,3))
#             b_ds_even, b_ds_odd = b_ds.split()
#             b_ds_even_even, b_ds_odd_even = b_ds_even.split()
#             b_ds_even_odd, b_ds_odd_odd = b_ds_odd.split()
#             # b_ds_even_even, b_ds_even_odd = b_ds_even.split()
#             # b_ds_odd_even, b_ds_odd_odd = b_ds_odd.split()
#             # b_ds_even_even = tl.sum(tl.where((tl.arange(0,4)==0)[None,None,:], b_ds, 0), -1)
#             # b_ds_even_odd = tl.sum(tl.where((tl.arange(0,4)==1)[None,None,:], b_ds, 0), -1)###[BT//2, BS//2]
#             # b_ds_odd_even = tl.sum(tl.where((tl.arange(0,4)==2)[None,None,:], b_ds, 0), -1)###[BT//2, BS//2]
#             # b_ds_odd_odd = tl.sum(tl.where((tl.arange(0,4)==3)[None,None,:], b_ds, 0), -1)
#             b_dk_even += tl.dot(b_ds_even_even.to(b_q_even.dtype), (b_q_even))
#             b_dk_odd += tl.dot(b_ds_odd_odd.to(b_q_odd.dtype), (b_q_odd))

#             b_dk_even_alpha = tl.dot(b_ds_even_odd.to(b_q_odd.dtype), (b_q_odd))##[BT//2, BK]
#             b_dk_odd_alpha = tl.dot(b_ds_odd_even.to(b_q_even.dtype), (b_q_even))##[BT//2, B=K]
#             b_dk_even += b_dk_even_alpha * b_alpha
#             b_dk_odd += b_dk_odd_alpha * b_alpha
#             b_dalpha += tl.sum(b_dk_even_alpha * b_k_even + b_dk_odd_alpha * b_k_odd,0) ##[BT//2, BK]
#         else:
#             b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)###[BT, BK]
#         if USE_G:
#             b_dg -= tl.sum(b_ds, 1)
#     if USE_ALPHA:
#         tl.store(p_dk_even, b_dk_even.to(p_dk_even.dtype.element_ty), boundary_check=(0, 1))
#         tl.store(p_dk_odd, b_dk_odd.to(p_dk_odd.dtype.element_ty), boundary_check=(0, 1))
#         p_dalpha = dalpha + ((i_b * NT + i_t )* HQ + i_hq )* K + tl.arange(0, BK)
#         tl.store(p_dalpha, b_dalpha.to(p_dalpha.dtype.element_ty), mask=m_alpha)
#     else:
#         tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))   
#     tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
#     if USE_G:
#         p_dg = tl.make_block_ptr(dg_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
#         tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


# def parallel_attn_fwd(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     alpha: torch.Tensor,
#     g_cumsum: torch.Tensor,
#     scale: float,
#     chunk_size: int = 64,
#     cu_seqlens: Optional[torch.LongTensor] = None,
# ):
#     B, T, H, K, V = *k.shape, v.shape[-1]
#     HQ = q.shape[2]
#     G = HQ // H
#     BT = chunk_size
#     if check_shared_mem('hopper', q.device.index):
#         BS = min(64, max(16, triton.next_power_of_2(T)))
#         BK = min(256, max(16, triton.next_power_of_2(K)))
#         BV = min(256, max(16, triton.next_power_of_2(V)))
#     elif check_shared_mem('ampere', q.device.index):
#         BS = min(32, max(16, triton.next_power_of_2(T)))
#         BK = min(256, max(16, triton.next_power_of_2(K)))
#         BV = min(128, max(16, triton.next_power_of_2(V)))
#     else:
#         BS = min(32, max(16, triton.next_power_of_2(T)))
#         BK = min(256, max(16, triton.next_power_of_2(K)))
#         BV = min(64, max(16, triton.next_power_of_2(V)))
#     NK = triton.cdiv(K, BK)
#     NV = triton.cdiv(V, BV)

#     chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
#     NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
#     assert NK == 1, "The key dimension can not be larger than 256"

#     o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
#     lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
#     grid = (NV, NT, B * HQ)
#     parallel_attn_fwd_kernel[grid](
#         q=q,
#         k=k,
#         v=v,
#         o=o,
#         g_cumsum=g_cumsum,
#         lse=lse,
#         scale=scale,
#         alpha=alpha,
#         cu_seqlens=cu_seqlens,
#         chunk_indices=chunk_indices,
#         B=B,
#         T=T,
#         H=H,
#         HQ=HQ,
#         G=G,
#         K=K,
#         V=V,
#         BT=BT,
#         BS=BS,
#         BK=BK,
#         BV=BV,
#     )
#     return o, lse


# def parallel_attn_bwd_preprocess(
#     o: torch.Tensor,
#     do: torch.Tensor
# ):
#     V = o.shape[-1]
#     delta = torch.empty_like(o[..., 0], dtype=torch.float)
#     parallel_attn_bwd_kernel_preprocess[(delta.numel(),)](
#         o=o,
#         do=do,
#         delta=delta,
#         B=triton.next_power_of_2(V),
#         V=V,
#     )
#     return delta


# def parallel_attn_bwd(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     o: torch.Tensor,
#     alpha: torch.Tensor,
#     g_cumsum: torch.Tensor,
#     lse: torch.Tensor,
#     do: torch.Tensor,
#     scale: float = None,
#     chunk_size: int = 64,
#     cu_seqlens: Optional[torch.LongTensor] = None,
# ):
#     B, T, H, K, V = *k.shape, v.shape[-1]
#     HQ = q.shape[2]
#     G = HQ // H
#     BT = chunk_size
#     BS = max(16, triton.next_power_of_2(T))
#     BS = min(32, BS) if check_shared_mem('ampere') else min(16, BS)  # SY:H100 should at least use BS=64 to use WGMMA
#     BK = max(16, triton.next_power_of_2(K))
#     BV = max(16, triton.next_power_of_2(V))

#     chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
#     NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
#     NV = triton.cdiv(V, BV)

#     delta = parallel_attn_bwd_preprocess(o, do)

#     dq = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device)
#     dk = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device)
#     dv = torch.empty(B, T, HQ, V, dtype=v.dtype if H == HQ else torch.float, device=q.device)
#     dalpha = torch.empty(B, NT, HQ, K, dtype=torch.float, device=q.device)
#     grid = (NV, NT, B * HQ)
#     dg_cumsum, dg_cumsum_k = None, None
#     if g_cumsum is not None:
#         dg_cumsum = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
#         dg_cumsum_k = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)

#     parallel_attn_bwd_kernel_dq[grid](
#         q=q,
#         k=k,
#         v=v,
#         alpha=alpha,
#         g_cumsum=g_cumsum,
#         lse=lse,
#         delta=delta,
#         do=do,
#         dq=dq,
#         dg_cumsum=dg_cumsum,
#         cu_seqlens=cu_seqlens,
#         chunk_indices=chunk_indices,
#         scale=scale,
#         T=T,
#         B=B,
#         H=H,
#         HQ=HQ,
#         G=G,
#         K=K,
#         V=V,
#         BT=BT,
#         BS=BS,
#         BK=BK,
#         BV=BV
#     )
#     # print('done_parallel_attn_bwd_kernel_dq')
#     parallel_attn_bwd_kernel_dkv[grid](
#         q=q,
#         k=k,
#         v=v,
#         alpha=alpha,
#         g_cumsum=g_cumsum,
#         lse=lse,
#         delta=delta,
#         dalpha=dalpha,
#         do=do,
#         dk=dk,
#         dv=dv,
#         dg_cumsum=dg_cumsum_k,
#         cu_seqlens=cu_seqlens,
#         chunk_indices=chunk_indices,
#         scale=scale,
#         T=T,
#         NT=NT,
#         B=B,
#         H=H,
#         HQ=HQ,
#         G=G,
#         K=K,
#         V=V,
#         BT=BT,
#         BS=BS,
#         BK=BK,
#         BV=BV
#     )
#     # print('done_parallel_attn_bwd_kernel_dkv')
#     dk = reduce(dk, 'b t (h g) k -> b t h k', g=G, reduction='sum')
#     dv = reduce(dv, 'b t (h g) v -> b t h v', g=G, reduction='sum')
#     dalpha = reduce(dalpha, 'b t (h g) k -> b t h k', g=G, reduction='sum')
#     dalpha = dalpha.reshape(B*NT, H, K).sum(dim=0)
#     if g_cumsum is not None:
#         dg_cumsum.add_(dg_cumsum_k)
#     return dq, dk, dv, dg_cumsum, dalpha


# @torch.compile
# class ParallelAttentionFunction(torch.autograd.Function):

#     @staticmethod
#     @contiguous
#     @autocast_custom_fwd
#     def forward(ctx, q, k, v, g, alpha, scale, cu_seqlens):
#         ctx.dtype = q.dtype
#         chunk_size = min(64, max(16, triton.next_power_of_2(q.shape[1])))###OOM 了 对于128
#         g_cumsum = chunk_global_cumsum(g, cu_seqlens=cu_seqlens) if g is not None else None
#         o, lse = parallel_attn_fwd(
#             q=q,
#             k=k,
#             v=v,
#             g_cumsum=g_cumsum,
#             alpha=alpha,
#             scale=scale,
#             chunk_size=chunk_size,
#             cu_seqlens=cu_seqlens,
#         )
#         # print('done_parallel_attn_fwd')
#         ctx.save_for_backward(q, k, v, o, g_cumsum, lse, alpha)
#         ctx.chunk_size = chunk_size
#         ctx.cu_seqlens = cu_seqlens
#         ctx.scale = scale
#         return o.to(q.dtype)

#     @staticmethod
#     @contiguous
#     @autocast_custom_bwd
#     def backward(ctx, do):
#         q, k, v, o, g_cumsum, lse, alpha = ctx.saved_tensors
#         # print(ctx.chunk_size)
#         dq, dk, dv, dg, dalpha = parallel_attn_bwd(
#             q=q,
#             k=k,
#             v=v,
#             o=o,
#             g_cumsum=g_cumsum,
#             lse=lse,
#             do=do,
#             alpha=alpha,
#             scale=ctx.scale,
#             chunk_size=ctx.chunk_size,
#             cu_seqlens=ctx.cu_seqlens,
#         )
#         if dg is not None:
#             dg = chunk_global_cumsum(dg, cu_seqlens=ctx.cu_seqlens, reverse=True)

#         return dq.to(q), dk.to(k), dv.to(v), dg, dalpha, None, None

# def ceildiv(a, b):
#     return -(a // -b)

# def parallel_attn(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     g: Optional[torch.Tensor] = None,
#     alpha: Optional[torch.Tensor] = None,
#     scale: Optional[float] = None,
#     cu_seqlens: Optional[torch.LongTensor] = None,
#     head_first: bool = False
# ) -> torch.Tensor:
#     r"""
#     Args:
#         q (torch.Tensor):
#             queries of shape `[B, T, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
#         k (torch.Tensor):
#             keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
#             GQA will be applied if HQ is divisible by H.
#         v (torch.Tensor):
#             values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
#         g (Optional[torch.Tensor]):
#             log decay factors of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
#         alpha (Optional[torch.Tensor]):
#             alpha of shape `[H, D]` used for compute alpha^2-beta^2.
#         scale (Optional[int]):
#             Scale factor for attention scores.
#             If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
#         cu_seqlens (torch.LongTensor):
#             Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
#             consistent with the FlashAttention API.
#         head_first (Optional[bool]):
#             Whether the inputs are in the head-first format. Default: `False`.

#     Returns:
#         o (torch.Tensor):
#             Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
#     """
#     if head_first:
#         raise DeprecationWarning(
#             "head_first is deprecated and will be removed in a future version. "
#             "Please use head_first=False for now instead."
#         )
#         q, k, v = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (q, k, v))
#         if g is not None:
#             g = rearrange(g, 'b h t ... -> b t h ...')
#     if not head_first and q.shape[1] < q.shape[2]:
#         warnings.warn(
#             f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
#             "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
#             "when head_first=False was specified. "
#             "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
#         )
#     if scale is None:
#         scale = k.shape[-1] ** -0.5
#     if cu_seqlens is not None:
#         assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"

#     B, T, HQ, K = q.shape
#     B, T, H, V = v.shape
#     padded_seq_len = ceildiv(T, 64) * 64
#     if padded_seq_len > T:
#         padq = torch.zeros(B,padded_seq_len-T,HQ,K,device=q.device,dtype=q.dtype)
#         padk = torch.zeros(B,padded_seq_len-T,H,K,device=k.device,dtype=k.dtype)
#         padv = torch.zeros(B,padded_seq_len-T,H,V,device=v.device,dtype=v.dtype)
#         q = torch.cat([q,padq],dim=1)
#         k = torch.cat([k,padk],dim=1)
#         v = torch.cat([v,padv],dim=1)
#         if g is not None:
#             padg = torch.zeros(B,padded_seq_len-T,H,device=g.device,dtype=g.dtype)
#             g = torch.cat([g,padg],dim=1)
#         # if alpha is not None:
#         #     padalpha = torch.zeros(B,padded_seq_len-T,H,K,device=alpha.device,dtype=alpha.dtype)
#         #     alpha = torch.cat([alpha,padalpha],dim=1)
#     o = ParallelAttentionFunction.apply(q, k, v, g, alpha, scale, cu_seqlens)
#     if head_first:
#         o = rearrange(o, 'b t h ... -> b h t ...')
#     o = o[:, :T, :, :]
#     return o



# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional

import torch
import triton
import triton.language as tl
from einops import rearrange, reduce

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cumsum import chunk_global_cumsum
from ...ops.utils.op import exp, log, safe_exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, contiguous


@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_K_SYM': lambda args: args['k_sym'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1,2,4] + ([8] if check_shared_mem('hopper') else [])
        for num_stages in [2,3,4,5]
    ],
    key=['B', 'H', 'HQ', 'G', 'K', 'V', 'BK', 'BV', 'USE_G', 'IS_VARLEN', 'USE_K_SYM'],
)
@triton.jit
def parallel_attn_fwd_kernel(
    q,
    k,
    v,
    o,
    g_cumsum,
    lse,
    k_sym,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_K_SYM: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
    # the Q block is kept in the shared memory throughout the whole kernel
    if USE_K_SYM:
        p_q_even = tl.make_block_ptr(q + (bos * HQ +      i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
        p_q_odd = tl.make_block_ptr (q + (bos * HQ + HQ + i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
        b_q_even = tl.load(p_q_even, boundary_check=(0, 1))
        b_q_odd = tl.load(p_q_odd, boundary_check=(0, 1))
        b_q_even = (b_q_even * scale).to(b_q_even.dtype)
        b_q_odd = (b_q_odd * scale).to(b_q_odd.dtype)
    else:
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    # [BT, BV]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    if USE_G:
        p_g = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_gq = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    else:
        b_gq = None

    for i_s in range(0, i_t * BT, BS):
        if USE_K_SYM:
            p_k_even = tl.make_block_ptr(k + (bos * H +     i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
            p_k_odd =  tl.make_block_ptr(k + (bos * H + H + i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))####存在两份k即可，根据奇数偶数使用
            b_k_even = tl.load(p_k_even, boundary_check=(0, 1))
            b_k_odd = tl.load(p_k_odd, boundary_check=(0, 1))

            p_k_sym_even = tl.make_block_ptr(k_sym + (bos * H +     i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
            p_k_sym_odd =  tl.make_block_ptr(k_sym + (bos * H + H + i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
            b_k_sym_even = tl.load(p_k_sym_even, boundary_check=(0, 1))
            b_k_sym_odd = tl.load(p_k_sym_odd, boundary_check=(0, 1))
        else:
            p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        if USE_K_SYM:
            b_s_ee = tl.dot(b_q_even, b_k_even)
            b_s_eo = tl.dot(b_q_even, b_k_sym_odd)
            b_s_oe = tl.dot(b_q_odd, b_k_sym_even)
            b_s_oo = tl.dot(b_q_odd, b_k_odd)
            b_s_e = tl.join(b_s_ee,b_s_oe)
            b_s_o = tl.join(b_s_eo,b_s_oo)
            b_s = tl.join(b_s_e,b_s_o) ###BT BS 2 2
            b_s = tl.permute(b_s,0,2,1,3).reshape(BT,BS)
        else:
            b_s = tl.dot(b_q, b_k)

        if USE_G:
            p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
            b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]

        # [BT, BS]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = exp(b_mp - b_m)
        # [BT, BS]
        b_p = safe_exp(b_s - b_m[:, None])
        # [BT]
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        # [BT, BV]
        if USE_K_SYM:
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q_even.dtype), b_v)
        else:
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

        b_mp = b_m

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        if USE_K_SYM:
            p_k_even = tl.make_block_ptr(k + (bos * H +     i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
            p_k_odd =  tl.make_block_ptr(k + (bos * H + H + i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
            b_k_even = tl.load(p_k_even, boundary_check=(0, 1))
            b_k_odd = tl.load(p_k_odd, boundary_check=(0, 1))
            p_k_sym_even = tl.make_block_ptr(k_sym + (bos * H +     i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
            p_k_sym_odd =  tl.make_block_ptr(k_sym + (bos * H + H + i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
            b_k_sym_even = tl.load(p_k_sym_even, boundary_check=(0, 1))
            b_k_sym_odd = tl.load(p_k_sym_odd, boundary_check=(0, 1))
        else:
            p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
            # [BK, BS]
            b_k = tl.load(p_k, boundary_check=(0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

        # [BS]
        o_k = i_s + tl.arange(0, BS)
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        if USE_K_SYM:
            b_s_ee = tl.dot(b_q_even, b_k_even)
            b_s_eo = tl.dot(b_q_even, b_k_sym_odd)
            b_s_oe = tl.dot(b_q_odd, b_k_sym_even)
            b_s_oo = tl.dot(b_q_odd, b_k_odd)

            b_s_e = tl.join(b_s_ee,b_s_oe)
            b_s_o = tl.join(b_s_eo,b_s_oo)
            b_s = tl.join(b_s_e,b_s_o) ###BT BS 2 2
            b_s = tl.permute(b_s,0,2,1,3).reshape(BT,BS)
            
        else:
            b_s = tl.dot(b_q, b_k)
        b_s = tl.where(o_q[:, None] >= o_k[None, :], b_s, float('-inf'))

        if USE_G:
            p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
            b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]

        # [BT]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = exp(b_mp - b_m)
        # [BT, BS]
        b_p = safe_exp(b_s - b_m[:, None])
        # [BT]
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        # [BT, BV]
        if USE_K_SYM:
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q_even.dtype), b_v)
        else:
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
        b_mp = b_m

    b_o = b_o / b_acc[:, None]
    b_m += log(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))


@triton.jit
def parallel_attn_bwd_kernel_preprocess(
    o,
    do,
    delta,
    B: tl.constexpr,
    V: tl.constexpr
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))

#####################num_stages=2 仍然超
####等于2比较极限，或许可以尝试优化一下以使得可用
@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_K_SYM': lambda args: args['k_sym'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1,2,4] + ([8] if check_shared_mem('hopper') else [])
        for num_stages in [2,3,4,5]
    ],
    key=['B', 'H', 'HQ', 'G', 'K', 'V', 'BK', 'BV', 'USE_G', 'IS_VARLEN', 'USE_K_SYM'],
)
@triton.jit(do_not_specialize=['T'])
def parallel_attn_bwd_kernel_dq(
    q,
    k,
    v,
    k_sym,
    lse,
    delta,
    do,
    dq,
    dg_cumsum,
    g_cumsum,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    USE_K_SYM: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
    if USE_K_SYM:###[BT BK]
        p_q_even = tl.make_block_ptr (q + (bos * HQ +      i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
        p_q_odd  = tl.make_block_ptr (q + (bos * HQ + HQ + i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
        b_q_even = tl.load(p_q_even, boundary_check=(0, 1))
        b_q_odd = tl.load(p_q_odd, boundary_check=(0, 1))
        b_q_even = (b_q_even * scale).to(b_q_even.dtype)
        b_q_odd = (b_q_odd * scale).to(b_q_odd.dtype)
        p_dq_even = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K,      (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
        p_dq_odd =  tl.make_block_ptr(dq + (bos * HQ + HQ + i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
    else:
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0)) ####这个思考一下比例
    p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    # [BT, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [BT]
    b_lse = tl.load(p_lse, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))

    # [BT, BK]
    if USE_K_SYM:
        b_dq_even = tl.zeros([BT//2, BK], dtype=tl.float32)
        b_dq_odd = tl.zeros([BT//2, BK], dtype=tl.float32)
    else:
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    if USE_G:
        b_dg = tl.zeros([BT, ], dtype=tl.float32)
        p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
    else:
        b_gq = None
        b_dg = None
    for i_s in range(0, i_t * BT, BS):###分前后段运算，二者有所区别
        if USE_K_SYM:
            p_k_even = tl.make_block_ptr(k + (bos * H +     i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
            p_k_odd =  tl.make_block_ptr(k + (bos * H + H + i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
            b_k_even = tl.load(p_k_even, boundary_check=(0, 1))
            b_k_odd = tl.load(p_k_odd, boundary_check=(0, 1))
            p_k_sym_even = tl.make_block_ptr(k_sym + (bos * H +     i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
            p_k_sym_odd =  tl.make_block_ptr(k_sym + (bos * H + H + i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
            b_k_sym_even = tl.load(p_k_sym_even, boundary_check=(0, 1))
            b_k_sym_odd = tl.load(p_k_sym_odd, boundary_check=(0, 1))
        else:
            p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        if USE_K_SYM:
            #####考虑use join替换或许会快，不知道
            b_s_ee = tl.dot(b_q_even, b_k_even)
            b_s_eo = tl.dot(b_q_even, b_k_sym_odd)
            b_s_oe = tl.dot(b_q_odd, b_k_sym_even)
            b_s_oo = tl.dot(b_q_odd, b_k_odd)

            b_s_e = tl.join(b_s_ee,b_s_oe)
            b_s_o = tl.join(b_s_eo,b_s_oo)
            b_s = tl.join(b_s_e,b_s_o) ###BT BS 2 2
            b_s = tl.permute(b_s,0,2,1,3).reshape(BT,BS)
        else:
            b_s = tl.dot(b_q, b_k)
        if USE_G:
            p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
            b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]

        b_p = safe_exp(b_s - b_lse[:, None])
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])##[BT, BS]
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        if USE_K_SYM:
            b_ds = tl.permute(b_ds.reshape(BT//2,2,BS//2,2),(0,2,1,3))
            b_ds_even, b_ds_odd = b_ds.split()
            b_ds_even_even, b_ds_odd_even = b_ds_even.split()
            b_ds_even_odd, b_ds_odd_odd = b_ds_odd.split()
            b_dq_even += tl.dot(b_ds_even_even.to(b_k_even.dtype), tl.trans(b_k_even))
            b_dq_even += tl.dot(b_ds_even_odd.to(b_k_odd.dtype), tl.trans(b_k_sym_odd))
            b_dq_odd += tl.dot(b_ds_odd_even.to(b_k_even.dtype), tl.trans(b_k_sym_even))
            b_dq_odd += tl.dot(b_ds_odd_odd.to(b_k_odd.dtype), tl.trans(b_k_odd))
        else:
            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
        if USE_G:
            b_dg += tl.sum(b_ds, 1)

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        if USE_K_SYM:
            p_k_even = tl.make_block_ptr(k + (bos * H +     i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
            p_k_odd =  tl.make_block_ptr(k + (bos * H + H + i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
            b_k_even = tl.load(p_k_even, boundary_check=(0, 1))
            b_k_odd = tl.load(p_k_odd, boundary_check=(0, 1))
            p_k_sym_even = tl.make_block_ptr(k_sym + (bos * H +     i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
            p_k_sym_odd =  tl.make_block_ptr(k_sym + (bos * H + H + i_h) * K, (K, T//2), (1, 2*H*K), (0, i_s//2), (BK, BS//2), (1, 0))
            b_k_sym_even = tl.load(p_k_sym_even, boundary_check=(0, 1))
            b_k_sym_odd = tl.load(p_k_sym_odd, boundary_check=(0, 1))
        else:
            p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
        # [BS]
        o_k = i_s + tl.arange(0, BS)
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        if USE_K_SYM:
            b_s_ee = tl.dot(b_q_even, b_k_even)
            b_s_eo = tl.dot(b_q_even, b_k_sym_odd)
            b_s_oe = tl.dot(b_q_odd, b_k_sym_even)
            b_s_oo = tl.dot(b_q_odd, b_k_odd)

            b_s_e = tl.join(b_s_ee,b_s_oe)
            b_s_o = tl.join(b_s_eo,b_s_oo)
            b_s = tl.join(b_s_e,b_s_o) ###BT BS 2 2
            b_s = tl.permute(b_s,0,2,1,3).reshape(BT,BS)
        else:
            b_s = tl.dot(b_q, b_k)

        if USE_G:
            p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
            b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]
            b_s = tl.where(o_q[:, None] >= o_k[None, :], b_s, -float('inf'))

        b_p = safe_exp(b_s - b_lse[:, None])  # SY: important to use safe_exp here to avoid NaN.
        b_p = tl.where(o_q[:, None] >= o_k[None, :], b_p, 0)

        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        # b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
        if USE_K_SYM:
            b_ds = tl.permute(b_ds.reshape(BT//2,2,BS//2,2),(0,2,1,3))
            b_ds_even, b_ds_odd = b_ds.split()
            b_ds_even_even, b_ds_odd_even = b_ds_even.split()
            b_ds_even_odd, b_ds_odd_odd = b_ds_odd.split()############不存在梯度为0的情况
            b_dq_even += tl.dot(b_ds_even_even.to(b_k_even.dtype), tl.trans(b_k_even))
            b_dq_even += tl.dot(b_ds_even_odd.to(b_k_odd.dtype), tl.trans(b_k_sym_odd))
            b_dq_odd += tl.dot(b_ds_odd_even.to(b_k_even.dtype), tl.trans(b_k_sym_even))
            b_dq_odd += tl.dot(b_ds_odd_odd.to(b_k_odd.dtype), tl.trans(b_k_odd))
        else:
            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
        if USE_G:
            b_dg += tl.sum(b_ds, 1)

    if USE_K_SYM:
        b_dq_even *= scale
        b_dq_odd *= scale
        tl.store(p_dq_even, b_dq_even.to(p_dq_even.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dq_odd, b_dq_odd.to(p_dq_odd.dtype.element_ty), boundary_check=(0, 1))
    else:
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_dg = tl.make_block_ptr(dg_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_K_SYM': lambda args: args['k_sym'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1,2,4] + ([8] if check_shared_mem('hopper') else [])
        # for num_stages in [1]
        for num_stages in [2,3,4,5]
    ],
    key=['B', 'H', 'HQ', 'G', 'K', 'V', 'BK', 'BV', 'USE_G', 'IS_VARLEN', 'USE_K_SYM'],
)
@triton.jit(do_not_specialize=['T'])
def parallel_attn_bwd_kernel_dkv(
    q,
    k,
    v,
    k_sym,
    g_cumsum,
    lse,
    delta,
    dk_sym,
    do,
    dk,
    dv,
    dg_cumsum,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    NT,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_K_SYM:tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T

    if USE_K_SYM:
        p_k_even = tl.make_block_ptr(k + (bos * H +     i_h) * K, (T//2, K), (2*H*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
        p_k_odd =  tl.make_block_ptr(k + (bos * H + H + i_h) * K, (T//2, K), (2*H*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
        p_k_sym_even = tl.make_block_ptr(k_sym + (bos * H +     i_h) * K,(T//2, K), (2*H*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
        p_k_sym_odd =  tl.make_block_ptr(k_sym + (bos * H + H + i_h) * K, (T//2, K), (2*H*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
        p_dk_even = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K,      (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
        p_dk_odd =  tl.make_block_ptr(dk + (bos * HQ + HQ + i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
        p_dk_sym_even = tl.make_block_ptr(dk_sym + (bos * HQ + i_hq) * K,      (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
        p_dk_sym_odd =  tl.make_block_ptr(dk_sym + (bos * HQ + HQ + i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_t * BT//2, 0), (BT//2, BK), (1, 0))
        b_k_even = tl.load(p_k_even, boundary_check=(0, 1))
        b_k_odd = tl.load(p_k_odd, boundary_check=(0, 1))
        b_k_sym_even = tl.load(p_k_sym_even, boundary_check=(0, 1))
        b_k_sym_odd = tl.load(p_k_sym_odd, boundary_check=(0, 1))
    else:
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    if USE_K_SYM:
        b_dk_even = tl.zeros([BT//2, BK], dtype=tl.float32)###almost 130KB for stages=1 需要额外保存的信息翻倍了，这里需要思考这个问题
        b_dk_odd = tl.zeros([BT//2, BK], dtype=tl.float32)

        b_dk_sym_even = tl.zeros([BT//2, BK], dtype=tl.float32)
        b_dk_sym_odd = tl.zeros([BT//2, BK], dtype=tl.float32)
    else:
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)####同样130KB OOM了
    o_k = i_t * BT + tl.arange(0, BT)
    if USE_G:
        p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
        b_dg = tl.zeros([BT,], dtype=tl.float32)
    else:
        b_gk = None
        b_dg = None

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        if USE_K_SYM:
            p_q_even = tl.make_block_ptr(q + (bos * HQ + i_hq) * K,    (T//2, K), (2*HQ*K, 1), (i_s//2, 0), (BS//2, BK), (1, 0))
            p_q_odd =  tl.make_block_ptr(q + (bos * HQ + HQ + i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_s//2, 0), (BS//2, BK), (1, 0))
            b_q_even = tl.load(p_q_even, boundary_check=(0, 1))
            b_q_odd = tl.load(p_q_odd, boundary_check=(0, 1))
            b_q_even = (b_q_even * scale).to(b_q_even.dtype)
            b_q_odd = (b_q_odd * scale).to(b_q_odd.dtype)
        else:
            p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        # [BS]
        o_q = i_s + tl.arange(0, BS)
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BS]
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        # [BT, BS]
        if USE_K_SYM:
            b_s_ee = tl.dot(b_k_even,          tl.trans(b_q_even))
            b_s_eo = tl.dot(b_k_sym_even,      tl.trans(b_q_odd))
            b_s_oe = tl.dot(b_k_sym_odd,       tl.trans(b_q_even))
            b_s_oo = tl.dot(b_k_odd,           tl.trans(b_q_odd))
            b_s_e = tl.join(b_s_ee,b_s_oe)
            b_s_o = tl.join(b_s_eo,b_s_oo)
            b_s = tl.join(b_s_e,b_s_o) ###BT BS 2 2
            b_s = tl.permute(b_s,0,2,1,3).reshape(BT,BS)
        else:
            b_s = tl.dot(b_k, tl.trans(b_q))###[BT, BS]

        if USE_G:
            p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
            b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[None, :] - b_gk[:, None]
            b_s = tl.where(o_k[:, None] <= o_q[None, :], b_s, -float('inf'))
        b_p = safe_exp(b_s - b_lse[None, :])
        b_p = tl.where(o_k[:, None] <= o_q[None, :], b_p, 0)
        # [BT, BS] @ [BS, BV] -> [BT, BV]
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_v, tl.trans(b_do))
        # [BT, BS]
        b_ds = b_p * (b_dp - b_delta[None, :])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        if USE_K_SYM:
            b_ds = tl.permute(b_ds.reshape(BT//2,2,BS//2,2),(0,2,1,3))
            b_ds_even, b_ds_odd = b_ds.split()
            b_ds_even_even, b_ds_odd_even = b_ds_even.split()
            b_ds_even_odd, b_ds_odd_odd = b_ds_odd.split()

            b_dk_even += tl.dot(b_ds_even_even.to(b_q_even.dtype), (b_q_even))
            b_dk_odd += tl.dot(b_ds_odd_odd.to(b_q_odd.dtype), (b_q_odd))
            b_dk_sym_even += tl.dot(b_ds_even_odd.to(b_q_odd.dtype), (b_q_odd))
            b_dk_sym_odd += tl.dot(b_ds_odd_even.to(b_q_even.dtype), (b_q_even))
        else:
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        if USE_G:
            b_dg -= tl.sum(b_ds, 1)

    for i_s in range((i_t + 1) * BT, tl.cdiv(T, BS) * BS, BS):
        if USE_K_SYM:
            p_q_even = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_s//2, 0), (BS//2, BK), (1, 0))
            p_q_odd =  tl.make_block_ptr(q + (bos * HQ + HQ + i_hq) * K, (T//2, K), (2*HQ*K, 1), (i_s//2, 0), (BS//2, BK), (1, 0))
            b_q_even = tl.load(p_q_even, boundary_check=(0, 1))
            b_q_odd = tl.load(p_q_odd, boundary_check=(0, 1))
            b_q_even = (b_q_even * scale).to(b_q_even.dtype)
            b_q_odd = (b_q_odd * scale).to(b_q_odd.dtype)
        else:
            p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        # [BS]
        o_q = i_s + tl.arange(0, BS)
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BS]
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        # [BT, BS]
        if USE_K_SYM:
            b_s_ee = tl.dot(b_k_even,         tl.trans(b_q_even))
            b_s_eo = tl.dot(b_k_sym_even, tl.trans(b_q_odd))
            b_s_oe = tl.dot(b_k_sym_odd,  tl.trans(b_q_even))
            b_s_oo = tl.dot(b_k_odd,          tl.trans(b_q_odd))

            b_s_e = tl.join(b_s_ee,b_s_oe)
            b_s_o = tl.join(b_s_eo,b_s_oo)
            b_s = tl.join(b_s_e,b_s_o) ###BT BS 2 2
            b_s = tl.permute(b_s,0,2,1,3).reshape(BT,BS)
        else:
            b_s = tl.dot(b_k, tl.trans(b_q))###[BT, BS]

        if USE_G:
            p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
            b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[None, :] - b_gk[:, None]
        b_p = safe_exp(b_s - b_lse[None, :])###BT BT
        # [BT, BS] @ [BS, BV] -> [BT, BV]
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_v, tl.trans(b_do))
        # [BT, BS]
        b_ds = b_p * (b_dp - b_delta[None, :])
        # [BT, BS] @ [BS, BK] -> [BT, BK]

        if USE_K_SYM:
            b_ds = tl.permute(b_ds.reshape(BT//2,2,BS//2,2),(0,2,1,3))
            b_ds_even, b_ds_odd = b_ds.split()
            b_ds_even_even, b_ds_odd_even = b_ds_even.split()
            b_ds_even_odd, b_ds_odd_odd = b_ds_odd.split()
            b_dk_even += tl.dot(b_ds_even_even.to(b_q_even.dtype), (b_q_even))
            b_dk_odd += tl.dot(b_ds_odd_odd.to(b_q_odd.dtype), (b_q_odd))
            b_dk_sym_even += tl.dot(b_ds_even_odd.to(b_q_odd.dtype), (b_q_odd))
            b_dk_sym_odd += tl.dot(b_ds_odd_even.to(b_q_even.dtype), (b_q_even))
            # b_dk_even_alpha = tl.dot(b_ds_even_odd.to(b_q_odd.dtype), (b_q_odd))##[BT//2, BK]
            # b_dk_odd_alpha = tl.dot(b_ds_odd_even.to(b_q_even.dtype), (b_q_even))##[BT//2, B=K]
            # b_dk_even += b_dk_even_alpha * b_alpha
            # b_dk_odd += b_dk_odd_alpha * b_alpha
            # b_dalpha += tl.sum(b_dk_even_alpha * b_k_even + b_dk_odd_alpha * b_k_odd,0) ##[BT//2, BK]
        else:
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)###[BT, BK]
        if USE_G:
            b_dg -= tl.sum(b_ds, 1)
    if USE_K_SYM:
        tl.store(p_dk_even, b_dk_even.to(p_dk_even.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk_odd, b_dk_odd.to(p_dk_odd.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk_sym_even, b_dk_sym_even.to(p_dk_sym_even.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk_sym_odd, b_dk_sym_odd.to(p_dk_sym_odd.dtype.element_ty), boundary_check=(0, 1))
    else:
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))   
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_dg = tl.make_block_ptr(dg_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


def parallel_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_sym: torch.Tensor,
    g_cumsum: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BT = chunk_size
    if check_shared_mem('hopper', q.device.index):
        BS = min(64, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(256, max(16, triton.next_power_of_2(V)))
    elif check_shared_mem('ampere', q.device.index):
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(128, max(16, triton.next_power_of_2(V)))
    else:
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(64, max(16, triton.next_power_of_2(V)))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert NK == 1, "The key dimension can not be larger than 256"

    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    grid = (NV, NT, B * HQ)
    parallel_attn_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        g_cumsum=g_cumsum,
        lse=lse,
        scale=scale,
        k_sym=k_sym,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    return o, lse


def parallel_attn_bwd_preprocess(
    o: torch.Tensor,
    do: torch.Tensor
):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float)
    parallel_attn_bwd_kernel_preprocess[(delta.numel(),)](
        o=o,
        do=do,
        delta=delta,
        B=triton.next_power_of_2(V),
        V=V,
    )
    return delta


def parallel_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    k_sym: torch.Tensor,
    g_cumsum: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    scale: float = None,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BT = chunk_size
    BS = max(16, triton.next_power_of_2(T))
    BS = min(32, BS) if check_shared_mem('ampere') else min(16, BS)  # SY:H100 should at least use BS=64 to use WGMMA
    BK = max(16, triton.next_power_of_2(K))
    BV = max(16, triton.next_power_of_2(V))

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NV = triton.cdiv(V, BV)

    delta = parallel_attn_bwd_preprocess(o, do)

    dq = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device)
    dk = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device)
    dv = torch.empty(B, T, HQ, V, dtype=v.dtype if H == HQ else torch.float, device=q.device)
    dk_sym = torch.empty(B, T, HQ, K, dtype=torch.float, device=q.device)
    grid = (NV, NT, B * HQ)
    dg_cumsum, dg_cumsum_k = None, None
    if g_cumsum is not None:
        dg_cumsum = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
        dg_cumsum_k = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)

    parallel_attn_bwd_kernel_dq[grid](
        q=q,
        k=k,
        v=v,
        k_sym=k_sym,
        g_cumsum=g_cumsum,
        lse=lse,
        delta=delta,
        do=do,
        dq=dq,
        dg_cumsum=dg_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV
    )
    # print('done_parallel_attn_bwd_kernel_dq')
    parallel_attn_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        k_sym=k_sym,
        g_cumsum=g_cumsum,
        lse=lse,
        delta=delta,
        dk_sym=dk_sym,
        do=do,
        dk=dk,
        dv=dv,
        dg_cumsum=dg_cumsum_k,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        NT=NT,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV
    )
    # print('done_parallel_attn_bwd_kernel_dkv')
    dk = reduce(dk, 'b t (h g) k -> b t h k', g=G, reduction='sum')
    dv = reduce(dv, 'b t (h g) v -> b t h v', g=G, reduction='sum')
    dk_sym = reduce(dk_sym, 'b t (h g) k -> b t h k', g=G, reduction='sum')
    if g_cumsum is not None:
        dg_cumsum.add_(dg_cumsum_k)
    return dq, dk, dv, dg_cumsum, dk_sym


@torch.compile
class ParallelAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, k_sym, scale, cu_seqlens):
        ctx.dtype = q.dtype
        chunk_size = min(64, max(16, triton.next_power_of_2(q.shape[1])))###OOM 了 对于128
        g_cumsum = chunk_global_cumsum(g, cu_seqlens=cu_seqlens) if g is not None else None
        o, lse = parallel_attn_fwd(
            q=q,
            k=k,
            v=v,
            g_cumsum=g_cumsum,
            k_sym=k_sym,
            scale=scale,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
        )
        # print('done_parallel_attn_fwd')
        ctx.save_for_backward(q, k, v, o, g_cumsum, lse, k_sym)
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens = cu_seqlens
        ctx.scale = scale
        return o.to(q.dtype)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, o, g_cumsum, lse, k_sym = ctx.saved_tensors
        # print(ctx.chunk_size)
        dq, dk, dv, dg, dk_sym = parallel_attn_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            g_cumsum=g_cumsum,
            lse=lse,
            do=do,
            k_sym=k_sym,
            scale=ctx.scale,
            chunk_size=ctx.chunk_size,
            cu_seqlens=ctx.cu_seqlens,
        )
        if dg is not None:
            dg = chunk_global_cumsum(dg, cu_seqlens=ctx.cu_seqlens, reverse=True)

        return dq.to(q), dk.to(k), dv.to(v), dg, dk_sym.to(k_sym), None, None

def ceildiv(a, b):
    return -(a // -b)

def parallel_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    k_sym: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            GQA will be applied if HQ is divisible by H.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g (Optional[torch.Tensor]):
            log decay factors of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        k_sym (Optional[torch.Tensor]):
            k_sym of shape `[B, T, H, K]` used for compute k_sym+ and k_sym-.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (q, k, v))
        if g is not None:
            g = rearrange(g, 'b h t ... -> b t h ...')
        if k_sym is not None:
            k_sym = rearrange(k_sym, 'b h t ... -> b t h ...')
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"

    B, T, HQ, K = q.shape
    B, T, H, V = v.shape
    padded_seq_len = ceildiv(T, 64) * 64
    if padded_seq_len > T:
        padq = torch.zeros(B,padded_seq_len-T,HQ,K,device=q.device,dtype=q.dtype)
        padk = torch.zeros(B,padded_seq_len-T,H,K,device=k.device,dtype=k.dtype)
        padv = torch.zeros(B,padded_seq_len-T,H,V,device=v.device,dtype=v.dtype)
        q = torch.cat([q,padq],dim=1)
        k = torch.cat([k,padk],dim=1)
        v = torch.cat([v,padv],dim=1)
        if g is not None:
            padg = torch.zeros(B,padded_seq_len-T,H,device=g.device,dtype=g.dtype)
            g = torch.cat([g,padg],dim=1)
        if k_sym is not None:
            padk_sym = torch.zeros(B,padded_seq_len-T,H,K,device=k_sym.device,dtype=k_sym.dtype)
            k_sym = torch.cat([k_sym,padk_sym],dim=1)
    o = ParallelAttentionFunction.apply(q, k, v, g, k_sym, scale, cu_seqlens)
    if head_first:
        o = rearrange(o, 'b t h ... -> b h t ...')
    o = o[:, :T, :, :]
    return o

