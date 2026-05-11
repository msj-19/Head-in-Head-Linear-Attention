import time
import torch
import triton
import triton.language as tl
from einops import rearrange
from fla.utils import autocast_custom_bwd, autocast_custom_fwd,contiguous
from fla.ops.utils import chunk_local_cumsum
import torch.nn.functional as F
from typing import Optional
import sys
# sys.path.append('/mnt/jfzn/msj/flash-linear-attention/evals/lm_eval/models')
# from .mask_gated_delta_rule_t.recurrent_fuse import mask_fused_recurrent_gated_delta_rule

def solve_trilss(A,mask,k,BT,output_dtype=torch.float32):
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
        p_mask = tl.make_block_ptr(mask_ij + i_bh * T*r*r,(T,r,r),(r*r,r,1),(i_t*BT,0,i_r),(BT,r,1),(2,1,0))
        b_mask = tl.load(p_mask)#BT r 1
        for i_k in range(tl.cdiv(dk, BK)):
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*dk + i_k * BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)[:,None,:]*b_mask.to(b_k.dtype)#BT*r*d
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
        p_mask = tl.make_block_ptr(mask_ij + i_bh * T*r*r,(T,r,r),(r*r,r,1),(i_t*BT,0,i_r),(BT,r,1),(2,1,0))
        b_mask = tl.load(p_mask)#BT r 1
        ij_mask = b_mask*r_mask[None,None,:]#行数 #BT [r,r]
        for i_k in range(tl.cdiv(dk, BK)):#分块k读取计算
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r * dk + i_k * BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
            dot = tl.dot(b_kb, tl.trans(b_k), allow_tf32=False)#BT BT 
            b_A += dot[:,:,None,None]*ij_mask[:,None,:,:]#BT r r

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
def solve_tril_16x16_kernel_org(
    A,
    Ad,
    s_A_bh,
    s_Ad_bh,
    T,
    r:  tl.constexpr,
    BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    offset = (i_t * 16) % (BT * r)

    p_A = tl.make_block_ptr(A + (i_bh)*s_A_bh, (T*r,BT*r),(BT*r,1) ,(i_t * 16, offset), (16, 16), (1,0))
    b_A = tl.load(p_A, boundary_check=(0,1)).to(tl.float32)
    b_A = -tl.where((tl.arange(0, 16)[:,None] > tl.arange(0, 16)[None,:]), b_A, 0)####0.0008
    p_Ad = tl.make_block_ptr(Ad + (i_bh)*T*r*16, (T*r,16),(16,1) ,(i_t * 16, 0), (16, 16), (1,0))
    o_i = tl.arange(0, 16)
    for i in range(r,16):
        b_a = -tl.load(A + (i_bh)*s_A_bh + (i_t * 16 + i) * BT * r + offset + o_i)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        mask = o_i == i
        b_A = tl.where(mask[:, None], b_a, b_A)
    b_A += o_i[:, None] == o_i[None, :]
    tl.store(p_Ad, b_A.to(p_Ad.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


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
def merge_r1_to_r2_inverse_kernel(
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
    ###T//16
    offset = ((i_t*16) % BT) *r

    p_A21 = tl.make_block_ptr(A + (i_bh)*s_A_bh, (T*r,BT*r),(BT*r,1) ,(i_t * 16 * r + 16, offset), (16, 16), (1,0))
    b_A21 = tl.load(p_A21, boundary_check=(0,1)).to(tl.float32)

    p_Ad11  = tl.make_block_ptr(Ad + (i_bh)*s_Ad_bh,(T*r,16),(16,1), (i_t * 16 * r, 0), (16,16), (1,0))
    p_Ad22  = tl.make_block_ptr(Ad + (i_bh)*s_Ad_bh,(T*r,16),(16,1), (i_t * 16 * r +16 , 0), (16,16), (1,0))

    p_Ai11 = tl.make_block_ptr(Ai+ (i_bh)*s_Ad_bh*r, (T*r,16*r), (16*r, 1), (i_t * 16 * r,     0), (16, 16), (1, 0))
    p_Ai22 = tl.make_block_ptr(Ai+ (i_bh)*s_Ad_bh*r, (T*r,16*r), (16*r, 1), (i_t * 16 * r +16,16), (16, 16), (1, 0))
    p_Ai21 = tl.make_block_ptr(Ai+ (i_bh)*s_Ad_bh*r, (T*r,16*r), (16*r, 1), (i_t * 16 * r +16, 0), (16, 16), (1, 0))

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
def merge_r1_to_r4_inverse_kernel(###我想复用算子
        A,
        Ad,
        Ai,
        s_A_bh,
        s_Ad_bh,
        s_Ai_bh,
        T,
        r: tl.constexpr,
        BT: tl.constexpr 
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    offset = ((i_t*16*r) % (BT*r)) 

    p_A21 = tl.make_block_ptr(A + (i_bh)*s_A_bh, (T*r,BT*r),(BT*r,1) ,(i_t * 16 *r +16, offset),    (16, 16), (1,0))
    p_A31 = tl.make_block_ptr(A + (i_bh)*s_A_bh, (T*r,BT*r),(BT*r,1) ,(i_t * 16 *r +32, offset),    (16, 16), (1,0))
    p_A32 = tl.make_block_ptr(A + (i_bh)*s_A_bh, (T*r,BT*r),(BT*r,1) ,(i_t * 16 *r +32, offset+16), (16, 16), (1,0))
    p_A41 = tl.make_block_ptr(A + (i_bh)*s_A_bh, (T*r,BT*r),(BT*r,1) ,(i_t * 16 *r +48, offset),    (16, 16), (1,0))
    p_A42 = tl.make_block_ptr(A + (i_bh)*s_A_bh, (T*r,BT*r),(BT*r,1) ,(i_t * 16 *r +48, offset+16), (16, 16), (1,0))
    p_A43 = tl.make_block_ptr(A + (i_bh)*s_A_bh, (T*r,BT*r),(BT*r,1) ,(i_t * 16 *r +48, offset+32), (16, 16), (1,0))
    
    b_A21 = tl.load(p_A21, boundary_check=(0,1)).to(tl.float32)
    b_A31 = tl.load(p_A31, boundary_check=(0,1)).to(tl.float32)
    b_A32 = tl.load(p_A32, boundary_check=(0,1)).to(tl.float32)
    b_A41 = tl.load(p_A41, boundary_check=(0,1)).to(tl.float32)
    b_A42 = tl.load(p_A42, boundary_check=(0,1)).to(tl.float32)
    b_A43 = tl.load(p_A43, boundary_check=(0,1)).to(tl.float32)


    p_Ad11  = tl.make_block_ptr(Ad + (i_bh)*s_Ad_bh,(T*r,16),(16,1), (i_t * 16 *r    , 0), (16,16), (1,0))
    p_Ad22  = tl.make_block_ptr(Ad + (i_bh)*s_Ad_bh,(T*r,16),(16,1), (i_t * 16 *r +16, 0), (16,16), (1,0))
    p_Ad33  = tl.make_block_ptr(Ad + (i_bh)*s_Ad_bh,(T*r,16),(16,1), (i_t * 16 *r +32, 0), (16,16), (1,0))
    p_Ad44  = tl.make_block_ptr(Ad + (i_bh)*s_Ad_bh,(T*r,16),(16,1), (i_t * 16 *r +48, 0), (16,16), (1,0))
    ###这里是对的


    p_Ai11 = tl.make_block_ptr(Ai+ (i_bh)*s_Ai_bh, (T*r,16*r), (16*r, 1), (i_t * 16 *r, 0),     (16, 16), (1, 0))
    p_Ai22 = tl.make_block_ptr(Ai+ (i_bh)*s_Ai_bh, (T*r,16*r), (16*r, 1), (i_t * 16 *r+16, 16), (16, 16), (1, 0))
    p_Ai33 = tl.make_block_ptr(Ai+ (i_bh)*s_Ai_bh, (T*r,16*r), (16*r, 1), (i_t * 16 *r+32, 32), (16, 16), (1, 0))
    p_Ai44 = tl.make_block_ptr(Ai+ (i_bh)*s_Ai_bh, (T*r,16*r), (16*r, 1), (i_t * 16 *r+48, 48), (16, 16), (1, 0))
    
    p_Ai21 = tl.make_block_ptr(Ai+ (i_bh)*s_Ai_bh, (T*r,16*r), (16*r, 1), (i_t * 16 *r+16, 0),  (16, 16), (1, 0))
    p_Ai31 = tl.make_block_ptr(Ai+ (i_bh)*s_Ai_bh, (T*r,16*r), (16*r, 1), (i_t * 16 *r+32, 0),  (16, 16), (1, 0))
    p_Ai32 = tl.make_block_ptr(Ai+ (i_bh)*s_Ai_bh, (T*r,16*r), (16*r, 1), (i_t * 16 *r+32, 16), (16, 16), (1, 0))
    p_Ai41 = tl.make_block_ptr(Ai+ (i_bh)*s_Ai_bh, (T*r,16*r), (16*r, 1), (i_t * 16 *r+48 ,0),  (16, 16), (1, 0))
    p_Ai42 = tl.make_block_ptr(Ai+ (i_bh)*s_Ai_bh, (T*r,16*r), (16*r, 1), (i_t * 16 *r+48, 16), (16, 16), (1, 0))
    p_Ai43 = tl.make_block_ptr(Ai+ (i_bh)*s_Ai_bh, (T*r,16*r), (16*r, 1), (i_t * 16 *r+48, 32), (16, 16), (1, 0))


    Ai11 = tl.load(p_Ad11, boundary_check=(0, 1)).to(tl.float32)
    Ai22 = tl.load(p_Ad22, boundary_check=(0, 1)).to(tl.float32)
    Ai33 = tl.load(p_Ad33, boundary_check=(0, 1)).to(tl.float32)
    Ai44 = tl.load(p_Ad44, boundary_check=(0, 1)).to(tl.float32)####这里计算应该是对的


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
def merge_r4_to_r8_inverse_kernel(
        A,###B H T 8 BT 8
        Ad,###B H T 8 16 8
        Ai,###B H T 8 16 4
        s_A_bh,
        s_Ad_bh,
        s_Ai_bh,
        T,
        r: tl.constexpr,
        BT: tl.constexpr 
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    ###T//16
    offset = ((i_t*16*8)) % (BT * 8)

    p_A21 = tl.make_block_ptr(A + (i_bh)*s_A_bh,    (T*r,BT*r),(BT*r,1) ,    (i_t * 16 * 8 + 64, offset), (64, 64), (1,0))
    b_A21 = tl.load(p_A21, boundary_check=(0,1)).to(tl.float32)

    p_Ad11  = tl.make_block_ptr(Ad + (i_bh)*s_Ad_bh,(T*r,16*4),(16*4,1),  (i_t * 16 * 8,  0),      (64,64), (1,0))
    p_Ad22  = tl.make_block_ptr(Ad + (i_bh)*s_Ad_bh,(T*r,16*4),(16*4,1),  (i_t * 16 * 8 + 64 , 0), (64,64), (1,0))

    p_Ai11 = tl.make_block_ptr(Ai+ (i_bh)*s_Ai_bh, (T*r,16*8), (16*8, 1), (i_t * 16 * 8,     0),   (64, 64), (1, 0))
    p_Ai22 = tl.make_block_ptr(Ai+ (i_bh)*s_Ai_bh, (T*r,16*8), (16*8, 1), (i_t * 16 * 8 +64,64),   (64, 64), (1, 0))
    p_Ai21 = tl.make_block_ptr(Ai+ (i_bh)*s_Ai_bh, (T*r,16*8), (16*8, 1), (i_t * 16 * 8 +64, 0),   (64, 64), (1, 0))

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



# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=1),
#         triton.Config({}, num_warps=2),
#         triton.Config({}, num_warps=4),
#         triton.Config({}, num_warps=8),
#         triton.Config({}, num_warps=16)
#     ],
#     key=["r"],
# )
@triton.jit
def merge_16x16_to_32x32_inverse_kernelr8(
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

    p_Ad11  = tl.make_block_ptr(Ad + (i_bh)*s_Ad_bh,(T*r,16*r),(16*r,1), (i_t * 32 * r, 0), (16*r,16*r), (1,0))
    p_Ad22  = tl.make_block_ptr(Ad + (i_bh)*s_Ad_bh,(T*r,16*r),(16*r,1), ((i_t *32 +16) * r, 0), (16*r,16*r), (1,0))
    p_Ai11 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,32*r), (32*r, 1), (i_t * 32 * r , 0), (16*r, 16*r), (1, 0))
    p_Ai22 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,32*r), (32*r, 1), ((i_t * 32 + 16) * r , 16*r), (16*r, 16*r), (1, 0))
    Ai11 = tl.load(p_Ad11, boundary_check=(0, 1)).to(tl.float32)
    Ai22 = tl.load(p_Ad22, boundary_check=(0, 1)).to(tl.float32)
    tl.store(p_Ai11,Ai11.to(p_Ai11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai22,Ai22.to(p_Ai22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


    p_Ai21_1 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,32*r), (32*r, 1), ((i_t * 32 + 16) * r, 0), (8*r, 8*r), (1, 0))
    p_Ai21_2 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,32*r), (32*r, 1), ((i_t * 32 + 16) * r, 8*r), (8*r, 8*r), (1, 0))
    p_Ai21_3 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,32*r), (32*r, 1), ((i_t * 32 + 16) * r +8*r, 0), (8*r, 8*r), (1, 0))
    p_Ai21_4 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,32*r), (32*r, 1), ((i_t * 32 + 16) * r +8*r, 8*r), (8*r, 8*r), (1, 0))

    p_Ai22_1 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,32*r), (32*r, 1), ((i_t * 32 + 16) * r,      16*r), (8*r, 8*r), (1, 0))
    p_Ai22_2 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,32*r), (32*r, 1), ((i_t * 32 + 16) * r,      16*r + 8*r), (8*r, 8*r), (1, 0))
    p_Ai22_3 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,32*r), (32*r, 1), ((i_t * 32 + 16) * r +8*r, 16*r), (8*r, 8*r), (1, 0))
    p_Ai22_4 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,32*r), (32*r, 1), ((i_t * 32 + 16) * r +8*r, 16*r + 8*r), (8*r, 8*r), (1, 0))

    p_bA21_1 = tl.make_block_ptr(A +(i_bh)*s_A_bh,  (T*r,32*r), (32*r, 1) ,((i_t * 32 + 16) * r, 0), (8*r, 8*r), (1, 0))
    p_bA21_2 = tl.make_block_ptr(A+ (i_bh)*s_A_bh,  (T*r,32*r), (32*r, 1), ((i_t * 32 + 16) * r, 8*r), (8*r, 8*r), (1, 0))
    p_bA21_3 = tl.make_block_ptr(A+ (i_bh)*s_A_bh,  (T*r,32*r), (32*r, 1), ((i_t * 32 + 16) * r +8*r, 0), (8*r, 8*r), (1, 0))
    p_bA21_4 = tl.make_block_ptr(A+ (i_bh)*s_A_bh,  (T*r,32*r), (32*r, 1), ((i_t * 32 + 16) * r +8*r, 8*r), (8*r, 8*r), (1, 0))

    p_Ai11_1 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,32*r), (32*r, 1), (i_t * 32 * r ,     0), (8*r, 8*r), (1, 0))
    p_Ai11_2 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,32*r), (32*r, 1), (i_t * 32 * r ,   8*r), (8*r, 8*r), (1, 0))
    p_Ai11_3 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,32*r), (32*r, 1), (i_t * 32 * r+8*r , 0), (8*r, 8*r), (1, 0))
    p_Ai11_4 = tl.make_block_ptr(Ai+ (i_bh)*s_A_bh, (T*r,32*r), (32*r, 1), (i_t * 32 * r+8*r,8*r), (8*r, 8*r), (1, 0))
    

    a1 = tl.load(p_Ai22_1,boundary_check=(0, 1)).to(tl.float32)
    a2 = tl.load(p_Ai22_2,boundary_check=(0, 1)).to(tl.float32)
    b1 = tl.load(p_bA21_1,boundary_check=(0, 1)).to(tl.float32)
    b2 = tl.load(p_bA21_3,boundary_check=(0, 1)).to(tl.float32)
    ans1= tl.dot(a1,b1, input_precision='ieee')+tl.dot(a2,b2, input_precision='ieee')

    a1 = tl.load(p_Ai22_1,boundary_check=(0, 1)).to(tl.float32)
    a2 = tl.load(p_Ai22_2,boundary_check=(0, 1)).to(tl.float32)
    b1 = tl.load(p_bA21_2,boundary_check=(0, 1)).to(tl.float32)
    b2 = tl.load(p_bA21_4,boundary_check=(0, 1)).to(tl.float32)
    ans2 = tl.dot(a1,b1, input_precision='ieee')+tl.dot(a2,b2, input_precision='ieee')

    a1 = tl.load(p_Ai22_3,boundary_check=(0, 1)).to(tl.float32)
    a2 = tl.load(p_Ai22_4,boundary_check=(0, 1)).to(tl.float32)
    b1 = tl.load(p_bA21_1,boundary_check=(0, 1)).to(tl.float32)
    b2 = tl.load(p_bA21_3,boundary_check=(0, 1)).to(tl.float32)
    ans3 = tl.dot(a1,b1, input_precision='ieee')+tl.dot(a2,b2, input_precision='ieee')

    a1 = tl.load(p_Ai22_3,boundary_check=(0, 1)).to(tl.float32)
    a2 = tl.load(p_Ai22_4,boundary_check=(0, 1)).to(tl.float32)
    b1 = tl.load(p_bA21_2,boundary_check=(0, 1)).to(tl.float32)
    b2 = tl.load(p_bA21_4,boundary_check=(0, 1)).to(tl.float32)
    ans4 = tl.dot(a1,b1, input_precision='ieee')+tl.dot(a2,b2, input_precision='ieee')

    b1 = tl.load(p_Ai11_1,boundary_check=(0, 1)).to(tl.float32)
    b2 = tl.load(p_Ai11_3,boundary_check=(0, 1)).to(tl.float32)
    ans = -tl.dot(ans1,b1, input_precision='ieee')-tl.dot(ans2,b2, input_precision='ieee')
    tl.store(p_Ai21_1,ans.to(p_Ai21_1.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))

    b1 = tl.load(p_Ai11_2,boundary_check=(0, 1)).to(tl.float32)
    b2 = tl.load(p_Ai11_4,boundary_check=(0, 1)).to(tl.float32)
    ans = -tl.dot(ans1,b1, input_precision='ieee')-tl.dot(ans2,b2, input_precision='ieee')
    tl.store(p_Ai21_2,ans.to(p_Ai21_2.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


    ans = -tl.dot(ans3,b1, input_precision='ieee')-tl.dot(ans4,b2, input_precision='ieee')
    tl.store(p_Ai21_4,ans.to(p_Ai21_4.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


    b1 = tl.load(p_Ai11_1,boundary_check=(0, 1)).to(tl.float32)
    b2 = tl.load(p_Ai11_3,boundary_check=(0, 1)).to(tl.float32)
    ans = -tl.dot(ans1,b1, input_precision='ieee')-tl.dot(ans2,b2, input_precision='ieee')
    tl.store(p_Ai21_3,ans.to(p_Ai21_3.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))



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
    B, H, T, K = k.shape
    r = mask.shape[-1] #B H T r r
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

    A = rearrange(A,'b (t l) (c r)->b (t c) (l r)',t=BT,c=r).contiguous()#BT*r BT*r  ###########if r==8 如何解决
    ########我们已经获得了

    N_64 = triton.cdiv(r*T, 16)
    Ad_1 = torch.empty(B,H,N_64*16,16,device=A.device, dtype=torch.float)
    solve_tril_16x16_kernel_org[(N_64, B*H)](#use merge 计算solve_tril
            A,Ad_1,
            T*BT*r*r,#s_abh
            T*16*r*r,#s_adbh
            T,
            r, BT
    )
    if r==1:
        Ad = torch.zeros(B,H,NT*r*16,16*r,device=A.device, dtype=torch.float if BT != 16 else output_dtype)###根据r考虑如何merge
        Ad = Ad_1       
    if r==2:
        Ad = torch.zeros(B,H,NT*r*16,16*r,device=A.device, dtype=torch.float if BT != 16 else output_dtype)###根据r考虑如何merge
        merge_r1_to_r2_inverse_kernel[(NT, B*H)](
            A,Ad_1,Ad,
            T*BT*r*r,#s_a_bh and s_ai_bh
            T*16*r,#s_ad_bh
            T,r,BT
        )
    if r==4:
        Ad = torch.zeros(B,H,NT*r*16,16*r,device=A.device, dtype=torch.float if BT != 16 else output_dtype)###根据r考虑如何merge
        merge_r1_to_r4_inverse_kernel[(NT, B*H)](
            A,Ad_1,Ad,
            T*BT*r*r,#s_a_bh and s_ai_bh
            T*16*r,#s_ad_bh
            T*16*r*r,
            T,r,BT
        )
    if r==8:
        Ad = torch.zeros(B,H,NT*r*16,16*4,device=A.device, dtype=torch.float if BT != 16 else output_dtype)###根据r考虑如何merge
        merge_r1_to_r4_inverse_kernel[(2*NT, B*H)](
            A,Ad_1,Ad,
            T*BT*8*8,#s_a_bh and s_ai_bh
            T*16*8,#s_ad_bh
            2*T*16*4*4,
            2*T,4,2*BT####等价将长宽看成原来的两倍r不变
        )
        Ad1 = torch.zeros(B,H,NT*r*16,16*r,device=A.device, dtype=torch.float if BT != 16 else output_dtype)###根据r考虑如何merge
        merge_r4_to_r8_inverse_kernel[(NT,B*H)](
            A,Ad,Ad1,
            T*BT*8*8,#s_a_bh and s_ai_bh
            T*16*8*4,#s_ad_bh
            T*16*8*8,
            T,8,BT,
        )
    if BT == 16:    
        if r==8: 
            return Ad1                                                                                                                   
        return Ad
    if BT == 32:
        if r==8:####
            NT = triton.cdiv(T, BT)
            Ai = torch.zeros(B,H,NT*BT*r,BT*r,device=A.device, dtype=output_dtype)
            merge_16x16_to_32x32_inverse_kernelr8[(NT, B*H)](###这个计算是对的
                A,Ad1,Ai,
                T*BT*r*r,#s_a_bh and s_ai_bhnvi
                T*16*r*r,#s_ad_bh
                T,r,BT
            )
            return Ai
        NT = triton.cdiv(T, BT)
        Ai = torch.zeros(B,H,NT*BT*r,BT*r,device=A.device, dtype=output_dtype)
        merge_16x16_to_32x32_inverse_kernel[(NT, B*H)](
            A,Ad,Ai,
            T*BT*r*r,#s_a_bh and s_ai_bhnvi
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


#finish
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16)
    ],
    key=["BT", "BK", "BV","r"],
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
    # B,H,NV,NT
    for i_t in range(NT):
        p_h = tl.make_block_ptr(h + i_bh * NT * K * V + i_t * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
        b_h_cumsum = tl.zeros([r, BK//r, BV], dtype=tl.float32)
        for i_r in range(r):
            r_mask = tl.arange(0,r) == i_r
            p_k = tl.make_block_ptr(k + i_bh * K * T, (K, T), (1, K),
                                    (i_k * BK + i_r * BK//r, i_t * BT), (BK//r,BT), (0, 1))#读取对应
            p_v_new = tl.make_block_ptr(v_new + (i_bh * r + i_r)* T *  V, (T , V), (V, 1),
                                        (i_t * BT , i_v * BV), (BT , BV), (1, 0))        
            p_d = tl.make_block_ptr((d + i_bh * T * r * K),(T, r * K ),(r * K, 1),
                                    (i_t * BT, i_r * K + i_k * BK), (BT,BK),(1,0))
            p_v = tl.make_block_ptr((v + i_bh * T * r * V),(T, r * V ),(r * V, 1),
                                    (i_t * BT, i_r * K + i_v * BV), (BT,BV),(1,0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_d = tl.load(p_d, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_v -= tl.dot(b_d, b_h.to(b_d.dtype)).to(b_v.dtype)
            tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))#至少到这里第一步结果相同
            kv = tl.dot((b_k),b_v)####小数乘以大数的精度问题
            b_h_cumsum = tl.where(r_mask[:,None,None],b_h_cumsum + kv[None,:,:] ,b_h_cumsum)
        
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + i_bh*T + last_idx)
        b_g_last = tl.exp(b_g_last)
        b_h = b_g_last * b_h
        
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
        b_o += tl.dot(b_q, b_h.to(b_q.dtype))
        b_s += tl.dot(b_q, b_k)

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
    
    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1
    h = torch.empty(B, H, NT * K, V,device=k.device,dtype=k.dtype)
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
        v_new,
        g,h,
        initial_state,
        final_state,
        H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,r=r,      
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
    )###确认一下final_state是否相同
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
    dv2 = torch.empty_like(dv)#一样的 #bhr T V ####dv2反转了顺序？
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
    b_dg -= tl.sum(tl.reshape(b_w*b_dw,(BT,r*BK)),-1) ######多一个这个
    ############

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

# compute this 
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
    p_A = tl.make_block_ptr(Aw + i_bh*T*BT*r*r ,(T*r,BT*r), (BT*r,1), (i_t * BT * r,0), (BT*r,BT*r),(1,0))####直接就炸内存了
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(k.dtype.element_ty)
    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dA = tl.zeros([BT*r,BT*r], dtype=tl.float32)
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_dmask = tl.zeros([BT,r,r],dtype=tl.float32)
    block_k = K//r
    for i_r in range(r):
        p_mask = tl.make_block_ptr(mask_ij + i_bh * T*r*r,(T,r,r),(r*r,r,1),(i_t*BT,0,i_r),(BT,r,1),(2,1,0))
        b_mask = tl.load(p_mask)#BT r 1
        rmask = tl.arange(0, r) == i_r #第r列
        for i_k in range(tl.cdiv(block_k, BK)):
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*block_k + i_k * BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            p_dw = tl.make_block_ptr(dw + i_bh * s_qk_h*r, (T*r, K), (s_qk_t, s_qk_d), (i_t * BT * r, i_r*block_k + i_k * BK), (BT * r, BK), (1, 0))
            b_k_beta = ((b_k * b_beta[:, None])[:,None,:]*b_mask).to(b_k.dtype)#BT*r*d
            b_k_beta = tl.reshape(b_k_beta,(BT*r,BK))
            b_dw = tl.load(p_dw, boundary_check=(0, 1))
            b_dA += tl.dot(b_dw, tl.trans(b_k_beta), allow_tf32=False)
            b_dk_beta = tl.dot(tl.trans(b_A), b_dw, allow_tf32=False)
            b_dk_beta = tl.reshape(b_dk_beta,(BT,r,BK))#

            sum_dk = tl.sum(b_dk_beta * b_mask,1)
            b_dk = sum_dk* b_beta[:, None]
            b_dbeta += tl.sum(sum_dk * b_k, 1)
            
            b_ss = (tl.sum(b_dk_beta * ((b_beta[:,None,None] * b_k[:,None,:])),-1)) # BT r
            b_dmask += (b_ss[:,:,None]*rmask[None,None,:]).to(tl.float32)#BT r r

            p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*block_k + i_k * BK), (BT, BK), (1, 0))
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    ################到这是dk

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
    #####包括求逆以及 solve beta_kKt    
    for i_r in range(r):#只取ir项 
        p_mask = tl.make_block_ptr(mask_ij + i_bh * T*r*r,(T,r,r),(r*r,r,1),(i_t*BT,0,i_r),(BT,r,1),(2,1,0))
        b_mask = tl.load(p_mask)#BT r 1
        rmask = tl.arange(0, r) == i_r #第ir列
        g = tl.sum(tl.where(rmask[None,None,None,:], b_dA, 0), -1)#BT r BT #取出第ir列
        ir_A = tl.sum(g * b_mask,1).to(k.dtype.element_ty)#BT BT

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
            b_A += beta_kkt[:,:,None,None] * ((rmask[None,None,:] * b_mask)[:,None,:,:])

            betas = (tl.sum(beta_kkt[:,None,:]*g,-1))#BT r
            b_dmask +=  (betas[:,:,None]*rmask[None,None,:]).to(tl.float32)


    p_dbeta = tl.make_block_ptr(dbeta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty), boundary_check=(0,))
    
    p_dmask = tl.make_block_ptr(dmask + (i_bh * (T) + i_t * BT)* r * r , (BT,r,r), (r*r,r,1), (0,0,0), (BT,r,r), (2,1,0))
    tl.store(p_dmask, b_dmask.to(p_dmask.dtype.element_ty), boundary_check=(0,1,2))

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
    dg = torch.empty(*g.shape,dtype=torch.float32,device=g.device)
    dmask = torch.zeros([B,H,T,r,r],device=k.device,dtype=k.dtype).contiguous()
    assert BK <= K//r
    gated_bwd_prepare_wy_repr_kernel[(NT, B*H)](
        k, v, beta, mask, g, Aw,Au,
        dw, du,
        dk, dv, dbeta,dmask,dg,
        T*K, K, 1,
        T*V, V, 1,
        T, K, V, r, BT, BK, BV
    )
    return dk, dv, dbeta, dmask,dg



class gated_ChunkDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, beta,g,mask,BT, initial_state, output_final_state=False, checkpoint_level=1):
        # print(mask)
        B,H,L,K = q.shape
        r = mask.shape[-1]
        g = chunk_local_cumsum(g,BT,head_first=True,output_dtype=torch.float) #无需变化
        #注意 mask 变成 B H T r d
        Aw,Au = gated_chunk_scaled_dot_kkt_fwd(k=k,beta=beta,g_cumsum=g,mask=mask,BT=BT,output_dtype=torch.float32)
        Aw = solve_tril(A=Aw,mask=mask,k=k,BT=BT,output_dtype=k.dtype)#bh
        Au = solve_tril(A=Au,mask=mask,k=k,BT=BT,output_dtype=k.dtype)#bh
        w, u = gated_fwd_recompute_w_u(k, v, beta, mask,Aw,Au,BT)#
        final_state = None
        if output_final_state:
            final_state = q.new_empty(q.shape[0], q.shape[1], q.shape[-1], v.shape[-1],
                                      dtype=torch.float32, requires_grad=False)#这部分不需要修正
        h, v_new = gated_chunk_fwd_h_fn(k, w, u, g, BT, initial_state, final_state)#need change'      
        o = gated_chunk_fwd_o_fn(q, k, v_new, h, g, BT) 
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
        w, u = gated_fwd_recompute_w_u(k, v, beta, mask, Aw,Au,BT)#跳过
        if h is None:
            h, v_new = gated_chunk_fwd_h_fn(k, w, u, g, BT, initial_state, None)  
        #从这里开始重新书写计算代码
        dv = gated_fwd_prepare_dv(q, k, g, do, r, BT)#qk do v_new#因此这个dv应该是一个w的shape finish  
        dh, dv = gated_chunk_bwd_dhu_fn(q, k, w, g,initial_state,do, dv, BT)#new_dv dh #final for wyper dv        
        
        dq, dk, dw , dg = gated_chunk_bwd_dqkw_fn(q, k, v_new, w, g, h, dv, do, dh, BT)#这一步也巨慢
        dk2, dv, dbeta,dmask,dg2 = gated_bwd_prepare_wy_repr(k, v, beta, mask,g, Aw,Au, dw, dv, BT)#只有这里带mask
        dk.add_(dk2)
        dg.add_(dg2)
        dg = chunk_local_cumsum(dg, BT, reverse=True,head_first=True,output_dtype=torch.float)

        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dbeta.to(beta.dtype),dg,dmask.to(mask.dtype),None, None, None


# def delta_rule_recurrence(q, k, v, beta, g, mask,initial_state=None,BT=None,output_final_state=True):
#     g_exp = torch.exp(g).float()
#     b, h, l, d_k = q.shape
#     d_v = v.shape[-1]
#     r = mask.shape[-1]
#     o = torch.zeros_like(v)
#     if initial_state == None:
#         S = torch.zeros(b, h, d_k, d_v,device=k.device,dtype=torch.float32)
#     else:
#         S = initial_state
#     if beta.ndim < v.ndim:
#         beta = beta[..., None]
#     for i in range(l):
#         _k = k[:, :, i].float()
#         _q = q[:, :, i].float()*(d_k ** -0.5)
#         _v = v[:, :, i].float()
#         beta_i = beta[:, :, i].float()
#         _v = _v * beta_i
#         kkt = torch.einsum('b h d,b h v->b h d v',_k*beta_i,_k)
#         kkt = rearrange(kkt,' b h (r d) (l v)-> b h r d l v',r= r,l=r)
#         kkt = torch.einsum('b h r d l v,b h r l->b h r d l v',kkt,mask[:,:,i,:,:].float())#16d参数，几乎可以忽略
#         kkt = rearrange(kkt,'b h r d l v-> b h (r d) (l v)')
#         iplr = torch.eye(d_k).to(q)-kkt
#         iplr = torch.einsum(' b h q k ,b h->b h q k',iplr,g_exp[:,:,i])
#         S = torch.einsum('b h q k ,b h k v->b h q v',iplr.float(),S) + _k.unsqueeze(-1).float() * _v.unsqueeze(-2).float()
#         o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q.float(), S).to(k.dtype)
#     return o,S



# def delta_rule_recurrence(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     g: torch.Tensor,
#     mask: torch.Tensor,
#     beta: torch.Tensor,
#     scale: float | None = None,
#     initial_state: torch.Tensor | None = None,
#     output_final_state: bool = False,
# ):
#     dtype = v.dtype
#     B, T, H, K, V = *q.shape, v.shape[-1]
#     if scale is None:
#         scale = K ** -0.5
#     r = mask.shape[-1]
#     print(r)
#     q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
#     q = q * scale
#     S = k.new_zeros(B, H, K, V).to(q)
#     if initial_state is not None:
#         S += initial_state
#     o = torch.zeros_like(v)
#     for i in range(0, T):
#         q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i][...,None], beta[:, i]###B H K
#         S = S * g_i[..., None].exp()####B H K V
#         k_i = rearrange(k_i,'b h (r k)-> b h r k',r=r)
#         mask_i = mask[:,i,...]###B H r r
#         w_i = k_i[:,:,None,:,:]*mask_i[:,:,:,:,None]##B H r r BK//r
#         w_i = rearrange(w_i,('b h r d k ->b h r (d k)')) 
#         _v_minus = (S[:,:,None,:,:]*w_i[:,:,:,:,None]).sum(-2)###B H r BV
#         v_new = (v_i[:,:,None,:]-_v_minus)*b_i[:,:,None,None]###B H r BV
#         hs = rearrange(torch.einsum('b h r k,b h r v-> b h r k v',k_i,v_new),'b h r k v->b h (r k) v')
#         S = S + hs
#         o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
#     if not output_final_state:
#         S = None
#     return o.to(dtype), S


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
    seq_len = v.shape[-2]
    q, k, v = map(lambda x: pad(x,BT), [q, k, v])
    dim = v.shape[-1]
    r = mask.shape[-1]
    if dim < r*16:
        q,k,v = map(lambda x:rearrange(x,'b h l (r d)->b h l r d',r=r),[q,k,v])
        q,k,v = map(lambda x:F.pad(x, (0, 16 - dim//r),value=0),[q,k,v])#基本只有32存在意义
        q,k,v = map(lambda x:rearrange(x,'b h l r d->b h l (r d)',r=r),[q,k,v])
    beta = pad_b(beta,0,BT)#bhl
    g = pad_b(g,0,BT)#bhl   
    mask = pad_m(mask,0,BT)
    q,k,v,g,beta,mask = map(lambda x:x.contiguous(),[q,k,v,g,beta,mask]) 
    o, final_state = gated_ChunkDeltaRuleFunction.apply(q, k, v, beta,g,mask, BT, initial_state, output_final_state)
    o = o[..., :seq_len,:]
    if dim < r*16:
        o = rearrange(o,'b h l (r d)->b h l r d',r=r)
        o = o[...,:dim//r]#保留dim
        o = rearrange(o,'b h l r d->b h l (r d)')
    return o, final_state


if __name__ =="__main__":
    import sys
    import time
    from fla.modules.l2norm import l2_norm as l2_norm_fn 
    # from einops import rearrange
    # sys.path.append('/mnt/jfzn/msj/flash-linear-attention-main/legacy/training/fla2-copy')
    # sys.path.append('/mnt/jfzn/msj/flash-linear-attention/evals/lm_eval/models')
    torch.set_default_dtype(torch.bfloat16)
    # seq_len = 128
    # b = 2
    # h = 2
    # k = torch.nn.functional.normalize(torch.randn(b, h, seq_len, 128), dim=-1, p=2)#d=128
    # q = torch.nn.functional.normalize(torch.randn(b, h, seq_len, 128), dim=-1, p=2)#d=128
    # v = torch.randn(b, h, seq_len, 128)
    # beta = torch.rand(b, h, seq_len).sigmoid()
    # require_grad = True
    # BT = 16
    # r = 4
    # scale = 128**-0.5
    # mask = torch.tensor([[1,1,0,0],[1,1,1,0],[0,1,1,1],[0,0,1,1]],requires_grad=False).cuda().contiguous()
    # w = torch.nn.functional.normalize(torch.randn(b,h,seq_len*r,128).cuda())
    # u = torch.nn.functional.normalize(torch.randn(b,h,seq_len*r,128).cuda())#bhn tr d
    # initial_state = torch.randn(b,h,128,128).cuda().contiguous()
    # k, v, q, beta,w, u= map(lambda x: x.cuda().requires_grad_(require_grad).contiguous(), (k, v,q, beta,w,u))

    # final_state = None
    # if False:
    #     final_state = q.new_empty(q.shape[0], q.shape[1], q.shape[-1], v.shape[-1],
    #                                 dtype=torch.float32, requires_grad=False)#这部分不需要修正
    
    # h_state, v_new = chunk_fwd_h_fn(k, w, u, BT, initial_state, final_state)#need change
    # o2 = chunk_fwd_o_fn(q, k, v_new, h_state, BT)#need change
    # o2 = rearrange(o2,'b h (n t) v-> b h n t v',n=seq_len//BT)
    # do = torch.rand_like(o2)
    # do_naive = do
    # do = rearrange(do,'b h n t v-> b h (n t) v')
    # dv0 = fwd_prepare_dv(q, k, do, r, BT)#qk do v_new#因此这个dv应该是一个w的shape finish
    # #bhrtv
    # #到这里计算结果相同
    # dh, dv = chunk_bwd_dhu_fn(q, k, w, do, dv0, BT)#new_dv dh 
    # ###到这里算的一样了
    # dq, dk, dw = chunk_bwd_dqkw_fn(q, k, v_new, w, h_state, dv, do, dh, BT)#需要dh和dv

    # #bhtrv
    # # dv0 = rearrange(dv0,'b h r (n t) v-> b h n t r v',n=seq_len//BT)
    # # dv = rearrange(dv,'b h (n t) r v->b h n t r v',n=seq_len//BT)#应该有二者在BT=1维度相等,yes
    # # dh = rearrange(dh,'b h (n k) v->b h n k v',n=seq_len//BT)
    # # 应该有
    # # NT = seq_len//BT
    # # state = torch.zeros(b,h,NT,128,128,device=q.device,dtype=q.dtype)
    # # v_new = torch.zeros_like(u)
    # # q_na,k_na,w_na,u_na,v_new = map(lambda x:rearrange(x,'b h (n t) d->b h n t d',n = NT),(q,k,w,u,v_new))
    # # wr = rearrange(w_na,'b h n (t r) k->b h n t r k',r = r)
    # # dh0 = -torch.einsum('b h t r v,b h t r k-> b h k v ',dv[:,:,1,:,:,:],wr[:,:,1,:,:,:]) 
    # # dh0 += torch.einsum('b h t v,b h t q->b h q v',do_naive[:,:,1,:,:],q_na[:,:,1,:,:])*scale
    # # k_r = rearrange(k_na,'b h n t (r d)->b h n t r d',r = r)
    # # dh0 = rearrange(dh0,'b h (r k) v->b h r k v',r=r)#need b h t r v
    # # dvs = dv0[:,:,0,:,:,:] + torch.einsum('b h r k v,b h t r k->b h t r v',dh0,k_r[:,:,0,:,:,:]) 
    # # dh0 = rearrange(dh0,'b h r k v->b h (r k) v')
    # # #这样计算流程是对的
    # # print((dv[:,:,0,:,:,:]-dvs).abs().max())
    # # print((dh0-dh[:,:,0,:,:]).abs().max())
    
    # #####here is naive
    # NT = seq_len//BT
    # state = torch.zeros(b,h,NT,128,128,device=q.device,dtype=q.dtype)
    # v_new = torch.zeros_like(u)
    # from einops import rearrange
    # q_na,k_na,w_na,u_na,v_new = map(lambda x:rearrange(x,'b h (n t) d->b h n t d',n = NT),(q,k,w,u,v_new))
    # # u_na = u_na.detach().requires_grad_(True)#b h n (t r) d
    # v_new = rearrange(v_new,'b h n (t r) d->b h n t r d',r = r)
    # if initial_state is not None:
    #     state[:,:,0,:,:] = initial_state
    # else:
    #     state[:,:,0,:,:] = 0
    # for i in range(NT):
    #     ki = rearrange(k_na[:,:,i,:,:],'b h t (r d)->b h t r d',r = r)
    #     ui = rearrange(u_na[:,:,i,:,:],'b h (t r) d->b h t r d',r = r)
    #     wi = rearrange(w_na[:,:,i,:,:],'b h (t r) d->b h t r d',r = r)
    #     v_newi = ui - torch.einsum('b h t r d, b h d v-> b h t r v',wi,state[:,:,i,:,:].clone())
    #     v_new[:,:,i,:,:,:] = v_newi.clone()#这里保存的结果是相等
    #     kui = torch.einsum('b h t r k,b h t r v-> b h r k v',ki,v_newi)
    #     if i+1 < seq_len//BT:
    #         state[:,:,i+1,:,:] = state[:,:,i,:,:].clone() + rearrange(kui,'b h r k v-> b h (r k ) v')
    # q_r = rearrange(q_na,'b h n t (r d)->b h n t r d',r = r)*scale
    # k_r = rearrange(k_na,'b h n t (r d)->b h n t r d',r = r)
    # s_r = torch.einsum('b h n t r d,b h n l r d->b h n r t l',q_r,k_r)
    # s_r = torch.tril(s_r,diagonal=0)#mask get bhnrtl
    # v_newnew = v_new#.detach().requires_grad_(True)
    # os = torch.einsum('b h n r t l, b h n l r v-> b h n t r v',s_r,v_newnew)
    # os = os.sum(dim=-2)#bhntv
    # oss = torch.einsum('b h n t q,b h n q v-> b h n t v',q_na,state)*scale#只看state 算的对不对
    # o_naive = os + oss
    # # o_naive = rearrange(o_naive,'b h n t v->b h (n t) v')
    # o_naive.backward(do_naive,retain_graph=True)
    # una_grad = u.grad
    # w_grad = w.grad
    # k_grad = k.grad
    # q_grad = q.grad
    # print((una_grad-dv).abs().max())#基本相等
    # print((k_grad-dk).abs().max())
    # print((w_grad-dw).abs().max())
    # print((q_grad-dq).abs().max())
    # print(k_grad)
    # print(dk)

    B = 8
    H = 8
    L = 227
    DK = 256
    DV = 256
    q = (torch.randn(B, H, L, DK)).cuda().requires_grad_(True)
    k = (torch.randn(B, H, L, DK)).cuda()
    k = torch.nn.functional.normalize(k, dim=-1, p=2).requires_grad_(True)
    v = (torch.randn(B, H, L, DV)).cuda().requires_grad_(True)

    r = 8
    mask = torch.randn(r,r).cuda().requires_grad_(True)

    target_matrix = torch.softmax(mask,dim=-1)#h r c
    eye_mask = torch.eye(r, dtype=torch.bool, device=target_matrix.device).unsqueeze(0)
    target_matrix = torch.where(eye_mask, torch.tensor(1.0, device=target_matrix.device), target_matrix)
    target_matrix = target_matrix.unsqueeze(1).unsqueeze(0).expand(B,H,L,r,r)


    beta = torch.randn(B, H, L).cuda().sigmoid().requires_grad_(True)
    g = torch.nn.functional.logsigmoid(torch.randn(B, H, L).cuda()).requires_grad_(True) ######g=0意味这此时Mt相等了

    do = torch.randn(B, H, L, DV).cuda()
    dict = {"q":q,"k":k,"v":v,'beta':beta,"g":g,"mask":target_matrix,"do":do}
    torch.save(dict,'/9950backfile/meishj/log.pth')


    dicts= torch.load('/9950backfile/meishj/log.pth')
    q = dicts["q"]
    k = dicts["k"]
    v = dicts["v"]
    beta = dicts["beta"]
    g = dicts["g"]
    do = dicts["do"]
    mask = target_matrix = dicts["mask"]
    B,H,L,DV = v.shape


    # g_exp = torch.exp(g)
    o11,h_11,ss = delta_rule_recurrence(q,k,v,beta,g,target_matrix)
    o11.backward(do,retain_graph=True)
    q_grad0, q.grad = q.grad, None
    k_grad0, k.grad = k.grad, None
    v_grad0, v.grad = v.grad, None
    beta_grad0, beta.grad = beta.grad, None
    g_grad0, g.grad = g.grad, None
    mask_grad0, mask.grad = mask.grad, None
    # print('done')  
    # o1 = o11
    # o1,ss = mask_fused_recurrent_gated_delta_rule(q.contiguous(),k.contiguous(),v.contiguous(),beta.contiguous(),g.contiguous(),target_matrix.contiguous(),initial_state=None,output_final_state=True)
    # o11.backward(do, retain_graph=True)
    # q_grad, q.grad = q.grad, None
    # k_grad, k.grad = k.grad, None
    # v_grad, v.grad = v.grad, None
    # beta_grad, beta.grad = beta.grad, None
    # g_grad, g.grad = g.grad, None
    # mask_grad, mask.grad = mask.grad, None
    o22,f_state = mask_gated_chunk_delta_rule(q, k, v, beta, g,target_matrix,BT=16,output_final_state=True)#10s嘛 额
    # o2.backward(do,retain_graph=True)
    o22.backward(do,retain_graph=True)
    q_grad1, q.grad = q.grad, None
    k_grad1, k.grad = k.grad, None
    v_grad1, v.grad = v.grad, None
    beta_grad1, beta.grad = beta.grad, None
    g_grad1, g.grad = g.grad, None
    mask_grad1, mask.grad = mask.grad, None
    # o = rearrange(o2,'b h t v->b t h v')
    # dicts = {"o":o,"f":f_state}
    # torch.save(dicts,"/9950backfile/meishj/o.pth")
    # dicts = torch.load("/9950backfile/meishj/o.pth")
    # o1 = dicts["o"]
    # print(o1.shape)
    # assert 1==0
    # o2.backward(do,retain_graph=True)
    # q_grad0, q.grad = q.grad, None
    # k_grad0, k.grad = k.grad, None
    # v_grad0, v.grad = v.grad, None
    # beta_grad0, beta.grad = beta.grad, None
    # g_grad0, g.grad = g.grad, None
    # mask_grad0, mask.grad = mask.grad, None
    # dict
    # dicts= torch.load('/9950backfile/meishj/o.pth')
    # # o1 = dicts["o"]
    # o1 = q_grad1
    # o2 = q_grad0
    o1_list = [o11,q_grad0,k_grad0,v_grad0,beta_grad0,g_grad0,mask_grad0]
    o2_list = [o22,q_grad1,k_grad1,v_grad1,beta_grad1,g_grad1,mask_grad1]

    for i in range(len(o1_list)):
        o1 = o1_list[i]
        o2 = o2_list[i]
        diff = ((o1 - o2)).abs()
        # if i==5:   
        #     print(f"最大差值: {max_val.item()}")
        #     print(f"坐标: {index}")
        #     print(f"recurrent 在该坐标的值: {o1[index].item()}")
        #     print(f"triton 在该坐标的值: {o2[index].item()}")
        #     for l in range(diff.shape[-1]):
        #         print(o1[0,2,l],(o2[0,2,l]))##########只有几个点不一样)##########只有几个点不一样
        max_val, flat_index = diff.max(), diff.argmax()
        index = torch.unravel_index(flat_index, diff.shape)
        print(f"最大差值: {max_val.item()}")
        print(f"坐标: {index}")
        print(f"recurrent 在该坐标的值: {o1[index].item()}")
        print(f"triton 在该坐标的值: {o2[index].item()}")