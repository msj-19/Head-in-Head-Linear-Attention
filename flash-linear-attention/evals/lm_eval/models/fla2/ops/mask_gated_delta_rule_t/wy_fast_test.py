# -*- coding: utf-8 -*-
import pdb
import torch
import triton
import triton.language as tl
from einops import rearrange
# from ...utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
# Inspired by "THE WY REPRESENTATION FOR PRODUCTS OF HOUSEHOLDER MATRICES" https://epubs.siam.org/doi/pdf/10.1137/0908009
# o: cumprod
# o2: cumprodsum

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
def fwd_prepare_wy_repr_kernel(
    k,
    v,
    beta,
    mask_ij,
    w,
    u,
    A,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    T,
    K,
    V,
    r:  tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
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
    b_A = -tl.where((tl.arange(0, BT)[:,None] > tl.arange(0, BT)[None,:])[:,:,None,None], b_A, 0)
    #先save这个看看

    for i in range(1, BT):#此时矩阵为 BT,r,BT,r
        mask = tl.arange(0, BT) == i 
        b_a = tl.sum(tl.where(mask[:,None,None,None], b_A, 0), 0)#get ba BT*r*r
        q = tl.sum(b_a[:,None,:,:,None]*b_A[:,:,None,:,:],-2)#矩阵乘法解决，get BT,BT*r*r
        b_a = b_a + tl.sum(q,0)*((tl.arange(0, BT) < i)[:,None,None])#BT*r*r
        b_A = tl.where(mask[:,None,None,None],b_a,b_A)#按行计算 ，逐步交换结果
    b_A += ((tl.arange(0, BT)[:, None, None, None] == tl.arange(0, BT)[None, :, None, None])&(tl.arange(0, r)[None, None, :, None] == tl.arange(0, r)[None, None, None, :]))
    b_A = tl.permute(b_A,(0,2,1,3))
    b_A = tl.reshape(b_A,(BT*r,BT*r))#BT*r BT*r
    p_A = tl.make_block_ptr(A + i_bh*T*BT*r*r ,(T*r,BT*r), (BT*r,1), (i_t*BT*r,0), (BT*r,BT*r),(1,0))#旧版本实现需要很多乘法
    tl.store(p_A, (b_A).to(p_A.dtype.element_ty),boundary_check=(0, 1))
    #解决矩阵求逆
    b_A = b_A.to(k.dtype.element_ty)#ok 解决求逆了 #下一步计算结果

    for i_r in range(r):
        p_mask = mask_ij + tl.arange(0,r)*r+i_r#读取第ir列
        b_mask = tl.load(p_mask)
        for i_k in range(tl.cdiv(dk, BK)):
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*dk + i_k * BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)[:,None,:]*b_mask[None,:,None].to(b_k.dtype)#BT*r*d
            b_kb = tl.reshape(b_kb,(BT*r,BK))
            b_w = tl.dot(b_A, b_kb, allow_tf32=False)#get BT*r *BK
            p_w = tl.make_block_ptr(w + i_bh * s_qk_h*r, (T*r, K), (s_qk_t, s_qk_d), (i_t * BT * r, i_r*dk + i_k * BK), (BT*r, BK), (1, 0))
            tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):#no need for 任意mask不使用 #无需for 循环 ，这里也不存在mask
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)[:,None,:]*tl.full([r],1, dtype=b_v.dtype)[None,:,None]
        b_vb = tl.reshape(b_vb,(BT*r,BV))
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
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
    key=["BT", "BK", "BV"],
)
@triton.jit
def fwd_recompute_w_u_kernel(
    k,
    v,
    beta,
    mask_ij,
    w,
    u,
    A,
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
    p_A = tl.make_block_ptr(A + i_bh*T*BT*r*r ,(T*r,BT*r), (BT*r,1), (i_t*BT*r,0), (BT*r,BT*r),(1,0))
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(k.dtype.element_ty)
    for i_r in range(r):
        # r_mask = tl.arange(0, r) == i_r # 
        p_mask = mask_ij + tl.arange(0,r)*r+i_r#读取第ir列
        b_mask = tl.load(p_mask)
        for i_k in range(tl.cdiv(dk, BK)):
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*dk + i_k * BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)[:,None,:]*b_mask[None,:,None].to(b_k.dtype)#BT*r*d
            b_kb = tl.reshape(b_kb,(BT*r,BK))
            b_w = tl.dot(b_A, b_kb, allow_tf32=False)#get BT*r *BK
            p_w = tl.make_block_ptr(w + i_bh * s_qk_h*r, (T*r, K), (s_qk_t, s_qk_d), (i_t * BT * r, i_r*dk + i_k * BK), (BT*r, BK), (1, 0))
            tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):#no need for 任意mask不使用 #无需for 循环 ，这里也不存在mask
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)[:,None,:]*tl.full([r],1, dtype=b_v.dtype)[None,:,None]
        b_vb = tl.reshape(b_vb,(BT*r,BV))
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        p_u = tl.make_block_ptr(u + i_bh * s_vo_h*r, (T*r, V), (s_vo_t, s_vo_d), (i_t * BT*r, i_v * BV), (BT*r, BV), (1, 0))
        tl.store(p_u, (b_u).to(p_u.dtype.element_ty), boundary_check=(0, 1))

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
def bwd_prepare_wy_repr_kernel(
    k, v, beta,mask_ij,A,
    dw, du,
    dk, dv, dbeta,dmask,
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
    p_A = tl.make_block_ptr(A + i_bh*T*BT*r*r ,(T*r,BT*r), (BT*r,1), (i_t * BT * r,0), (BT*r,BT*r),(1,0))
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(k.dtype.element_ty)
    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dA = tl.zeros([BT*r,BT*r], dtype=tl.float32)
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    b_dmask = tl.zeros([r,r],dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):#分块r 
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(du + i_bh * s_vo_h * r, (T * r, V), (s_vo_t, s_vo_d), (i_t * BT * r, i_v * BV), (BT * r, BV), (1, 0))#r*BT BV
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v_beta = ((b_v * b_beta[:, None])[:,None,:]*tl.full([r],1, dtype=b_v.dtype)[None,:,None]).to(b_v.dtype)##BT*r*BV
        b_v_beta = tl.reshape(b_v_beta,(BT*r,BV))
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA += tl.dot(b_du, tl.trans(b_v_beta), allow_tf32=False)#BT*r,BT*r
        b_dv_beta = tl.dot(tl.trans(b_A), b_du, allow_tf32=False)#BT*r,BV
        b_dv_beta = tl.reshape(b_dv_beta,(BT,r,BV))#
        sum_dv = tl.sum(b_dv_beta,-2)#这里不一样，结果
        b_dv = (sum_dv * b_beta[:, None])#？哪一步结果不一样呢
        b_dbeta += tl.sum(sum_dv * b_v, 1)
        p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
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


            b_ss = b_dk_beta * b_beta[:,None,None] * b_k[:,None,:]
            b_ss = tl.reshape(tl.permute(b_ss,(2,0,1)),(BT*BK,r))
            b_ss = tl.sum(b_ss,0)
            # b_ss = (tl.sum(tl.sum(b_dk_beta * b_beta[:,None,None] * b_k[:,None,:],0),-1))
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
    

    b_dA = tl.reshape(b_dA,(BT,r,BT,r))
    #bt r bt r


    for i_r in range(r):#只取ir项 
        p_mask = mask_ij + tl.arange(0,r)*r+i_r#读取第ir列
        b_mask = tl.load(p_mask)#第ir列
        rmask = tl.arange(0, r) == i_r #第ir列
        g = tl.sum(tl.where(rmask[None,None,None,:], b_dA, 0), -1)#BT r BT #取出第ir列
        ir_A = tl.sum(g * b_mask[None,:,None],1).to(k.dtype.element_ty)#BT BT
        #对应的c部分

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

            beta_y = (beta_kkt[:,None,:]*g)
            beta_y = tl.reshape(tl.permute(beta_y,(2,0,1)),(BT*BT,r))
            betas = tl.sum(beta_y,0)
            b_dmask +=  (betas[:,None]*rmask[None,:]).to(tl.float32)

    p_dbeta = tl.make_block_ptr(dbeta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty), boundary_check=(0,))
    
    p_dmask = tl.make_block_ptr(dmask + (i_bh * (T//BT) + i_t)* r * r , (r,r), (r,1), (0,0), (r,r), (1,0))
    tl.store(p_dmask, b_dmask.to(p_dmask.dtype.element_ty), boundary_check=(0,1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16)
    ],
    key=["BT", "BK", "r"],
)
@triton.jit
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    beta,
    mask_ij,
    A,
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

    p_A21 = tl.make_block_ptr(A + (i_bh)*s_A_bh, (T,32,r,r),(32*r*r,r*r,r,1) ,(i_t * 32 + 16, 0, 0, 0), (16, 16,r,r), (3,2,1,0))
    b_A21 = tl.load(p_A21, boundary_check=(0,1,2,3)).to(tl.float32)
    b_A21 = tl.permute(b_A21,(0,2,1,3))
    b_A21 = tl.reshape(b_A21,(16*r,16*r))#BT*r BT*r

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

def chunk_scaled_dot_kkt_fwd(k,beta,mask,BT,output_dtype=torch.float32):
    B, H, T, K = k.shape
    r = mask.shape[-1]
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K//r), 64)
    A = torch.empty(B*H*NT,BT*BT,r*r,device=k.device, dtype=output_dtype).contiguous()                                                                                                                         
    chunk_scaled_dot_kkt_fwd_kernel[(NT, B*H)](
        k, beta, mask, A,
        T*K, K, 1,
        T, K, r, BT, BK
    )
    return A

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

    NT = triton.cdiv(T, BT)
    Ai = torch.zeros(B,H,NT*BT*r,BT*r,device=A.device, dtype=output_dtype)
    merge_16x16_to_32x32_inverse_kernel[(NT, B*H)](
        A,Ad,Ai,
        T*BT*r*r,#s_a_bh and s_ai_bh
        T*16*r*r,#s_ad_bh
        T,r,BT
    )
    return Ai


def fwd_prepare_wy_repr2(k, v, beta,mask, BT):
    A = chunk_scaled_dot_kkt_fwd(k,beta,mask,BT,torch.float32)
    A = solve_tril(A=A,mask=mask,k=k,BT=BT,output_dtype=k.dtype)
    w, u = fwd_recompute_w_u(k, v, beta,mask, A, BT)
    return w, u, A

def fwd_prepare_wy_repr(k, v, beta,mask, BT):
    B, H, T, K, V = *k.shape, v.shape[-1]
    r = mask.shape[-1]
    u = torch.empty(B,H,r*T,V,device=k.device, dtype=k.dtype)
    w = torch.empty(B,H,r*T,K,device=k.device, dtype=k.dtype)
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K//r), 64)
    BV = min(triton.next_power_of_2(V), 64)
    A = torch.empty(B,H,NT*BT*r,BT*r,device=k.device, dtype=k.dtype)                                                                                                                         
    fwd_prepare_wy_repr_kernel[(NT, B*H)](
        k, v, beta, mask, w, u, A,
        T*K, K, 1,
        T*V, V, 1,
        T, K, V, r, BT, BK, BV
    )
    return w, u, A


def fwd_recompute_w_u(k, v, beta,mask, A, BT):
    B, H, T, K, V = *k.shape, v.shape[-1]
    r = mask.shape[-1]
    u = torch.empty(B,H,r*T,V,device=k.device, dtype=k.dtype)
    w = torch.empty(B,H,r*T,K,device=k.device, dtype=k.dtype)
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K//r), 64)#32
    BV = min(triton.next_power_of_2(V), 64)
    fwd_recompute_w_u_kernel[(NT, B*H)](
        k, v, beta,mask, w, u, A,
        T*K, K, 1,
        T*V, V, 1,
        T, K, V, r,BT, BK, BV
    )
    return w, u

def bwd_prepare_wy_repr(k, v, beta, mask, A, dw, du, BT):
    B, H, T, K, V = *k.shape, v.shape[-1]
    r = mask.shape[-1]
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K//r), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT = triton.cdiv(T, BT)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v).contiguous()
    dbeta = torch.zeros_like(beta)
    dmask = torch.zeros([B*H*NT,r,r],device=k.device,dtype=k.dtype).contiguous()
    bwd_prepare_wy_repr_kernel[(NT, B*H)](
        k, v, beta, mask, A,
        dw, du,
        dk, dv, dbeta,dmask,
        T*K, K, 1,
        T*V, V, 1,
        T, K, V, r, BT, BK, BV
    )
    dmask = dmask.sum(0)
    return dk, dv, dbeta, dmask


class WYRepresentationPrepration(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, k, v, beta,mask,chunk_size=64):
        ctx.BT = chunk_size
        w, u, A = fwd_prepare_wy_repr(k, v,beta,mask, ctx.BT)
        ctx.save_for_backward(k, v, beta,mask,A)
        return w, u
    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, dw, du):
        k, v, beta,mask, A = ctx.saved_tensors
        BT = ctx.BT
        dk, dv, dbeta,dmask = bwd_prepare_wy_repr(k, v, beta,mask, A, dw, du, BT)
        return dk, dv, dbeta, dmask, None

prepare_wy_repr = WYRepresentationPrepration.apply


def naive(k, v, beta,maskij,chunk_size):
    l_org = k.shape[2]
    l_new = triton.next_power_of_2(l_org)
    k = torch.cat([k, torch.zeros_like(k)[:, :, :l_new-l_org, :]], dim=2)
    v = torch.cat([v, torch.zeros_like(v)[:, :, :l_new-l_org, :]], dim=2)
    beta = torch.cat([beta, torch.zeros_like(beta)[:, :, :l_new-l_org]], dim=2)
    k, v = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), (k, v))
    beta = rearrange(beta, 'b h (n c) -> b h n c', c=chunk_size)
    
    b,h,nt,BT,dk = k.shape
    dv = v.shape[-1]
    r = maskij.shape[-1] 
    k_beta = k * beta[..., None]
    k_beta = rearrange(k_beta,'b h n t (r k)->b h n t r k', r=r)
    k_beta = torch.einsum('b h n t r k,l r-> b h n t l r k',k_beta,maskij)
    k_beta = rearrange(k_beta,'b h n t l r k->b h n t l (r k)')#l=1 rk=org
    v_beta = v * beta[..., None]
    v_beta = v_beta
    v_beta = v_beta.unsqueeze(-2).expand(-1,-1,-1,-1,r,-1)
    ki = rearrange(k,'b h n c (r k)-> b h n r c k',r=r)
    
    attn = (ki @ ki.transpose(-1, -2))
    attn = torch.tril(attn, diagonal=-1)#bhnr cc
    attn = torch.einsum('b h n r t l,c r->b h n t l c r',attn,maskij)#bhn  rr cc
    attn = torch.einsum('b h n t l c r,b h n t->b h n t l c r',attn,beta)

    o = torch.zeros_like(k_beta)
    o2 = torch.zeros_like(v_beta)

    o[..., 0, :,:] = k_beta[..., 0,:,:].clone()
    o2[..., 0,:, :] = v_beta[..., 0,:,:].clone()
    for i in range(1, chunk_size):
        o_i = (o[..., :i,:,:]).clone()#bhn :t cc  
        o[..., i,:,:] =  (-(attn[:,:,:,i, :i,:,:]@o_i).sum(3) + k_beta[..., i,:,:])
        o2_i = (o2[..., :i,:,:]).clone()#少一个维度
        o2[..., i,:,:] = (-(attn[:,:,:,i, :i,:,:]@o2_i).sum(3) + v_beta[..., i,:,:])
    return map(lambda x: rearrange(x, 'b h n c r k -> b h (n c r) k'), (o, o2))


if __name__ == "__main__":
    #all compute here
    import sys
    torch.manual_seed(42)  
    sys.path.append('/mnt/jfzn/msj/flash-linear-attention-main/legacy/training/fla2-copy')
    torch.set_default_dtype(torch.bfloat16)
    seq_len = 128
    b = 2
    h = 2
    k = torch.nn.functional.normalize(torch.randn(b, h, seq_len, 128), dim=-1, p=2)#d=128
    v = torch.randn(b, h, seq_len, 128)
    beta = torch.rand(b, h, seq_len).sigmoid()
    require_grad = True
    BT = 32
    k, v, beta = map(lambda x: x.cuda().requires_grad_(require_grad).contiguous(), (k, v, beta))
    r = 4
    # mask = torch.tensor([[1,1,0,0],[0.5,1,0.5,0],[0,0.5,1,0.5],[0,0,1,1]]).cuda().contiguous()
    mask = torch.randn([r,r])
    mask = mask.cuda().requires_grad_(require_grad).contiguous()
    # w,u,a0 = fwd_prepare_wy_repr(k,v,beta,mask, 16)
    # w2,u2 = fwd_recompute_w_u(k,v,beta,mask,a0,16)
    # from einops import rearrange

    k2 = rearrange(k,'b h (n t) (r k)-> b h n r t k',t = BT,r=r)
    b2 = rearrange(beta,'b h (n t)-> b h n t',t = BT)
    a1 = (k2*b2.unsqueeze(-2).unsqueeze(-1))@k2.transpose(-1,-2)#bhnrtt
    qq = torch.tril(a1,diagonal=-1)
    qq = torch.einsum('b h n r t l,c r-> b h n t c l r',qq,mask)
    sf = rearrange(qq,'b h n t c l r->b h n (t c) (l r)')
    sf = rearrange(sf,'b h n (t c) (l r)->b h n t l c r',c=r ,r =r)#这个
    
    # #长条对角线
    i_mask = ((torch.arange(0, BT)[:, None, None, None] == torch.arange(0, BT)[None, :, None, None]) & (torch.arange(0, r)[None, None, :, None] == torch.arange(0, r)[None, None, None, :]))
    s = sf+i_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda()
    s = rearrange(s,'b h n a d c r->b h n (a c) (d r)')
    s = torch.linalg.inv(s.float()).to(k)#矩阵逆#bhn tr tr
    

    # A = chunk_scaled_dot_kkt_fwd(k,beta,mask,BT,output_dtype=torch.float32)#bh nt BT bt r r
    # Ad = solve_tril(A,mask,k,BT,output_dtype=torch.bfloat16)
    # s = rearrange(s,'b h n a c->(b h n) a c')
    # print(Ad.shape)
    # print(s.shape)

    w,u,As = fwd_prepare_wy_repr2(k, v, beta,mask, BT)
    # w2,u2,Ad2 = fwd_prepare_wy_repr(k, v, beta,mask, BT)
    
    # print((w2-w).abs().max())
    # print((u2-u).abs().max())
    # print((As-Ad2).abs().max())

    # print((Ad-s).abs().max())
    # print(Ad-s)

    # print((As-s).abs().max())
    # print(As-s)
    # B*H*NT,BT*r,16*r
    # k_exp = torch.einsum('b h n r t k,b h n t-> b h n r t k',k2,b2)
    # k_exp = torch.einsum('b h n r t k,c r-> b h n r t k c',k_exp,mask)
    # k_exp = rearrange(k_exp,'b h n r t k c->b h n (t c) (r k)')
    # wc = s_copy@k_exp

    # v_exp = rearrange(v,'b h (n t) v-> b h n t v',t = BT)
    # v_exp = torch.einsum('b h n t v,b h n t-> b h n t v',v_exp,b2)
    # v_exp = v_exp.unsqueeze(4).expand(-1,-1,-1,-1,r,-1)
    # v_exp = rearrange(v_exp, ' b h n t r v-> b h n (t r) v')
    # uc = s_copy@v_exp
    # wc,uc = map(lambda x: rearrange(x,"b h n t r->b h (n t) r"), (wc,uc))
    # do = torch.rand_like(wc)
    # do2 = torch.rand_like(uc)#b h n t t
    # o1, o2 = naive(k.clone(), v.clone(), beta.clone(),mask.clone(), BT)#这个代码有问题
    # do = torch.rand_like(o1)
    # do2 = torch.rand_like(o2)#b h n t t
    # if require_grad:
    #     o1.backward(do, retain_graph=True)
    #     o2.backward(do2, retain_graph=True)
    #     k_grad2, v_grad2, beta_grad2,mask_grad2 = k.grad, v.grad, beta.grad, mask.grad

    # w0,u0,s0 = fwd_prepare_wy_repr(k, v, beta,mask, 16)
    # k_grad, v_grad, beta_grad,mask_grad = bwd_prepare_wy_repr(k,v,beta,mask,s0,do,do2,BT)   
    
    # print((o1-w0).abs().max())
    # print((o2-u0).abs().max())
    # print((k_grad-k_grad2).abs().max())
    # print((v_grad-v_grad2).abs().max())
    # print((beta_grad-beta_grad2).abs().max())
    # print((mask_grad-mask_grad2).abs().max())
    # print(mask_grad)
    # print(mask_grad2)


