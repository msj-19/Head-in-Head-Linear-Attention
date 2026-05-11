# -*- coding: utf-8 -*-

import torch
from einops import rearrange


import time
import torch
import triton
import triton.language as tl
from einops import rearrange
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
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
        p_mask = mask_ij + tl.arange(0,r)* r + i_r
        b_mask = tl.load(p_mask)
        ij_mask = b_mask[:,None]*r_mask[None,:]
        for i_k in range(tl.cdiv(dk, BK)):#分块k读取计算
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r * dk + i_k * BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
            b_kb = (b_k).to(b_k.dtype)
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
    b_A = tl.permute(b_A,(0,2,1,3))
    b_A = tl.reshape(b_A,(BT*r,BT*r))#BT*r BT*r
    b_A += tl.arange(0, BT*r)[:,None] == tl.arange(0, BT*r)[None,:]
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
            b_kb = (b_k).to(b_k.dtype)[:,None,:]*b_mask[None,:,None].to(b_k.dtype)#BT*r*d
            # b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)[:,None,:]*b_mask[None,:,None].to(b_k.dtype)#BT*r*d
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
        p_u = tl.make_block_ptr(u + i_bh * s_vo_h*r, (T*r, V), (s_vo_t, s_vo_d), (i_t * BT * r, i_v * BV), (BT*r, BV), (1, 0))
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
        p_mask = mask_ij + tl.arange(0,r)*r+i_r#读取第ir列
        b_mask = tl.load(p_mask)
        for i_k in range(tl.cdiv(dk, BK)):
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*dk + i_k * BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_kb = (b_k).to(b_k.dtype)[:,None,:]*b_mask[None,:,None].to(b_k.dtype)#BT*r*d
            # b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)[:,None,:]*b_mask[None,:,None].to(b_k.dtype)#BT*r*d
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
    dk, dv, dbeta,
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
        p_mask = mask_ij + tl.arange(0,r)*r+i_r 
        b_mask = tl.load(p_mask)
        for i_k in range(tl.cdiv(block_k, BK)):#assert block_k = BK
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*block_k + i_k * BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            p_dw = tl.make_block_ptr(dw + i_bh * s_qk_h*r, (T*r, K), (s_qk_t, s_qk_d), (i_t * BT * r, i_r*block_k + i_k * BK), (BT * r, BK), (1, 0))
            # b_k_beta = ((b_k * b_beta[:, None])[:,None,:]*b_mask[None,:,None]).to(b_k.dtype)#BT*r*d
            b_k_beta = ((b_k)[:,None,:]*b_mask[None,:,None]).to(b_k.dtype)
            b_k_beta = tl.reshape(b_k_beta,(BT*r,BK))
            b_dw = tl.load(p_dw, boundary_check=(0, 1))
            b_dA += tl.dot(b_dw, tl.trans(b_k_beta), allow_tf32=False)
            b_dk_beta = tl.dot(tl.trans(b_A), b_dw, allow_tf32=False)#get BT*r*BT*r
            b_dk_beta = tl.reshape(b_dk_beta,(BT,r,BK))
            sum_dk = tl.sum(b_dk_beta * b_mask[None,:,None],1)
            # b_dk = sum_dk* b_beta[:, None]
            b_dk = sum_dk
            # b_dbeta += tl.sum(sum_dk * b_k, 1)
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
    b_dA = tl.where(da_mask, -b_dA, 0)
    b_dA = tl.reshape(b_dA,(BT,r,BT,r)).to(k.dtype.element_ty)#到这应该都是对的

    for i_r in range(r):#只取ir项 
        p_mask = mask_ij + tl.arange(0,r)*r+i_r#读取第ir列
        b_mask = tl.load(p_mask)#第ir列
        mask = tl.arange(0, r) == i_r 
        g = tl.sum(tl.where(mask[None,None,None,:], b_dA, 0), -1)#BT r BT 取最后一列，
        #这里对应 kr 部分
        ir_A = tl.sum(g * b_mask[None,:,None],1).to(k.dtype.element_ty)#BT BT
        for i_k in range(tl.cdiv(block_k, BK)):
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*block_k + i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*block_k + i_k * BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dk = tl.load(p_dk, boundary_check=(0, 1))
            # b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)
            b_k_beta = (b_k).to(b_k.dtype)

            b_dk_beta = tl.dot(ir_A, b_k, allow_tf32=False)
            # b_dbeta += tl.sum(b_dk_beta * b_k, 1)
            b_dk += tl.dot(tl.trans(ir_A), b_k_beta, allow_tf32=False)
            b_dk += b_dk_beta #* b_beta[:, None]
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))#这里也没问题吧
    p_dbeta = tl.make_block_ptr(dbeta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty), boundary_check=(0,))

def fwd_prepare_wy_repr(k, v, beta,mask, BT):
    B, H, T, K, V = *k.shape, v.shape[-1]
    r = mask.shape[-1]
    u = torch.empty(B,H,r*T,V,device=k.device, dtype=k.dtype)
    w = torch.empty(B,H,r*T,K,device=k.device, dtype=k.dtype)
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K//r), 64)
    assert BK == K//r 
    BV = min(triton.next_power_of_2(V), 64)
    A = torch.empty(B,H,NT*BT*r,BT*r,device=k.device, dtype=torch.float32)                                                                                                                     
    fwd_prepare_wy_repr_kernel[(NT, B*H)](
        k, v, beta, mask, w, u, A,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
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
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
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
    assert BK == K//r
    bwd_prepare_wy_repr_kernel[(NT, B*H)](
        k, v, beta, mask, A,#da,
        dw, du,
        dk, dv, dbeta,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        T, K, V, r, BT, BK, BV
    )
    return dk, dv, dbeta#,da


# from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
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
def fwd_prepare_dv_kernel(
    q,
    k,
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
        b_q = (b_q * scale).to(b_k.dtype)
        b_A += tl.dot(b_k, b_q, allow_tf32=False)
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], b_A, 0).to(do.dtype.element_ty)
    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        p_dv = tl.make_block_ptr(dv + i_bhr * s_vo_h , (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_dv = tl.dot(b_A, b_do, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

#finish
def fwd_prepare_dv(q, k, do, r,BT):
    B, H, T, K, V = *k.shape, do.shape[-1]
    dv = torch.empty(B,H,r,T,V,device = do.device, dtype= do.dtype)#没法like
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K//r),64)
    BV = min(triton.next_power_of_2(V), 64)
    fwd_prepare_dv_kernel[(NT, B*H*r)](
        q, k, do, dv,
        k.stride(1), k.stride(2), k.stride(3),
        do.stride(1), do.stride(2), do.stride(3),
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
def chunk_delta_rule_fwd_kernel_h(
    k,
    v,#u
    d,#w
    v_new,
    h,
    initial_state,  # initial state of the chunk [B, H, D_head_K, D_head_V]
    final_state,  # final state of the chunk [B, H, D_head_K, D_head_V]
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
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
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)#assert ik=1 all use
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
                b_v = tl.load(p_v, boundary_check=(0, 1, 2))#BC
                b_v = tl.reshape(b_v,(BC,BV))
                b_d = tl.reshape(b_d,(BC,BK))
                b_v -= tl.dot(b_d, b_h.to(b_k.dtype), allow_tf32=False)#ok #到这相等的 这里BC
                tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))#至少到这里第一步结果相同
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
def chunk_linear_attn_fwd_kernel_o(
    q,
    k,
    v,
    h,
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
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K//r, BK)):#这里需要注意拆分#这里K//BK = r
        #问题是不同r_block读取了同一份qk，有影响吗
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r * rk + i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r * rk + i_k * BK), (BT, BK), (1, 0))
        # p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_r * rk + i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_r * rk + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        b_k = tl.trans(tl.load(p_k, boundary_check=(0, 1)))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)

    b_s = tl.where(m_s, b_s, 0)#置为0 Bs = 0
    p_v = tl.make_block_ptr(v + i_bhr * T * V, (T, V), (V, 1), (i_t * BT , i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o + (tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False))
    p_o = tl.make_block_ptr(o + i_bhr * T * V, (T, V), (V,1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

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
def chunk_delta_rule_bwd_kernel_dhu(
    q,
    k,
    d,
    do,
    dh,
    dv,
    dv2,
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
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    r: tl.constexpr,
    KR: tl.constexpr,
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)#这个不变 读取所有
    for i_t in range(NT - 1, -1, -1):# 向前偏移了一位,计算流程是对的
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V , (K, V), (s_h_t, 1), (i_k * BK , i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1)) 
        b_dh_tmp = tl.zeros([BK, BV], dtype=tl.float32)
        #全列
        for i_c in range(tl.cdiv(BT, BC) - 1, -1, -1):
            p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
                                    (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))#全读取
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
                                    (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))#  
            p_d = tl.make_block_ptr(d + i_bh * (T * K * r), (T*r,K), (K, 1),
                                    (i_t * BT * r + i_c * BC *r,i_k * BK), (BC * r,BK), (1, 0))#读取 BC r BK的内容
            p_dv = tl.make_block_ptr(dv + i_bh * r * T * V, (T*r, V), (V , 1),
                                    (i_t * BT * r + i_c * BC * r, i_v * BV), (BC*r, BV), (1, 0))
            p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1),
                                    (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            b_q = (tl.load(p_q, boundary_check=(0, 1)))
            b_q = (b_q * scale).to(b_q.dtype)
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))#BT*r Bv 
            b_d = tl.trans(tl.load(p_d,boundary_check=(0, 1)))
            b_k = tl.permute(tl.reshape(b_k,(BC,r,KR)),(1,0,2))#r BC KR
            b_dhtrans = tl.reshape(b_dh,(r,KR,BV))
            dv_sum = tl.sum(b_k[:,:,:,None]*b_dhtrans.to(b_k.dtype)[:,None,:,:],-2) #get r BC BV
            b_dv += tl.reshape(tl.permute(dv_sum,(1,0,2)),(BC*r,BV))
            #bhtrv
            p_dv2 = tl.make_block_ptr(dv2 + i_bh * r * T * V, (T*r, V), (V , 1),
                                    (i_t * BT * r + i_c * BC * r, i_v * BV), (BC*r, BV), (1, 0))
            tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
            b_dh_tmp += tl.dot(b_q, b_do.to(b_q.dtype), allow_tf32=False)
            b_dh_tmp -= tl.dot(b_d,b_dv.to(b_q.dtype),allow_tf32=False)
        b_dh += b_dh_tmp

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
def chunk_delta_rule_bwd_kernel_dqkw(
    q,
    k,
    v,
    w,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    dw,
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
    NT: tl.constexpr,
    r: tl.constexpr,
):
    i_k, i_t, i_bhr = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_r = i_bhr%r
    i_bh = i_bhr//r
    o_i = tl.arange(0, BT)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*K//r + i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*K//r + i_k * BK), (BT, BK), (1, 0))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT,r,BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bhr * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, (NT * K, V), (s_h_t, 1), (i_t * K +  i_r * K // r + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, (NT * K, V), (s_h_t, 1), (i_t * K + i_r* K// r + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h*r, (T*r, V), (s_vo_t, s_vo_d), (i_t * BT * r, i_v * BV), (BT * r, BV), (1, 0))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.trans(tl.load(p_h, boundary_check=(0, 1)))#BV BK
        # [BK, BV]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        # [BT, BT]
        b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)#ok 
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)#d_do 全， bh应该包含 i_Kbufen
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)#用来计算dk,yes 行独立没问题
        b_dv = tl.reshape(tl.load(p_dv, boundary_check=(0, 1)),(BT,r,BV))#BT*r BV
        b_dw += tl.sum(b_dv.to(b_v.dtype)[:,:,:,None]*b_h.to(b_v.dtype)[None,None,:,:],-2)#get BT r BK
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds, 0).to(b_q.dtype)#BT*BT
    b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
    b_dq *= scale
    b_dk += tl.trans(tl.dot(tl.trans(b_q), b_ds, allow_tf32=False)) #这些应该没啥问题

    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*K//r + i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_r*K//r + i_k * BK), (BT, BK), (1, 0))
    p_dw = tl.make_block_ptr(dw + i_bh * T*r*K, (T, r, K), (r*K,K,1), (i_t * BT, 0 ,i_r*K//r + i_k * BK), (BT, r ,BK), (2, 1, 0))
    # p_dw = tl.make_block_ptr(dw + i_bh *  T*r*K, (T, r, K), (r*K,K,1), (i_t * BT ,i_r, i_k * BK), (BT, 1, BK), (2, 1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dw, (tl.reshape(-b_dw.to(p_dw.dtype.element_ty),(BT,r,BK))), boundary_check=(0, 1))

#finish
def chunk_fwd_h_fn(k, w, u, BT, initial_state, final_state):
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
    grid = (NK, NV, B * H)
    v_new = torch.empty(B,H,r,T,V,dtype=u.dtype,device=u.device)#做了v_new的r_first
    chunk_delta_rule_fwd_kernel_h[grid](#r没有for循环
        k, u, w, v_new, h, initial_state, final_state,
        k.stride(1), k.stride(2), k.stride(3),
        u.stride(1), u.stride(2), u.stride(3), #rt*v,v,1
        h.stride(1), h.stride(2), 
        H=H, T=T, K=K, V=V, BT=BT, BC=BC, BK=BK, BV=BV, NT=NT,r=r,      
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
    )
    return h, v_new

#finish
def chunk_bwd_dhu_fn(q, k, w, do, dv, BT):
    B,H,r,T,V,K = *dv.shape,q.shape[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension being larger than 256."
    BV = 16 if BK > 128 else 32
    BV = 64 if BK <= 64 else BV
    BC = 16 if BK > 128 else 32
    BC = 64 if BK <= 64 else BC
    BC = min(BT, BC)
    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)#感觉可以放并行度
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'

    dh = q.new_empty(B , H, NT * K,V)#一样的#need 求和 得一起算
    grid = (NK, NV, B * H)
    dv = rearrange(dv,'b h r t v-> b h (t r) v').contiguous()
    dv2 = torch.empty_like(dv)#一样的 #bhr T V
    chunk_delta_rule_bwd_kernel_dhu[grid](
        q, k, w, do, dh, dv, dv2,
        q.stride(1), q.stride(2), q.stride(3),
        do.stride(1), do.stride(2), do.stride(3),
        dh.stride(1), dh.stride(2),
        K**-0.5,
        H=H, T=T, K=K, V=V, BT=BT, BC=BC, BK=BK, BV=BV, NT=NT,r=r,KR = K//r,
    )
    return dh, dv2

#finish
def chunk_fwd_o_fn(q, k, v_new, h, BT):
    B,H,r,T,V,K = *v_new.shape,q.shape[-1]
    BK = triton.next_power_of_2(K//r)
    o = torch.empty_like(v_new)#there_fore,bhr nT,bv
    BK = min(triton.next_power_of_2(K//r), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)
    grid = (NV, NT, B * H * r)
    #h shape b h nk v
    chunk_linear_attn_fwd_kernel_o[grid](
        q, k, v_new, h, o,
        q.stride(1), q.stride(2), q.stride(3),
        v_new.stride(1), v_new.stride(2), v_new.stride(3),
        h.stride(1), h.stride(2),
        scale=K**-0.5,
        H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,r = r,
    )
    o = o.sum(dim=2)#沿着r维度求和
    return o


def chunk_bwd_dqkw_fn(q, k, v_new, w, h, du, do, dh, BT):
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
    chunk_delta_rule_bwd_kernel_dqkw[grid](
        q, k, v_new, w, h, do, dh, dq, dk, du, dw,
        q.stride(1), q.stride(2), q.stride(3),
        T*V, V, 1,
        dh.stride(1), dh.stride(2),
        scale=K ** -0.5,
        H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,r = r
    )
    return dq.to(q.dtype), dk.to(k.dtype), dw.to(w.dtype)


class ChunkDeltaRuleFunction(torch.autograd.Function):
    #前向写完了
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, beta,mask,BT, initial_state, output_final_state, checkpoint_level=1):
        start = time.time()
        w, u, A = fwd_prepare_wy_repr(k, v,beta, mask, BT)#compute for A matrix #compute all
        final_state = None
        if output_final_state:
            final_state = q.new_empty(q.shape[0], q.shape[1], q.shape[-1], v.shape[-1],
                                      dtype=torch.float32, requires_grad=False)#这部分不需要修正
        end = time.time()
        print('compute_A:',end-start)
        start = time.time()
        h, v_new = chunk_fwd_h_fn(k, w, u, BT, initial_state, final_state)#need change'
        end = time.time()
        print('compute_h_s:',end-start)
        
        start = time.time()
        o = chunk_fwd_o_fn(q, k, v_new, h, BT)#need change
        end = time.time()
        print('compute_h_s:',end-start)
        if checkpoint_level == 1:
            h, v_new = None, None
        ctx.save_for_backward(q, k, v, beta,mask, A, h, v_new, initial_state)
        ctx.BT = BT
        return o.to(q.dtype), final_state

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, d_ht=None):
        q, k, v, beta,mask , A, h, v_new, initial_state = ctx.saved_tensors
        BT = ctx.BT
        r = mask.shape[-1]
        start = time.time()
        w, u = fwd_recompute_w_u(k, v, beta, mask, A, BT)#跳过
        end = time.time()
        print('recompute_wu:',end-start)
        # checkpont_level=1, recomputation.
        if h is None:
            h, v_new = chunk_fwd_h_fn(k, w, u, BT, initial_state, None)
        #v_new b h r T V
        start = time.time() 
        dv = fwd_prepare_dv(q, k, do, r, BT)#qk do v_new#因此这个dv应该是一个w的shape finish
        end = time.time()
        print('pre:',end-start)
        #dv BHR T V
        
        start = time.time()
        dh, dv = chunk_bwd_dhu_fn(q, k, w, do, dv, BT)#new_dv dh #final for wyper dv
        end = time.time()
        print('chunk_bwd_dhu_fn:',end-start)
        
        start = time.time()
        dq, dk, dw = chunk_bwd_dqkw_fn(q, k, v_new, w, h, dv, do, dh, BT)
        end = time.time()
        print('chunk_bwd_dqkw_fn:',end-start)

        start = time.time()
        dk2, dv, dbeta = bwd_prepare_wy_repr(k, v, beta, mask, A, dw, dv, BT)#这一步误差较大
        dk.add_(dk2)
        end = time.time()
        print('bwd_prepare_wy_repr:',end-start)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dbeta.to(beta.dtype), None, None, None, None


def mask_chunk_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    mask: torch.Tensor,#use for mask org_tensor 
    BT: int,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
):
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "FusedChunkDeltaRuleFunction does not support float32. Please use bfloat16."
    o, final_state = ChunkDeltaRuleFunction.apply(q, k, v, beta,mask, BT, initial_state, output_final_state)
    return o, final_state


def naive(q,k,w,u,initial_state,BT,r):
    B,H,seq_len,dk = q.shape
    dv = u.shape[-1]
    NT = seq_len//BT
    state = torch.empty(B,H,NT,dk,dv,device=q.device,dtype=q.dtype)
    g = torch.zeros(B,H,NT,dk,dv,device=q.device,dtype=q.dtype)
    v_new = torch.empty_like(u)
    from einops import rearrange
    q,k,w,u,v_new = map(lambda x:rearrange(x,'b h (n t) d->b h n t d',n = NT),(q,k,w,u,v_new))
    q = q*(dk**-0.5)
    v_new = rearrange(v_new,'b h n (t r) d->b h n t r d',r = r)
    if initial_state is not None:
        state[:,:,0,:,:] = initial_state
    else:
        state[:,:,0,:,:] = 0
    for i in range(NT):
        ki = rearrange(k[:,:,i,:,:],'b h t (r d)->b h t r d',r = r)
        ui = rearrange(u[:,:,i,:,:],'b h (t r) d->b h t r d',r = r)
        wi = rearrange(w[:,:,i,:,:],'b h (t r) d->b h t r d',r = r)
        v_newi = ui - torch.einsum('b h t r d, b h d v-> b h t r v',wi,state[:,:,i,:,:])
        v_new[:,:,i,:,:,:] = v_newi#这里保存的结果是相等
        kui = torch.einsum('b h t r k,b h t r v-> b h r k v',ki,v_newi)
        g[:,:,i,:,:] = rearrange(kui,'b h r k v-> b h (r k) v')
        if i+1 < seq_len//BT:
            state[:,:,i+1,:,:] = state[:,:,i,:,:] + g[:,:,i,:,:]
    q_r = rearrange(q,'b h n t (r d)->b h n t r d',r = r)
    k_r = rearrange(k,'b h n t (r d)->b h n t r d',r = r)
    s_r = torch.einsum('b h n t r d,b h n l r d->b h n r t l',q_r,k_r)
    s_r = torch.tril(s_r,diagonal=0)#mask get bhnrtl
    o1 = torch.einsum('b h n r t l, b h n l r v-> b h n t r v',s_r,v_new)
    o1 = o1.sum(dim=-2)#bhntv
    o2 = torch.einsum('b h n t q,b h n q v-> b h n t v',q,state)#只看state 算的对不对
    o = o1 + o2
    return state,v_new,o,g


def delta_rule_recurrence(q, k, v, beta, mask):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    r = mask.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)
    q = q * (d_k ** -0.5)
    if beta.ndim < v.ndim:
        beta = beta[..., None]
    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].clone()
        beta_i = beta[:, :, i]
        _v = _v * beta_i
        # kkt = torch.einsum('b h d,b h v->b h d v',_k*beta_i,_k)
        kkt = torch.einsum('b h d,b h v->b h d v',_k,_k)
        kkt = rearrange(kkt,' b h (r d) (l v)-> b h r d l v',r= r,l=r)
        kkt = torch.einsum('b h r d l v,r l->b h r d l v',kkt,mask.to(kkt))
        kkt = rearrange(kkt,'b h r d l v-> b h (r d) (l v)')
        iplr = torch.eye(d_k).to(q)-kkt
        S = torch.einsum(' b h q k ,b h k v->b h q v',iplr,S.clone()) + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
    return o


if __name__ =="__main__":
    import sys
    import time
    # from einops import rearrange
    # sys.path.append('/mnt/jfzn/msj/flash-linear-attention-main/legacy/training/fla2-copy')
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

    B = 2
    H = 1
    L = 128
    DK = 256
    DV = 256
    q = (torch.randn(B, H, L, DK)).cuda().requires_grad_(True)
    k = (torch.randn(B, H, L, DK)).cuda()
    k = torch.nn.functional.normalize(k, dim=-1, p=2).requires_grad_(True)
    v = (torch.randn(B, H, L, DV)).cuda().requires_grad_(True)
    beta = torch.randn(B, H, L).cuda().sigmoid().requires_grad_(True)
    # mask = torch.tensor([[1,1,0,0],[1,1,1,0],[0,1,1,1],[0,0,1,1]],requires_grad=False).cuda().contiguous()
    mask = torch.tensor([[1,1,0,0],[1,1,1,0],[0,1,1,1],[0,0,1,1]],requires_grad=False).cuda().contiguous()

    start = time.time()
    o1 = delta_rule_recurrence(q,k,v,beta,mask)
    do = torch.randn(B, H, L, DV).cuda()
    o1.backward(do, retain_graph=True)
    q_grad, q.grad = q.grad, None
    k_grad, k.grad = k.grad, None
    v_grad, v.grad = v.grad, None
    beta_grad, beta.grad = beta.grad, None
    end = time.time()
    print(end-start)

    # start = time.time()
    # w, u, A = fwd_prepare_wy_repr(k, v,beta, mask, 64)
    o,f_state = mask_chunk_delta_rule(q, k, v, beta,mask,BT=32)
    o.backward(do,retain_graph=True)
    q_grad0, q.grad = q.grad, None
    k_grad0, k.grad = k.grad, None
    v_grad0, v.grad = v.grad, None
    beta_grad0, beta.grad = beta.grad, None
    # end = time.time()
    # print(end-start)
    print((o1-o).abs().max())
    print((q_grad-q_grad0).abs().max())
    print((k_grad-k_grad0).abs().max())#计算结果差距大 差距到1
    print((v_grad-v_grad0).abs().max())
    print((beta_grad-beta_grad0).abs().max())
    # print(beta_grad)
    # print(beta_grad0)
    print(k_grad)
    print(k_grad0)




