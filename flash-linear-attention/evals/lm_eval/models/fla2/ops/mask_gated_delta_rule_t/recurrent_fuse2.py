# -*- coding: utf-8 -*-
# Copyright (c) 2023, Yu Zhang, Songlin Yang

from typing import Tuple

import torch
import triton
import triton.language as tl

from fla.utils import contiguous
from einops import rearrange
# on-the-fly computation without materializing hidden statets into HBMs

@triton.jit(do_not_specialize=['T'])
def fused_recurrent_fwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    q,  # query [B, L, H, K]
    k,  # key 
    v,  # value .
    beta,  # beta
    g,#g 
    mask,#mask 
    o,  # output 
    h0,
    ht,  # final hidden state [B, H, K, V]
    scale,  # K ** -0.5
    B,  # batch size
    H,  # n_heads
    T,  # seq_len
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    r: tl.constexpr,  # r
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final stat
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_HEADWISE_BETA: tl.constexpr,  # whether beta is headwise vector or scalar
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos, eos = i_n * T, i_n * T + T


    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * H + i_h) * V + o_v


    if IS_HEADWISE_BETA:
        p_beta = beta + bos * H + i_h
    else:
        p_beta = beta + (bos * H + i_h) * V + o_v


    p_g = g + bos * H + i_h
    p_o = o + (bos * H + i_h) * V + o_v
    p_mask = mask + (bos * H + i_h) * r* r +tl.arange(0,r*r)

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    
    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)#BK
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)#BV
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)*scale#BK
        
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)

        b_g = tl.load(p_g).to(tl.float32)
        h *= tl.exp(b_g)
        b_k = tl.reshape(b_k,(r,BK//r))
        b_mask = tl.load(p_mask)
        b_mask = tl.reshape(b_mask,(r,r))
        b_w = b_k[None,:,:]*b_mask[:,:,None] #get r r BK//r
        b_w = tl.reshape(b_w,(r,BK))
        _v_minus = tl.sum(h[None,:]*b_w[:,:,None],axis=1)#r BV
        b_v_new = b_v[None,:] - _v_minus#r BV

        if IS_HEADWISE_BETA:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        
        b_v_new *= b_beta

        h_s = b_k[:, :,None] * b_v_new[:, None,:] #r BK//r BV
        h_s = (tl.reshape(h_s,(BK,BV)))
        h += h_s

        _o = tl.sum(h * b_q[:, None], 0)
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H*K
        p_k += H*K
        p_o += H*V
        p_v += H*V
        p_beta += H * (1 if IS_HEADWISE_BETA else V)
        p_mask += H*r*r
        p_g += H

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, h.to(p_ht.dtype.element_ty), mask=mask_h)




class FusedRecurrentFunction(torch.autograd.Function):
    @contiguous
    @staticmethod
    def forward(ctx, q, k, v, beta,g,mask ,scale=None, initial_state=None, output_final_state=False,use_qk_l2norm_in_kernel = False,):
        B, T,H, K, V = *q.shape, v.shape[-1]
        BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        r = mask.shape[-1]
        num_stages = 3
        num_warps = 1
        o = torch.empty_like(v)
        if output_final_state:
            final_state = q.new_empty(B, H, K, V)
        else:
            final_state = None

        grid = (NV, B * H)
        fused_recurrent_fwd_kernel[grid](
            q, k, v, beta,g,mask,o, initial_state, final_state,
            scale,
            B=B, H=H, T=T, K=K, V=V,r=r,
            BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None,
            IS_HEADWISE_BETA=(beta.ndim != v.ndim),###true
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return o, final_state

    @contiguous
    @staticmethod
    def backward(ctx, do, dht=None):
        raise NotImplementedError(
            "Do not use"
        )


def mask_fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    mask: torch.Tensor,
    scale: float = -1,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale == -1:
        scale = q.shape[-1] ** -0.5
    if initial_state is not None:
        initial_state = initial_state.detach()
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = FusedRecurrentFunction.apply(q, k, v, beta,g,mask,scale,initial_state, output_final_state,use_qk_l2norm_in_kernel)
    return o, final_state



def delta_rule_recurrence(q, k, v, beta, g, mask,initial_state=None,output_final_state=True):
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
        _v = v[:, :, i]
        beta_i = beta[:, :, i]
        _v = _v * beta_i
        kkt = torch.einsum('b h d,b h v->b h d v',_k*beta_i,_k)
        kkt = rearrange(kkt,' b h (r d) (l v)-> b h r d l v',r= r,l=r)
        kkt = torch.einsum('b h r d l v,b h r l->b h r d l v',kkt,mask[:,:,i,:,:].to(kkt))#16d参数，几乎可以忽略
        kkt = rearrange(kkt,'b h r d l v-> b h (r d) (l v)')
        iplr = torch.eye(d_k).to(q)-kkt
        iplr = torch.einsum(' b h q k ,b h->b h q k',iplr,g[:,:,i])
        S = torch.einsum(' b h q k ,b h k v->b h q v',iplr.float(),S) + _k.unsqueeze(-1).float() * _v.unsqueeze(-2).float()
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q.float(), S).to(k.dtype)
    return o,S


if __name__ =="__main__":
    import sys
    import time
    torch.set_default_dtype(torch.bfloat16)
    B = 8
    H = 8
    L = 227
    DK = 256
    DV = 256
    q = (torch.randn(B, H, L, DK)).cuda().requires_grad_(True)
    k = (torch.randn(B, H, L, DK)).cuda()
    k = torch.nn.functional.normalize(k, dim=-1, p=2).requires_grad_(True)
    v = (torch.randn(B, H, L, DV)).cuda().requires_grad_(True)
    beta = torch.randn(B, H, L).cuda().sigmoid().requires_grad_(True)
    r = 4 
    mask = torch.randn(B,H,L,r,r).cuda().requires_grad_(True)
    g = torch.nn.functional.logsigmoid(torch.randn(B, H, L).cuda()).requires_grad_(True)
    g_exp = torch.exp(g)
    o1,ss = delta_rule_recurrence(q,k,v,beta,g_exp,mask)

    q_t,k_t,v_t,beta_t,mask_t,g_t = map(lambda x:rearrange(x,'b h l ...-> b l h ...'),(q,k,v,beta,mask,g))
    o,f_state = mask_fused_recurrent_gated_delta_rule(q_t, k_t, v_t, beta_t, g_t,mask_t,output_final_state=True)#10s嘛 额
    o = rearrange(o,'b l h d->b h l d')

    print((o-o1).abs().max())
    print((o-o1))

