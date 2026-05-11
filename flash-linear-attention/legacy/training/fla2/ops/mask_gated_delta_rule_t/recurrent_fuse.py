# -*- coding: utf-8 -*-
# Copyright (c) 2023, Yu Zhang, Songlin Yang

from typing import Tuple

import torch
import triton
import triton.language as tl

from fla.utils import contiguous
from fla.ops.utils.op import exp
from einops import rearrange
# on-the-fly computation without materializing hidden statets into HBMs


#大致use this as speed
@triton.jit
def fused_recurrent_fwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    q,  # query [B, H, L, K]
    k,  # key [B, H, L, V]
    v,  # value [B, H, L, V].
    beta,  # beta [B, H, L]
    g, # g [B H L]
    mask, #mask [B H r r]
    o,  # output [B, H, L, V]
    h0,
    ht,  # final hidden state [B, H, K, V]
    s_qk_h,  # stride size: L * K
    s_vo_h,  # stride size: L * V
    scale,  # K ** -0.5
    B,  # batch size
    H,  # n_heads
    T,  # seq_len
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    r: tl.constexpr,
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
):

    # indices
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)

    p_beta = beta + i_bh * T
    p_g = g + i_bh * T
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_mask = tl.make_block_ptr(mask + i_bh * T * r * r, (r, r), (r, 1), (0, 0), (r, r), (1, 0))

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bk[:, None] & mask_bv[None, :]

    h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K*V + (i_k * BK + tl.arange(0, BK))[:, None] * V + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_mask = tl.load(p_mask).to(tl.float32)#r * r
        b_beta = tl.load(p_beta).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)

        h *= exp(b_g)
        #需要先处理b_k 
        b_k_reshape = tl.reshape(b_k,(r,BK//r))
        b_kk = b_k_reshape[None,:,:]*b_mask[:,:,None] #get r r BK//r
        b_kk = tl.reshape(b_kk,(r,BK))
        _v_minus = tl.sum(h[None,:,:] * b_kk[:,:,None],axis=1) #r BV
        b_vnew = b_v[None,:] - _v_minus #r BV
        b_vnew *= b_beta #r BV

        # bk r BK None  bv r None BV
        h_add = b_vnew[:,None,:] * b_k_reshape[:,:, None]
        h += tl.reshape(h_add,(BK,BV))

        _o = h * b_q[:, None]
        _o = tl.sum(_o, axis=0)
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_bv)

        p_q += K
        p_k += K
        p_o += V
        p_v += V
        p_g += 1
        p_beta += 1 
        p_mask = tl.advance(p_mask, (r, 0))

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK))[:, None] * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_ht, h.to(p_ht.dtype.element_ty), mask=mask_kv)


class FusedRecurrentFunction(torch.autograd.Function):

    @contiguous
    @staticmethod
    def forward(ctx, q, k, v, beta, g, mask, scale=None, initial_state=None, output_final_state=False):
        B, H, T, K, V = *q.shape, v.shape[-1]
        r = mask.shape[-1]
        BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)#8???
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 1
        assert NK == 1, "NK > 1 is not supported yet"
        o = q.new_empty(NK, B, H, T, V)

        if output_final_state:
            final_state = q.new_empty(B, H, K, V)
        else:
            final_state = None
        grid = (NV, NK, B * H)
        fused_recurrent_fwd_kernel[grid](
            q, k, v, beta,g,mask, o, initial_state, final_state,
            q.stride(1),
            v.stride(1),
            scale,
            B=B, H=H, T=T, K=K, V=V,r=r,
            BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        o = o.sum(0)
        return o, final_state

    @contiguous
    @staticmethod
    def backward(ctx, do, dht=None):
        raise NotImplementedError(
            "Backward pass for mask_gdn_recurrent is not implemented. "
        )


def mask_fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor ,
    g: torch.Tensor , 
    mask: torch.Tensor ,
    scale: float = -1,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale == -1:
        scale = q.shape[-1] ** -0.5
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = FusedRecurrentFunction.apply(q, k, v, beta, g, mask, scale, initial_state, output_final_state)
    return o, final_state

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

if __name__ =="__main__":
    import sys
    import time
    from fla.modules.l2norm import l2_norm as l2_norm_fn 
    
    torch.set_default_dtype(torch.bfloat16)
    B = 8
    H = 8
    L = 2048
    DK = 256
    DV = 256
    q = (torch.randn(B, H, L, DK)).cuda().requires_grad_(True)
    k = (torch.randn(B, H, L, DK)).cuda()
    k = torch.nn.functional.normalize(k, dim=-1, p=2).requires_grad_(True)
    v = (torch.randn(B, H, L, DV)).cuda().requires_grad_(True)
    r = 4 
    mask = torch.randn(r,r).cuda().requires_grad_(True)
    target_matrix = torch.softmax(mask,dim=-1)#h r c
    eye_mask = torch.eye(r, dtype=torch.bool, device=target_matrix.device).unsqueeze(0)
    target_matrix = torch.where(eye_mask, torch.tensor(1.0, device=target_matrix.device), target_matrix)
    target_matrix = target_matrix.unsqueeze(1).unsqueeze(0).expand(B,H,L,r,r)
    beta = torch.randn(B, H, L).cuda().sigmoid().requires_grad_(True)#*0.0+1.0
    g = torch.nn.functional.logsigmoid(torch.randn(B, H, L).cuda()).requires_grad_(True)#*0
    for i in range(10):
        with torch.no_grad():
            import time
            start = time.time()
            o11,h_11,ss = delta_rule_recurrence(q,k,v,beta,g,target_matrix)
            end = time.time()
            print(start-end)
            start = time.time()
            o22,f_state = mask_fused_recurrent_gated_delta_rule(q, k, v, beta, g,target_matrix,output_final_state=True)#10s嘛 额
            end = time.time()
            print(start-end)
    # print((o11-o22).abs().max())
    # print((o11-o22).shape)


