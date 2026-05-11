# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import rearrange

# from ...ops.generalized_delta_rule import fused_recurrent_dplr_delta_rule
from fla.ops.utils.op import exp
from fla.utils import input_guard
def delta_rule_recurrence(
    q,
    k,
    v,
    a,
    b,
    g,
    mask,
    scale: float = 1.0,
    initial_state=None,
    BT=None,
    output_final_state=True,
    verbose: bool = False,
):
    """与 ``testss.py`` 中实现一致（默认不逐步 ``print(S)``）。"""
    g_exp = torch.exp(g).float()
    B, H, L, DK = q.shape
    DV = v.shape[-1]
    r = mask.shape[-1]
    o = torch.zeros_like(v)
    if initial_state is None:
        S = torch.zeros(B, H, DK, DV, device=k.device, dtype=torch.float32)
    else:
        S = initial_state
    for i in range(L):
        _k = k[:, :, i].float()
        _q = q[:, :, i].float() * scale
        _v = v[:, :, i].float()
        _b = b[:, :, i].float()
        _a = a[:, :, i].float()
        _w = g_exp[:, :, i].float()

        abt = torch.einsum("b h d,b h v->b h d v", _b, _a)
        abt = rearrange(abt, "b h (r d) (l v)-> b h r d l v", r=r, l=r)
        abt = torch.einsum(
            "b h r d l v,b h r l->b h r d l v", abt, mask[:, :, i, :, :].float()
        )
        abt = rearrange(abt, "b h r d l v-> b h (r d) (l v)")
        dplr = torch.diag_embed(_w).to(q) + abt
        S = torch.einsum(
            "b h q k ,b h k v->b h q v", dplr.float(), S
        ) + _k.unsqueeze(-1).float() * _v.unsqueeze(-2).float()
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", _q.float(), S).to(k.dtype)
        if verbose:
            print(S)
    return o, S


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def mask_fused_recurrent_rwkv7_fwd_kernel(
    r,
    w,
    k,
    v,
    kk,
    a,
    mask,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B,
    H, 
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    R: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_r = r + (bos) * H*K + i_h * K + o_k
    p_w = w + (bos) * H*K + i_h * K + o_k
    p_k = k + (bos) * H*K + i_h * K + o_k
    p_v = v + (bos) * H*V + i_h * V + o_v
    p_a = a + (bos) * H*K + i_h * K + o_k
    p_kk = kk + (bos) * H*K + i_h * K + o_k
    p_mask = mask + (bos) * H*R*R + i_h * R*R + tl.arange(0,R*R)

    p_o = o + (bos) * H*V + i_h * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:,None] & mask_v[None,:] 
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:,None] * V + o_v[None,:]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_r = tl.load(p_r, mask=mask_k, other=0).to(tl.float32) * scale
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_kk = tl.load(p_kk, mask=mask_k, other=0).to(tl.float32)
        b_mask = tl.reshape(tl.load(p_mask),(R,R)).to(tl.float32)
        b_act_a = tl.reshape(-b_kk,(R,BK//R))
        b_b = tl.reshape(b_kk * b_a,(R,BK//R))

        b_a_mask = tl.reshape(b_act_a[None,:,:]*b_mask[:,:,None],(R,BK))##r BK
        tmp = tl.sum(b_h[None,:,:] * b_a_mask[:,:,None], axis=1)##r BV(需要补充一个负号)    
        h_add = tl.reshape(tmp[:, None, :] * b_b[:, :, None],(BK,BV)) ###R BK//R BV

        b_h = exp(b_w)[:,None] * b_h + h_add + b_k[:,None] * b_v[None,:]
        b_o = tl.sum(b_h * b_r[:,None], axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_r +=  H*K
        p_w +=  H*K
        p_k +=  H*K
        p_v +=  H*V
        p_a +=  H*K
        p_kk +=  H*K
        p_o +=  H*V
        p_mask +=  H*R*R

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + o_k[:,None] * V + o_v[None,:]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@input_guard
def mask_fused_recurrent_rwkv7_fwd(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    mask: torch.Tensor,
    a: torch.Tensor,
    scale: Optional[float] = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    R = mask.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V))
    h0 = initial_state
    if not output_final_state:
        ht = None
    else:
        ht = r.new_empty(N, H, K, V, dtype=torch.float32)
    o = torch.empty_like(v)
    def grid(meta): return (triton.cdiv(V, meta['BV']), N * H)
    mask_fused_recurrent_rwkv7_fwd_kernel[grid](
        r=r,
        w=w,
        k=k ,
        v=v,  
        kk=kk,
        a=a,
        mask=mask,
        o=o,
        h0=h0, 
        ht=ht,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        R=R,
        BV = BV,
        num_warps=1,
        num_stages=1
    )
    return o, ht


def mask_fused_mul_recurrent_rwkv7(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    mask: torch.Tensor,
    scale: Optional[float] = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    This function computes the recurrence S_t = S_t @ (I + a_t b_t^T) + v_t k_t^T in a recurrent manner.

    Args:
        r (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        w (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        v (torch.Tensor):
            a of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        kk (torch.Tensor):
            b of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        a (torch.Tensor):
            gk of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`. decay term in log space!
        mask (torch.Tensor):
            mask of shape `[B, T, H, r,r]` if `head_first=False` else `[B, H, T, r,r]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: 1.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        reverse (Optional[bool]):
            If `True`, process the state passing in reverse order. Default: `False`.
        cu_seqlens (Optional[torch.Tensor]):
            Cumulative sequence lengths of shape `[N + 1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.
    """
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        r, w, k, v, kk, a ,mask = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (r, w, k, v, kk, a,mask))
    if not head_first and r.shape[1] < r.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({r.shape[1]}) < num_heads ({r.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if r.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {r.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = r.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    o, final_state = mask_fused_recurrent_rwkv7_fwd(
        r=r,
        w=w,
        k=k,
        v=v,
        kk=kk,
        mask=mask,
        a=a,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
    if head_first:
        o = rearrange(o, 'b t h ... -> b h t ...')
    return o, final_state


if __name__ == "__main__":
    r = torch.randn(10, 2048, 4, 256).cuda()
    w = torch.randn(10, 2048, 4, 256).cuda()
    k = torch.randn(10, 2048, 4, 256).cuda()
    v = torch.randn(10, 2048, 4, 256).cuda()
    kk = torch.randn(10, 2048, 4, 256).cuda()
    a = torch.randn(10, 2048, 4, 256).cuda()
    mask = torch.randn(10, 2048, 4, 4, 4).cuda()
    recurrent_state = None
    import time
    for i in range(100):
        start_time = time.time()
        r_t = rearrange(r, 'b t h d -> b h t d')
        k_t = rearrange(k, 'b t h d -> b h t d')
        v_t = rearrange(v, 'b t h d -> b h t d')
        a_t = rearrange(-kk, 'b t h d -> b h t d')
        b_t = rearrange(kk*a, 'b t h d -> b h t d')
        mask_t = rearrange(mask, 'b t h r d -> b h t r d')
        gk_t = rearrange(w, 'b t h d -> b h t d')

        o1, recurrent_state1 = delta_rule_recurrence(
            q=r_t,
            k=k_t,
            v=v_t,
            a=a_t,
            b=b_t,
            g=gk_t,
            mask=mask_t,
            scale=1.,
            initial_state=recurrent_state,
            output_final_state=True
        )
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        start_time = time.time()
        o, final_state = mask_fused_mul_recurrent_rwkv7(r=r, w=w, k=k, v=v, kk=kk, a=a, mask=mask,
                scale=1., initial_state=recurrent_state, output_final_state=True)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        diff = (o-o1).abs().max()
        print(f"Max difference: {diff}")
        

