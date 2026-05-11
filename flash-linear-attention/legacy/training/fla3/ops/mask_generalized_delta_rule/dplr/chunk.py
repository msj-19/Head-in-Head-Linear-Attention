# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional

import torch
import triton
from einops import rearrange

from ....ops.mask_generalized_delta_rule.dplr.chunk_A_bwd import mask_chunk_dplr_bwd_dqk_intra
from ....ops.mask_generalized_delta_rule.dplr.chunk_A_fwd import mask_chunk_dplr_fwd_intra
from ....ops.mask_generalized_delta_rule.dplr.chunk_h_bwd import mask_chunk_dplr_bwd_dhu
from ....ops.mask_generalized_delta_rule.dplr.chunk_h_fwd import mask_chunk_dplr_fwd_h
from ....ops.mask_generalized_delta_rule.dplr.chunk_o_bwd import mask_chunk_dplr_bwd_dAu, mask_chunk_dplr_bwd_dv, mask_chunk_dplr_bwd_o
from ....ops.mask_generalized_delta_rule.dplr.chunk_o_fwd import mask_chunk_dplr_fwd_o
from ....ops.mask_generalized_delta_rule.dplr.wy_fast_bwd import mask_chunk_dplr_bwd_wy
from ....ops.mask_generalized_delta_rule.dplr.wy_fast_fwd import mask_prepare_wy_repr_fwd
from ....ops.rwkv6.chunk import chunk_rwkv6_fwd_cumsum
from ....utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

@input_guard
def mask_chunk_dplr_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
    gk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64
):
    T = q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    assert T % BT == 0
    gi, ge = chunk_rwkv6_fwd_cumsum(gk, BT, cu_seqlens=cu_seqlens)

    # print(mask)
    A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = mask_chunk_dplr_fwd_intra(
        q=q,
        k=k,
        a=a,
        b=b,
        mask=mask,
        gi=gi,
        ge=ge,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
    )
    del ge

    w, u, _ = mask_prepare_wy_repr_fwd(
        ag=ag,
        A_ab=A_ab,
        A_ak=A_ak,
        v=v,
        mask=mask,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )
    del A_ab, A_ak

    h, v_new, final_state = mask_chunk_dplr_fwd_h(
        kg=kg,
        bg=bg,
        v=v,
        w=w,
        u=u,
        gk=gi,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )
    del u, kg, bg, gi
    o = mask_chunk_dplr_fwd_o(
        qg=qg,
        v=v,
        v_new=v_new,
        A_qk=A_qk,
        A_qb=A_qb,
        h=h,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )
    del v_new, h, A_qk, A_qb
    return o, final_state


class MaskChunkDPLRDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        gk: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ):
        chunk_size = 16
        o, final_state = mask_chunk_dplr_fwd(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            gk=gk,
            mask=mask,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size
        )
        ctx.save_for_backward(q, k, v, a, b, gk, mask, initial_state)
        ctx.cu_seqlens = cu_seqlens
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        q, k, v, a, b, gk, mask, initial_state = ctx.saved_tensors
        BT = ctx.chunk_size
        cu_seqlens = ctx.cu_seqlens
        scale = ctx.scale
        gi, ge = chunk_rwkv6_fwd_cumsum(gk, BT, cu_seqlens=cu_seqlens)
        A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = mask_chunk_dplr_fwd_intra(
            q=q,
            k=k,
            a=a,
            b=b,
            mask=mask,
            gi=gi,
            ge=ge,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
        )
        w, u, A_ab_inv = mask_prepare_wy_repr_fwd(
            ag=ag,
            A_ab=A_ab,
            A_ak=A_ak,
            v=v,
            mask=mask,
            cu_seqlens=cu_seqlens,
            chunk_size=BT
        )
        del A_ab
        h, v_new, _ = mask_chunk_dplr_fwd_h(
            kg=kg,
            bg=bg,
            v=v,
            w=w,
            u=u,
            gk=gi,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            chunk_size=BT
        )
        del u

        dv_new_intra, dA_qk, dA_qb = mask_chunk_dplr_bwd_dAu(
            v=v,
            v_new=v_new,
            do=do,
            A_qb=A_qb,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=BT
        )

        dh, dh0, dv_new = mask_chunk_dplr_bwd_dhu(
            qg=qg,
            bg=bg,
            w=w,
            gk=gi,
            h0=initial_state,
            dht=dht,
            do=do,
            dv=dv_new_intra,
            cu_seqlens=cu_seqlens,
            chunk_size=BT
        )

        dv = mask_chunk_dplr_bwd_dv(
            A_qk=A_qk,
            kg=kg,
            do=do,
            dh=dh,
            cu_seqlens=cu_seqlens,
            chunk_size=BT
        )
        del A_qk
        dqg, dkg, dw, dbg, dgk_last = mask_chunk_dplr_bwd_o(
            k=kg,
            b=bg,
            v=v,
            v_new=v_new,
            do=do,
            h=h,
            dh=dh,
            dv=dv_new,
            w=w,
            gk=gi,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            scale=scale,
        )



        del v_new
        dA_ab, dA_ak, dv, dag, dmask = mask_chunk_dplr_bwd_wy(
            A_ab_inv=A_ab_inv,
            A_ak=A_ak,
            v=v,
            mask=mask,
            ag=ag,
            dw=dw,
            du=dv_new,
            dv0=dv,
            cu_seqlens=cu_seqlens,
            chunk_size=BT
        )
        del A_ak
        dq, dk, da, db, dgk, dmask2 = mask_chunk_dplr_bwd_dqk_intra(
            q=q,
            k=k,
            a=a,
            b=b,
            gi=gi,
            ge=ge,
            mask=mask,
            dAqk=dA_qk,
            dAqb=dA_qb,
            dAak=dA_ak,
            dAab=dA_ab,
            dgk_last=dgk_last,
            dqg=dqg,
            dkg=dkg,
            dag=dag,
            dbg=dbg,
            chunk_size=BT,
            scale=scale,
            cu_seqlens=cu_seqlens,
        )
        dmask.add_(dmask2)
        return dq.to(q), dk.to(k), dv.to(v), da.to(a), db.to(b), dgk.to(gk), dmask.to(mask), None, dh0, None, None


@torch.compiler.disable
def mask_chunk_dplr_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    mask: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        a (torch.Tensor):
            activations of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        b (torch.Tensor):
            betas of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        gk (torch.Tensor):
            gk of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`. decay term in log space!
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v, a, b, gk = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (q, k, v, a, b, gk))
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if q.dtype == torch.float32:
        raise DeprecationWarning(
            """ChunkDeltaRuleFunction does not support float32. Please use bfloat16.
            If you want to use float32, please solve the issue by yourself."""
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
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    o, final_state = MaskChunkDPLRDeltaRuleFunction.apply(
        q,
        k,
        v,
        a,
        b,
        gk,
        mask,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
    )
    if head_first:
        o = rearrange(o, 'b t h ... -> b h t ...')
    return o, final_state


if __name__ == "__main__":
    q = torch.randn(1, 128, 2, 256)
    k = torch.randn(1, 128, 2, 256)
    v = torch.randn(1, 128, 2, 256)
    a = torch.randn(1, 128, 2, 256)
    b = torch.randn(1, 128, 2, 256)
    gk = torch.randn(1, 128, 2, 256)
    mask = torch.randn(1, 128, 2, 4, 4)
    scale = 1.0
    initial_state = None
    output_final_state = False
    cu_seqlens = None
    head_first = False

    o, final_state = mask_chunk_dplr_delta_rule(q, k, v, a, b, gk, mask, scale, initial_state, output_final_state, cu_seqlens, head_first)
    print(o.shape)
    print(final_state.shape)
