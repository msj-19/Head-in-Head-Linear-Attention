# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from ..modules import FusedRMSNormSwishGate, RMSNorm
from fla.modules import ShortConvolution
import torch.nn.init  as init
import math
from ..modules.l2norm import l2_norm_fn 
from einops import rearrange
from fla.models.utils import Cache
from transformers.processing_utils import Unpack
import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
# from ..modules import RotaryEmbedding
# from ..ops.gla import chunk_gla,fused_recurrent_gla


def simple_norm(x):
    return (F.normalize(x, dim=-1) * x.shape[-1] ** 0.5).to(x)


# @torch.jit.script
def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


# @torch.jit.script
def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


# @torch.jit.script
def elu_norm(x):
    dtype = x.dtype
    x = F.elu(x, 1., False) + 1.
    return (x / x.sum(-1, keepdim=True)).to(dtype)


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss, 
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


####单层模型测速
from ..ops.mask_gated_delta_rule_t import mask_gated_chunk_delta_rule
from ..ops.mask_gated_delta_rule_t.recurrent_fuse2 import mask_fused_recurrent_gated_delta_rule
class mask_gdn(nn.Module):
    def __init__(
        self,
        d_model: int = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        mode: str = 'chunk',
        chunk_size: int = 32,
        use_beta: bool = True,
        use_gate: bool = False,
        use_output_norm: bool = True,
        use_elu: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        qk_activation: str = 'silu',
        qk_norm: str = 'l2',
        norm_first: bool = False,
        norm_eps: float = 1e-6,
        ratio :int = 4,
        topk : int = 1 ,
        **kwargs
    ) :
        super().__init__()

        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        assert self.qk_activation in ['silu', 'relu', 'elu', 'identity']
        assert self.qk_norm in ['l2', 'sum']

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.use_gate = use_gate
        self.use_output_norm = use_output_norm
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.norm_first = norm_first
        self.layer_idx = layer_idx
        self.top_k = topk
        self.silu = nn.SiLU()
        assert mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"
        if norm_first:
            self.norm = RMSNorm(self.hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True


        self.use_beta = use_beta
        self.use_elu = use_elu
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(self.key_dim,
                                             conv_size,
                                             activation='silu' if qk_activation == 'silu' else None)
            self.k_conv1d = ShortConvolution(self.key_dim,
                                             conv_size,
                                             activation='silu' if qk_activation == 'silu' else None)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        r = self.r = ratio

        self.mask = nn.Parameter(torch.empty([self.num_heads,r,r],dtype=self.o_proj.weight.dtype),requires_grad=True)##fixed mask
        self.mask_requiregrad = True
        init.kaiming_uniform_(self.mask, a=math.sqrt(5))
        self.apply(self._initialize_weights)

        # self.mask = nn.Linear(hidden_size, (self.num_heads*r*r), bias=False) #nn.Parameter(torch.empty([r,r-1],dtype=self.o_proj.weight.dtype),requires_grad=True)
        # self.mask_requiregrad = True 
        # print('mask_gdn_learn_mask_r'+str(self.r)+'_hrr_byt')
        # assert self.head_qk_dim % r == 0 
        # self.apply(self._initialize_weights)


    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True
    
    def delta_rule_recurrence(self,q, k, v, beta, g, mask,initial_state=None,output_final_state=True):
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
        g = torch.exp(g).float()

        for i in range(l):
            _k = k[:, :, i].float()
            _q = q[:, :, i].float()
            _v = v[:, :, i].float()
            beta_i = beta[:, :, i].float()
            _v = _v * beta_i
            kkt = torch.einsum('b h d,b h v->b h d v',_k*beta_i,_k)
            kkt = rearrange(kkt,' b h (r d) (l v)-> b h r d l v',r= r,l=r)
            kkt = torch.einsum('b h r d l v,b h r l->b h r d l v',kkt,mask[:,:,i,:,:].to(kkt))
            kkt = rearrange(kkt,'b h r d l v-> b h (r d) (l v)')
            iplr = torch.eye(d_k).float()-kkt
            iplr = torch.einsum('b h q k, b h->b h q k',iplr,g[:,:,i]).float()
            S = torch.einsum(' b h q k ,b h k v->b h q v',iplr.float(),S) + _k.unsqueeze(-1).float() * _v.unsqueeze(-2).float()
            o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q.float(), S).to(k.dtype)
        return o,S
        
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # change to inference mode.
        # mode = self.mode
        mode = 'recurrent' if hidden_states.shape[1] < 16 else 'chunk'
        batch,q_len,d = hidden_states.shape
        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        cu_seqlens = kwargs.get('cu_seqlens', None)
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
            offset = past_key_values.get_seq_length()
        if self.use_short_conv: 
            conv_state_q , conv_state_k , conv_state_v = None,None,None
            if last_state is not None:
                conv_state_q , conv_state_k , conv_state_v  = last_state['conv_state']
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            q,conv_state_q = self.q_conv1d(q, cache= conv_state_q,output_final_state = use_cache,cu_seqlens =cu_seqlens)
            k,conv_state_k = self.k_conv1d(k, cache= conv_state_k,output_final_state = use_cache,cu_seqlens =cu_seqlens)
            v,conv_state_v = self.v_conv1d(v, cache= conv_state_v,output_final_state = use_cache,cu_seqlens =cu_seqlens)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        if mode == 'chunk':
            q, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (q, k, v))
            if self.qk_activation != 'silu':
                if self.qk_activation == 'relu':
                    q, k = q.relu(), k.relu()
                elif self.qk_activation == 'elu':
                    q, k = elu_p1(q), elu_p1(k)
                elif self.qk_activation == 'identity':
                    pass
                else:
                    raise NotImplementedError
            if self.qk_norm is not None:
                if self.qk_norm == 'l2':
                    q = l2_norm_fn(q)
                    k = l2_norm_fn(k)
                elif self.qk_norm == 'sum':
                    q = sum_norm(q).to(v)
                    k = sum_norm(k).to(v)
            recurrent_state_sf = None
            if last_state is not None:
                recurrent_state_sf = last_state['recurrent_state']

            beta = rearrange(self.b_proj(hidden_states), 'b l h -> b h l').sigmoid()
            g = rearrange(-self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias), 'b l h -> b h l')

            # r = self.r
            # target_matrix = self.mask.abs()
            # target_matrix = l2_norm_fn(target_matrix)
            # target_matrix = target_matrix@target_matrix.transpose(-1,-2)
            # target_matrix = target_matrix.unsqueeze(1).unsqueeze(0).expand(batch,self.num_heads,q_len,self.r,self.r)###head first BHLD
            # target_matrix = self.target_matrix.unsqueeze(1).unsqueeze(0).expand(batch,self.num_heads,q_len,self.r,self.r)

            r = self.r ###nn.linear
            target_matrix = self.mask(hidden_states).abs()
            target_matrix = rearrange(target_matrix,'b l (h r c)->b h l r c',r=r,h=self.num_heads)#bhlrr
            target_matrix = l2_norm_fn(target_matrix)
            target_matrix = target_matrix@target_matrix.transpose(-1,-2)

            if r==8:
                BT = 16 
            else:
                BT = 32 
            o,recurrent_state_sf = mask_gated_chunk_delta_rule(q.contiguous(),k.contiguous(),v.contiguous(),beta.contiguous(),g.contiguous(),target_matrix.contiguous(),initial_state=recurrent_state_sf,BT=BT,output_final_state=True)
            o = rearrange(o,'b h l d-> b l h d')

        elif mode == 'recurrent':
            q, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b l h d', h=self.num_heads), (q, k, v))
            if self.qk_activation != 'silu':
                if self.qk_activation == 'relu':
                    q, k = q.relu(), k.relu()
                elif self.qk_activation == 'elu':
                    q, k = elu_p1(q), elu_p1(k)
                elif self.qk_activation == 'identity':
                    pass
                else:
                    raise NotImplementedError
            recurrent_state_sf = None
            if last_state is not None:
                (recurrent_state_sf) = last_state['recurrent_state']

            beta = (self.b_proj(hidden_states)).sigmoid()###blh
            g = (-self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias))
            r = self.r
            target_matrix = self.mask(hidden_states).abs()
            target_matrix = rearrange(target_matrix,'b l (h r c)->b l h r c',r=r,h=self.num_heads)#bhlrr
            target_matrix = l2_norm_fn(target_matrix)
            target_matrix = target_matrix@target_matrix.transpose(-1,-2)
            # r = self.r
            # target_matrix = self.mask.abs()
            # target_matrix = l2_norm_fn(target_matrix)
            # target_matrix = target_matrix@target_matrix.transpose(-1,-2)
            # target_matrix = target_matrix.unsqueeze(0).unsqueeze(0).expand(batch,q_len,self.num_heads,self.r,self.r)
            o,recurrent_state_sf = mask_fused_recurrent_gated_delta_rule(q,k,v,beta,g,target_matrix.contiguous(),initial_state=recurrent_state_sf,output_final_state=True,
                                                                         use_qk_l2norm_in_kernel=True)
        else:
            raise NotImplementedError

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=(recurrent_state_sf),
                conv_state=(conv_state_q,conv_state_k,conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len
            )
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), 'b l (h d) -> b l h d', h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b l h d -> b l (h d)')
        o = self.o_proj(o)
        return o, None, past_key_values,None
