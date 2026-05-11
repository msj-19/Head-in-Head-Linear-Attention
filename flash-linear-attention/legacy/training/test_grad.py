import sys
from tarfile import DIRTYPE
import time
from fla.modules.l2norm import l2_norm as l2_norm_fn 
import torch
from einops import rearrange
torch.set_default_dtype(torch.bfloat16)
import os

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
    g = torch.exp(g.float())

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
        iplr = torch.eye(d_k).to(q)-kkt
        iplr = torch.einsum('b h q k, b h->b h q k',iplr,g[:,:,i])
        S = torch.einsum(' b h q k ,b h k v->b h q v',iplr.float(),S) + _k.unsqueeze(-1).float() * _v.unsqueeze(-2).float()
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q.float(), S).to(k.dtype)
    return o,S
   

import torch
import random
import numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
B = 2
H = 2
L = 1024
DK = 256
DV = 256
q = (torch.randn(B, H, L, DK)).cuda().requires_grad_(True)
k = (torch.randn(B, H, L, DK)).cuda()
k = torch.nn.functional.normalize(k, dim=-1, p=2).requires_grad_(True)
v = (torch.randn(B, H, L, DV)).cuda().requires_grad_(True)
do = (torch.randn(B, H, L, DV)).cuda()
beta = torch.randn(B, H, L).cuda().sigmoid().requires_grad_(True)
r=4
mask = ((torch.randn(B,H,L,r,r))).cuda()
mask = l2_norm_fn(mask.abs())
mask = (mask @ mask.transpose(-1, -2))
mask = mask.requires_grad_(True)
g = ((torch.nn.functional.logsigmoid(torch.randn(B, H, L).cuda()))).requires_grad_(True)


from fla2.ops.mask_gated_delta_rule_t.chunk import mask_gated_chunk_delta_rule
from fla4.ops.mask_gated_delta_rule_t.chunk import mask_gated_chunk_delta_rule as mask_gated_chunk_delta_rule3
q_t,k_t,v_t,beta_t,mask_t,g_t = map(lambda x:rearrange(x,'b h l ...-> b l h ...').contiguous(),(q,k,v,beta,mask,g))
o11,f1 = mask_gated_chunk_delta_rule3(q=q_t,k=k_t,v=v_t,g=g_t,beta=beta_t,mask=mask_t,BT=32,output_final_state=True)
o11 = rearrange(o11,'b l h d-> b h l d')
o11.backward(do, retain_graph=True) 
q_grad, q.grad = q.grad, None
k_grad, k.grad = k.grad, None 
v_grad, v.grad = v.grad, None
beta_grad, beta.grad = beta.grad, None
g_grad, g.grad = g.grad, None
mask_grad, mask.grad = mask.grad, None

o22,f2 = delta_rule_recurrence(q=q,k=k,v=v,beta=beta,g=g,mask=mask,output_final_state=True)
o22.backward(do, retain_graph=True)
q_grad1, q.grad = q.grad, None
k_grad1, k.grad = k.grad, None
v_grad1, v.grad = v.grad, None
beta_grad1, beta.grad = beta.grad, None
g_grad1, g.grad = g.grad, None
mask_grad1, mask.grad = mask.grad, None

o1_list = [o11,f1,q_grad,k_grad,v_grad,beta_grad,g_grad,mask_grad]
o2_list = [o22,f2,q_grad1,k_grad1,v_grad1,beta_grad1,g_grad1,mask_grad1]
name_list = ['o11', 'f1','q_grad','k_grad','v_grad','beta_grad','g_grad','mask_grad']

for i in range(len(o1_list)):
    o1 = o1_list[i]
    o2 = o2_list[i]
    if o1 is not None and o2 is not None:
        diff = ((o1 - o2)).abs()
        max_val, flat_index = diff.max(), diff.argmax()
        index = torch.unravel_index(flat_index, diff.shape)
        print(name_list[i])
        print(f"最大差值: {max_val.item()}")
        print(f'mean:',diff.mean())
        print(f"坐标: {index}")
        print(f"recurrent 在该坐标的值: {o1[index].item()}")
        print(f"triton 在该坐标的值: {o2[index].item()}")
        if name_list[i] == 'g_grad':
            print(diff)
            print(o1)


def naive_recurrent_rwkv7_2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,  # Dynamic learning rate modulator
    b: torch.Tensor,  # State update modulator
    mask: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
):
    """
    Naive recurrent implementation of RWKV-7 (Goose) attention mechanism.

    Args:
        q, k, v: Query, Key, and Value tensors
        w: Time decay weights
        a: Dynamic learning rate modulator, influences the in-context learning rate
        b: State update modulator, directly participates in state update calculation
        scale: Scaling factor for attention scores
        initial_state: Initial state for the recurrent computation
        output_final_state: Whether to output the final state

    Returns:
        Attention output and optionally the final state
    """
    torch_dtype = q.dtype if q.dtype in [torch.float64, torch.float] else torch.float
    orig_dtype = q.dtype
    B, H, L, N, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[-1]
    q, k, v, w, a, b = (x.to(dtype=torch_dtype) for x in (q, k, v, w, a, b))
    # q, k, v, a, b, w,
    # shape: (B, H, L, D), (B, H, L, D), (B, H, T, V), (B, H, L, D), (B, H, L, D), (B, H, L, D)
    state = torch.zeros(B, H, N, V, dtype=torch_dtype, device=q.device)
    o = torch.zeros_like(v)
    r = mask.shape[-1]
    if scale == -1.0:
        scale = N ** -0.5

    if initial_state is not None:
        state += initial_state.to(dtype=torch_dtype)

    for t in range(L):
        for bi in range(B):
            for hi in range(H):
                q_t = q[bi, hi, t] * scale
                k_t = k[bi, hi, t]
                v_t = v[bi, hi, t]
                a_t = a[bi, hi, t]
                b_t = b[bi, hi, t]
                m_t = mask[bi, hi, t]
                w_t = torch.exp((w[bi, hi, t]))

                
                ab =torch.einsum('k,v->kv',b_t,a_t)
                abmask = rearrange(ab,'(r k) (c v)->r k c v',r=r,c=r)*m_t[:,None,:,None]
                DPLR = torch.diag(w_t) +  rearrange(abmask,'r k c v->(r k) (c v)')

                state[bi, hi] = DPLR @ state[bi, hi] + k_t[:, None] * v_t[None, :]
                y = (state[bi, hi] * q_t [:,None]).sum(dim=0)
                o[bi, hi, t] = y
    ht = state if output_final_state else None
    return o.to(orig_dtype), ht



from fla3.ops.mask_generalized_delta_rule.dplr.chunk import mask_chunk_dplr_delta_rule
from fla3.ops.mask_rwkv7 import mask_fused_mul_recurrent_rwkv7

B, H, T, D = 16,16,16,256
device = torch.device("cuda")
dtype = torch.bfloat16

q = (torch.randn(B,T,H, D, device=device)).to(dtype=dtype).requires_grad_(True)#r
k = (torch.randn(B, T, H, D, device=device)).to(dtype=dtype).requires_grad_(True)#k
v = (torch.randn(B, T, H, D, device=device)).to(dtype=dtype).requires_grad_(True)#v
gk = (torch.nn.functional.logsigmoid(torch.randn(B, T, H, D, device=device))).to(dtype=dtype).requires_grad_(True)#w
kk = (torch.randn(B, T, H, D, device=device))##kk
kk = torch.nn.functional.normalize(kk, dim=-1).to(dtype=dtype)
a = (-kk.clone()).requires_grad_(True)  # -kk ##a
do = 0.1*(torch.randn(B, T, H, D, device=device)).to(dtype=dtype)
a_scale = (torch.randn(B, T, H, D, device=device)).to(dtype=dtype)#####a_scale
b = (kk * a_scale).requires_grad_(True)  # kk*a


B,T,H,D = q.shape
scale = 1
initial_state = None
# scale = D ** -0.5
output_final_state = True
head_first = False
mask = ((torch.randn(B, T, H, 4, 4, device=device).to(dtype=dtype))).requires_grad_(True)
target_matrix = l2_norm_fn(mask.abs())
target_matrix = (target_matrix @ target_matrix.transpose(-1, -2))


# q_t,k_t,v_t,a_t,b_t,gk_t,mask_t = map(lambda x:rearrange(x,'b t h ...-> b h t ...'),(q,k,v,a,b,gk,target_matrix))
# o1,final_state1 = delta_rule_recurrence(q=q_t, k=k_t, v=v_t, a=a_t, b=b_t, g=gk_t, mask=mask_t, scale=scale, initial_state=initial_state, output_final_state=output_final_state)
# o1 = rearrange(o1,'b h t ...-> b t h ...')
o1,final_state1 = mask_fused_mul_recurrent_rwkv7(r=q,w=gk,k=k,v=v,a=a_scale,kk=kk,mask=target_matrix,scale=scale,initial_state=initial_state,output_final_state=output_final_state,head_first=head_first)
# o1.backward(do, retain_graph=True)
# q_grad, q.grad = q.grad, None
# k_grad, k.grad = k.grad, None
# v_grad, v.grad = v.grad, None
# gk_grad, gk.grad = gk.grad, None
# a_grad, a.grad = a.grad, None
# b_grad, b.grad = b.grad, None
# mask_grad, mask.grad = mask.grad, None

o2, final_state2 = mask_chunk_dplr_delta_rule(q=q, k=k, v=v, a=a, b=b, gk=gk, mask=target_matrix, scale=scale, initial_state=initial_state, output_final_state=output_final_state, head_first=head_first)
# o2.backward(do, retain_graph=True)
# q_grad1, q.grad = q.grad, None
# k_grad1, k.grad = k.grad, None
# v_grad1, v.grad = v.grad, None
# gk_grad1, gk.grad = gk.grad, None
# a_grad1, a.grad = a.grad, None
# b_grad1, b.grad = b.grad, None
# mask_grad1, mask.grad = mask.grad, None


o1_list = [o1,final_state1]#,q_grad,k_grad,v_grad,gk_grad,a_grad,b_grad,mask_grad]
o2_list = [o2,final_state2]#,q_grad1,k_grad1,v_grad1,gk_grad1,a_grad1,b_grad1,mask_grad1]
for i in range(len(o1_list)):
    print(f"第{i}个:")
    o1 = o1_list[i]
    o2 = o2_list[i]
    if o1 is not None and o2 is not None:
        diff = (o1-o2).abs()
        mean_diff = diff.mean()
        print(f"平均差值: {mean_diff.item()}")
        max_o = max((o1.abs().max()),(o2.abs().max()))
        max_val, flat_index = diff.max(), diff.argmax()
        index = torch.unravel_index(flat_index, diff.shape)
        print(f"最大差值: {max_val.item()}")
        print(f"坐标: {index}")
        print(f"recurrent 在该坐标的值: {o1[index].item()}")
        print(f"triton 在该坐标的值: {o2[index].item()}")
        print(f"最大差值/最大值: {max_val.item()/max_o.item()}")
