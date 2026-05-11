import torch
import time
import sys
sys.path.append('/mnt/jfzn/msj/flash-linear-attention/evals/lm_eval/models')
from fla2.layers import mask_gdn
from fla3.layers import GatedDeltaNet#,Attention,GatedLinearAttention
from fla.models.utils import Cache
import copy
def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}Yi{suffix}'
a_layer = mask_gdn(hidden_size=1024,num_heads=8,ratio=4,mode='chunk',chunk_size=32,layer_idx=0)
print(a_layer)
print(f"Total params: {sum(p.numel() for p in a_layer.parameters()):,}")
a_layer = a_layer.cuda()
# a_layer.eval()

# 构造输入（按你的真实 shape 改）
T_list = [4]  #[0.25,0.5,1,2,4,8,16,32,64,128,256,512]
T_list = [int(i*1024) for i in T_list]
# T_list = [256,512,1024,2048,4096,8192,16384]

#####train_speed_test
for T in T_list:
    B, D = 4, 1024
    x = torch.randn(B, T, D, device="cuda", requires_grad=True)
    # warmup（非常重要）
    for _ in range(10):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = a_layer(x)
            loss = y[0].sum()
            loss.backward()
            a_layer.zero_grad(set_to_none=True)
            x.grad = None

    torch.cuda.synchronize()

    # 正式测速
    iters = 50
    t0 = time.time()
    for _ in range(iters):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = a_layer(x)
            loss = y[0].sum()
            loss.backward()
            a_layer.zero_grad(set_to_none=True)
            x.grad = None
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[a_layer] avg time per iter: {(t1 - t0) * 1000 / iters:.3f} ms")
    print('current t',T)
    print(f"Max memory used: {sizeof_fmt(torch.cuda.max_memory_allocated())}")


####train
# mask_gdn_learn_mask_r4_hrr_byt
# [a_layer] avg time per iter: 5.510 ms
# current t 256
# [a_layer] avg time per iter: 5.851 ms
# current t 521
# [a_layer] avg time per iter: 8.661 ms
# current t 1024
# [a_layer] avg time per iter: 16.258 ms
# current t 2048
# [a_layer] avg time per iter: 31.509 ms
# current t 4096
# [a_layer] avg time per iter: 62.171 ms
# current t 8192
# [a_layer] avg time per iter: 124.171 ms
# current t 8192

#####
# mask_gdn_learn_mask_r2_hrr_byt
# [a_layer] avg time per iter: 5.349 ms
# current t 256
# [a_layer] avg time per iter: 5.584 ms
# current t 521
# [a_layer] avg time per iter: 5.370 ms
# current t 1024
# [a_layer] avg time per iter: 8.791 ms
# current t 2048
# [a_layer] avg time per iter: 16.461 ms
# current t 4096
# [a_layer] avg time per iter: 31.802 ms
# current t 8192
# [a_layer] avg time per iter: 62.802 ms
# current t 16384

#base_line gdn
# Total params: 5,271,696
# [a_layer] avg time per iter: 4.402 ms
# current t 256
# [a_layer] avg time per iter: 4.419 ms
# current t 521
# [a_layer] avg time per iter: 4.563 ms
# current t 1024
# [a_layer] avg time per iter: 4.557 ms
# current t 2048
# [a_layer] avg time per iter: 6.760 ms
# current t 4096
# [a_layer] avg time per iter: 12.719 ms
# current t 8192
# [a_layer] avg time per iter: 25.622 ms
# current t 16384

####transformer
# [a_layer] avg time per iter: 1.514 ms
# current t 256
# [a_layer] avg time per iter: 1.527 ms
# current t 521
# [a_layer] avg time per iter: 1.562 ms
# current t 1024
# [a_layer] avg time per iter: 2.764 ms
# current t 2048
# [a_layer] avg time per iter: 5.346 ms
# current t 4096
# [a_layer] avg time per iter: 14.919 ms
# current t 8192
# [a_layer] avg time per iter: 48.167 ms
# current t 16384


####eval_speed:
###mask_gdn_r8
# [a_layer] avg time per iter: 1.181 ms
# current t 256
# [a_layer] avg time per iter: 1.199 ms
# current t 512
# [a_layer] avg time per iter: 1.208 ms
# current t 1024
# [a_layer] avg time per iter: 1.202 ms
# current t 2048
# [a_layer] avg time per iter: 1.200 ms
# current t 4096
# [a_layer] avg time per iter: 1.208 ms
# current t 8192
# [a_layer] avg time per iter: 1.201 ms
# current t 16384

###mask_gdn_r4
# [a_layer] avg time per iter: 1.181 ms
# current t 256
# [a_layer] avg time per iter: 1.199 ms
# current t 512
# [a_layer] avg time per iter: 1.208 ms
# current t 1024
# [a_layer] avg time per iter: 1.202 ms
# current t 2048
# [a_layer] avg time per iter: 1.200 ms
# current t 4096
# [a_layer] avg time per iter: 1.208 ms
# current t 8192
# [a_layer] avg time per iter: 1.201 ms
# current t 16384

####r2
# [a_layer] avg time per iter: 1.164 ms
# current t 256
# [a_layer] avg time per iter: 1.154 ms
# current t 512
# [a_layer] avg time per iter: 1.157 ms
# current t 1024
# [a_layer] avg time per iter: 1.157 ms
# current t 2048
# [a_layer] avg time per iter: 1.156 ms
# current t 4096
# [a_layer] avg time per iter: 1.160 ms
# current t 8192
# [a_layer] avg time per iter: 1.163 ms
# current t 16384

###base_line_gdn
# [a_layer] avg time per iter: 0.839 ms
# current t 256
# [a_layer] avg time per iter: 0.838 ms
# current t 512
# [a_layer] avg time per iter: 0.818 ms
# current t 1024
# [a_layer] avg time per iter: 0.817 ms
# current t 2048
# [a_layer] avg time per iter: 0.820 ms
# current t 4096
# [a_layer] avg time per iter: 0.819 ms
# current t 8192
# [a_layer] avg time per iter: 0.820 ms
# current t 16384

###base_line_gla
# 512 hh
# [a_layer] avg time per iter: 0.629 ms
# current t 512
# 1024 hh
# [a_layer] avg time per iter: 0.629 ms
# current t 1024
# 2048 hh
# [a_layer] avg time per iter: 0.633 ms
# current t 2048
# 4096 hh
# [a_layer] avg time per iter: 0.630 ms
# current t 4096
