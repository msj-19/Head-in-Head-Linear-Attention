# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024, Songlin Yang, Yu Zhang.

import argparse
import time
import copy
import torch
import datasets
from datasets import load_dataset,load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
import sys
from torch.cuda import max_memory_allocated, memory_allocated
sys.path.append('/mnt/jfzn/msj/flash-linear-attention/evals/lm_eval/models')
import fla  
import fla4
from fla2.models import mask_gdnConfig,mask_gdnForCausalLM
print(mask_gdnConfig.model_type)
AutoConfig.register("mask_gdn",mask_gdnConfig)
AutoModelForCausalLM.register(mask_gdnConfig,mask_gdnForCausalLM)

from fla.models import TransformerConfig,TransformerForCausalLM,TransformerModel
print(TransformerConfig.model_type)
AutoConfig.register("transformer",TransformerConfig)
AutoModelForCausalLM.register(TransformerConfig,TransformerForCausalLM)

from fla.models import DeltaNetConfig, DeltaNetForCausalLM, DeltaNetModel
print(DeltaNetConfig.model_type)
AutoConfig.register("delta_net",DeltaNetConfig)
AutoModelForCausalLM.register(DeltaNetConfig,DeltaNetForCausalLM)

from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM, GatedDeltaNetModel
print(GatedDeltaNetConfig.model_type)
AutoConfig.register("gated_deltanet",GatedDeltaNetConfig)
AutoModelForCausalLM.register(GatedDeltaNetConfig,GatedDeltaNetForCausalLM)

from fla.models import GatedDeltaProductConfig,GatedDeltaProductForCausalLM,GatedDeltaProductModel
print(GatedDeltaProductConfig.model_type)
AutoConfig.register("gated_deltaproduct",GatedDeltaProductConfig)
AutoModelForCausalLM.register(GatedDeltaProductConfig,GatedDeltaProductForCausalLM)

from fla3.models import RWKV7ForCausalLM,RWKV7Model,RWKV7Config
print(RWKV7Config.model_type)
AutoConfig.register("rwkv7",RWKV7Config)
AutoModelForCausalLM.register(RWKV7Config,RWKV7ForCausalLM)

from fla3.models import mask_RWKV7ForCausalLM,mask_RWKV7Model,mask_RWKV7Config
print(mask_RWKV7Config.model_type)
AutoConfig.register("mask_rwkv7",mask_RWKV7Config)
AutoModelForCausalLM.register(mask_RWKV7Config,mask_RWKV7ForCausalLM)


from fla3.modes import 
def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}Yi{suffix}'


def main():
    parser = argparse.ArgumentParser(description="Generation benchmarking")
    # parser.add_argument("--path", type=str, default="/mnt/jfzn/msj/train_exp/mask_gdn_1B_hrr4_byt")
    # parser.add_argument("--path", type=str, default="/mnt/jfzn/msj/download_model/transformer-1.3B-100B")
    parser.add_argument("--path", type=str, default="/mnt/jfzn/msj/train_exp/mask_gdn_hrr4")
    # parser.add_argument("--path", type=str, default="/mnt/jfzn/msj/train_exp/gdn_1B_a800")
    # parser.add_argument("--path", type=str, default="/mnt/jfzn/msj/train_exp/gated_deltaproduct_layer17")
    parser.add_argument("--data", type=str, default="fla-hub/pg19")
    parser.add_argument("--maxlen", type=int, default=1)
    parser.add_argument("--no-cache", action='store_true')
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--topp", type=float, default=0.2)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    args = parser.parse_args()
    print('down')

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # print(f"Loading {args.path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.path,
        trust_remote_code=True,
        add_eos_token=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # print(f"{tokenizer}")

    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        device_map={"": device},
        torch_dtype=dtype,
        use_cache=not args.no_cache
    )
    model.eval()

    input_texts = datasets.load_from_disk('/mnt/jfzn/msj/yarn/pg19-test-tokenized/train')
    input_ids_org = input_texts[0]['input_ids']
    input_ids_org = torch.tensor(input_ids_org, device="cuda").to(device=device)
    input_ids_org = torch.cat([input_ids_org] * 9, dim=-1).unsqueeze(0)
    print(input_ids_org.shape)

    # T_list =  [0.25,0.5,1,2,4,8,16,32,64]
    T_list =  [0.25,0.5,1,2,4,8,16,32,64,128,256,512]
    T_list = [i*1024 for i in T_list]
    print(T_list)
    print(model)
    # org_length = int(512*1024)

    input_ids = input_ids_org[:, :10].contiguous()
    with torch.inference_mode():
        outs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True
        )
        logits = outs.logits
        past_key_values = outs.past_key_values
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    with torch.inference_mode():
        for ttt in range(2147483647):
            out = model(
                input_ids=next_token,      # 1 token
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = out.logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            if past_key_values.get_seq_length(0) in T_list:
                print(past_key_values.get_seq_length(0))
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(1000):
                    out = model(
                        input_ids=next_token,      # 1 token
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    logits = out.logits
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                torch.cuda.synchronize()
                t1 = time.time()
                print(f"next-token decode time: {(t1 - t0) * 1000} ms")
                print(f"Max memory allocated: {sizeof_fmt(memory_allocated(device))}")
main()


#### mask_gdn  $########byt
# next-token decode time: 30249.38988685608 ms
# Max memory allocated: 2.GiB
# current t 512
# next-token decode time: 30073.04329872131 ms
# Max memory allocated: 2.GiB
# current t 1024
# next-token decode time: 30308.321714401245 ms
# Max memory allocated: 2.9GiB
# current t 2048
# next-token decode time: 30382.089138031006 ms
# Max memory allocated: 3.0GiB
# current t 4096


#### mask_gdn fixed canbe 26000~1.2x速度变化
# next-token decode time: 30249.38988685608 ms
# Max memory allocated: 2.GiB
# current t 512
# next-token decode time: 30073.04329872131 ms
# Max memory allocated: 2.GiB
# current t 1024
# next-token decode time: 30308.321714401245 ms
# Max memory allocated: 2.9GiB
# current t 2048
# next-token decode time: 30382.089138031006 ms
# Max memory allocated: 3.0GiB
# current t 4096



######gdn
# current t 256
# next-token decode time: 24923.27117919922 ms
# Max memory allocated: 2.6GiB
# current t 512
# next-token decode time: 24835.999488830566 ms
# Max memory allocated: 2.6GiB
# current t 1024
# next-token decode time: 24888.691902160645 ms
# Max memory allocated: 2.7GiB
# current t 2048
# next-token decode time: 24930.066347122192 ms
# Max memory allocated: 2.7GiB
# current t 4096
# next-token decode time: 24965.673208236694 ms
# Max memory allocated: 2.9GiB
# current t 8192
# next-token decode time: 24960.266828536987 ms
# Max memory allocated: 3.1GiB
# current t 16384
# next-token decode time: 25009.57465171814 ms
# Max memory allocated: 3.6GiB
# current t 32768
# next-token decode time: 24959.109783172607 ms
# Max memory allocated: 4.6GiB
# current t 65536
# next-token decode time: 24917.89484024048 ms
# Max memory allocated: 6.5GiB
# current t 131072
# next-token decode time: 24951.22981071472 ms
# Max memory allocated: 10.4GiB
# current t 262144
# next-token decode time: 24994.324922561646 ms
# Max memory allocated: 18.2GiB



##########transformer
# current t 256
# next-token decode time: 22243.457317352295 ms
# Max memory allocated: 3.4GiB
# current t 512
# next-token decode time: 22260.204315185547 ms
# Max memory allocated: 3.4GiB
# current t 1024
# next-token decode time: 22290.91715812683 ms
# Max memory allocated: 3.6GiB
# current t 2048
# next-token decode time: 22295.89009284973 ms
# Max memory allocated: 3.8GiB
# current t 4096
# next-token decode time: 22357.794761657715 ms
# Max memory allocated: 4.3GiB
# current t 8192
# next-token decode time: 22323.17614555359 ms
# Max memory allocated: 5.3GiB
# current t 16384
# next-token decode time: 22310.540914535522 ms
# Max memory allocated: 7.3GiB
# current t 32768
# next-token decode time: 27021.798372268677 ms
# Max memory allocated: 11.3GiB
# current t 65536
# next-token decode time: 50395.55740356445 ms
# Max memory allocated: 19.2GiB







###gated_deltaproduct:
# current t 256
# next-token decode time: 41696.47932052612 ms
# Max memory allocated: 2.7GiB
# current t 512
# next-token decode time: 41789.73197937012 ms
# Max memory allocated: 2.7GiB
# current t 1024
# next-token decode time: 41457.9074382782 ms
# Max memory allocated: 2.7GiB



################线性推理时长与内存占用,use this draw memory 占用，用上面的画内存占用
######gated_deltaproduct_17l
# current t 256
# next-token decode time: 16106.451272964478 ms
# Max memory allocated: 2.7GiB
# current t 512
# next-token decode time: 21260.28871536255 ms
# Max memory allocated: 2.7GiB
# current t 1024
# next-token decode time: 42442.925453186035 ms
# Max memory allocated: 2.7GiB
# current t 2048
# next-token decode time: 84973.71315956116 ms
# Max memory allocated: 2.7GiB
# current t 4096
# next-token decode time: 170175.98295211792 ms
# Max memory allocated: 2.7GiB
# current t 8192
# next-token decode time: 338854.9962043762 ms
# Max memory allocated: 2.7GiB


############gdn
# next-token decode time: 6737.887382507324 ms
# Max memory allocated: 2.6GiB
# current t 512
# next-token decode time: 13411.81206703186 ms
# Max memory allocated: 2.6GiB
# current t 1024
# next-token decode time: 26739.73822593689 ms
# Max memory allocated: 2.6GiB
# current t 2048
# next-token decode time: 53390.804290771484 ms
# Max memory allocated: 2.6GiB
# current t 4096
# next-token decode time: 105663.95664215088 ms
# Max memory allocated: 2.6GiB
# current t 8192
# next-token decode time: 212001.98602676392 ms
# Max memory allocated: 2.6GiB


####mask_gdn
# current t 256
# next-token decode time: 8921.972274780273 ms
# Max memory allocated: 2.6GiB
# current t 512
# next-token decode time: 17894.01912689209 ms
# Max memory allocated: 2.6GiB
# current t 1024
# next-token decode time: 35472.59306907654 ms
# Max memory allocated: 2.6GiB
# current t 2048
# next-token decode time: 70049.07608032227 ms
# Max memory allocated: 2.6GiB
# current t 4096
# next-token decode time: 140247.71761894226 ms
# Max memory allocated: 2.6GiB
# current t 8192
# next-token decode time: 280026.5429019928 ms
# Max memory allocated: 2.6GiB
# current t 16384
# next-token decode time: 560680.6931495667 ms
# Max memory allocated: 2.6GiB
# current t 32768
# next-token decode time: 1120775.9864330292 ms
# Max memory allocated: 2.6GiB
# current t 65536
# next-token decode time: 2240647.1407413483 ms
# Max memory allocated: 2.6GiB
# current t 131072
# next-token decode time: 4499878.668546677 ms
# Max memory allocated: 2.6GiB
# current t 262144
# next-token decode time: 8954407.489776611 ms
# Max memory allocated: 2.6GiB


############transformer
# current t 256
# next-token decode time: 5767.763614654541 ms
# Max memory allocated: 3.2GiB
# current t 512
# next-token decode time: 11486.648321151733 ms
# Max memory allocated: 3.2GiB
# current t 1024
# next-token decode time: 22993.65544319153 ms
# Max memory allocated: 3.3GiB
# current t 2048
# next-token decode time: 46087.47673034668 ms
# Max memory allocated: 3.5GiB
# current t 4096
# next-token decode time: 92339.85090255737 ms
# Max memory allocated: 3.9GiB
# current t 8192
# next-token decode time: 184960.54816246033 ms
# Max memory allocated: 4.6GiB
# current t 16384
# next-token decode time: 370783.3569049835 ms
# Max memory allocated: 6.1GiB
# current t 32768
# next-token decode time: 753638.3321285248 ms
# Max memory allocated: 9.1GiB
# current t 65536
# next-token decode time: 2012742.9513931274 ms
# Max memory allocated: 15.1GiB
# current t 131072
# next-token decode time: 6868342.051029205 ms
# Max memory allocated: 27.1GiB
# current t 262144
# next-token decode time: 27440205.795288086 ms
# Max memory allocated: 51.3GiB


# 256
# next-token decode time: 22732.054710388184 ms
# Max memory allocated: 3.4GiB
# 2048
# next-token decode time: 22752.167224884033 ms
# Max memory allocated: 3.7GiB
# 4096
# next-token decode time: 22788.618326187134 ms
# Max memory allocated: 4.1GiB
# 8192
# next-token decode time: 22822.450637817383 ms
# Max memory allocated: 4.8GiB
# 16384
# next-token decode time: 22858.738660812378 ms
# Max memory allocated: 6.3GiB
# 32768
# next-token decode time: 27027.065753936768 ms
# Max memory allocated: 9.3GiB
# 65536
# next-token decode time: 50587.324380874634 ms
# Max memory allocated: 15.3GiB
# 131072
# next-token decode time: 98216.66598320007 ms
# Max memory allocated: 27.3GiB
# 262144
# next-token decode time: 206686.0704421997 ms
# Max memory allocated: 51.5GiB