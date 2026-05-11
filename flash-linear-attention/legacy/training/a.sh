export PDSH_RCMD_TYPE=ssh
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=0  # 明确启用IB

bash train_node.sh \
 type=gated_deltanet \
 lr=3e-4 \
 scheduler=cosine_with_min_lr \
 batch=8 \
 update=4 \
 warmup=512 \
 steps=50016 \
 context=2048 \
 gpus=8 \
 nodes=4 \
 path=/mnt/jfzn/msj/train_exp/gdn_1B_a800 \
 project=fla \
 model=configs/gdn_1B.json \
 data=cerebras/SlimPajama-627B \
 name=SlimPajama \
 cache=/mnt/jfzn/data/SlimPajama-627B/pre_slimp_chunk1/slimp/train,/mnt/jfzn/data/SlimPajama-627B/pre_slimp_chunk2/slimp/train \
 checkpoint=/mnt/jfzn/msj/train_exp/gdn_1B_a800/checkpoint-2048 \
# bash train_node.sh \
#  type=gdn \
#  lr=3e-4 \
#  scheduler=cosine_with_min_lr \
#  batch=8 \
#  update=4 \
#  warmup=512 \
#  steps=50016 \
#  context=2048 \
#  gpus=8 \
#  nodes=2 \
#  path=/mnt/jfzn/msj/train_exp/gdn_1B_hrr4_head10——test \
#  project=fla \
#  model=configs/gdn_1B.json \
#  data=cerebras/SlimPajama-627B \
#  name=SlimPajama \
#  cache=/mnt/jfzn/data/SlimPajama-627B/pre_slimp_chunk1/slimp/train,/mnt/jfzn/data/SlimPajama-627B/pre_slimp_chunk2/slimp/train \
 





