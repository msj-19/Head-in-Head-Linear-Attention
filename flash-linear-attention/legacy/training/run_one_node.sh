bash train.sh \
 type=mask_gdn \
 lr=3e-4 \
 scheduler=cosine_with_min_lr \
 batch=16 \
 update=2 \
 warmup=512 \
 steps=30720 \
 context=2048 \
 gpus=8 \
 nodes=1 \
 path=/mnt/jfzn/msj/train_exp/mask_path_attn_r4_test2 \
 project=fla \
 model=configs/mask_gdn_340M.json \
 data=cerebras/SlimPajama-627B \
 name=SlimPajama \
 cache=/mnt/jfzn/data/SlimPajama-627B/pre_slimp_chunk1/slimp/train \
 tasks=run \




