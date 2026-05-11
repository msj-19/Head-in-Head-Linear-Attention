export HF_HOME=/mnt/jfzn/HF/
python preprocess.py \
  --dataset /mnt/jfzn/data/SlimPajama-627B/train/chunk3 \
  --name slimp \
  --split train \
  --output /mnt/jfzn/data/SlimPajama-627B/pre_slimp_chunk3 \
  --tokenizer /mnt/jfzn/msj/models--fla-hub--gla-1.3B-100B
