export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_LAUNCH_BLOCKING=1

MODEL_PATH=''
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu  --num_processes 4 --main_process_port 29545 harness.py --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=bfloat16 \
    --tasks arc_easy,arc_challenge,hellaswag,lambada_standard,piqa,winogrande,wikitext \
    --output_path  \
    --batch_size 4 \
    --device cuda 





