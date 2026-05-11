args=$@
for arg in $args; do
    eval "$arg"
done

echo "model:            ${model:=fla-hub/gla-1.3B-100B}"
echo "tokenizer:        ${tokenizer:=/mnt/jfzn/msj/train_exp/gdn_1B_a800}"
echo "project:          ${project:=fla}"
echo "type:             ${type:=gla}"
echo "data:             ${data:=}"
echo "name:             ${name:=}"
echo "cache:            ${cache:=}"
echo "varlen:           ${varlen:=false}"
echo "seed:             ${seed:=42}"
echo "context:          ${context:=2048}"
echo "steps:            ${steps:=0}"
echo "save:             ${save:=2048}"
echo "limit:            ${limit:=1}"
echo "preprocessing:    ${preprocessing:=32}"
echo "workers:          ${workers:=32}"
echo "prefetch:         ${prefetch:=2}"
echo "logging:          ${logging:=32}"
echo "config:           ${config:=configs/deepspeed_sencore.yaml}"

echo "lr:               ${lr:=3e-4}"
echo "scheduler:        ${scheduler:=cosine_with_min_lr}"
echo "epochs:           ${epochs:=1}"
echo "optim:            ${optim:=adamw_torch_fused}"
echo "decay:            ${decay:=0.01}"
echo "beta1:            ${beta1:=0.9}"
echo "beta2:            ${beta2:=0.95}"
echo "norm:             ${norm:=1.0}"
echo "batch:            ${batch:=32}"
echo "update:           ${update:=1}"
echo "warmup:           ${warmup:=512}"
echo "path:             ${path:=}"
echo "checkpoint:       ${checkpoint:=}"
echo "node:             ${node:=}"
echo "rank:             ${rank:=}"
echo "ip:               ${ip:=}"
echo "port:             ${port:=}"
echo "nodes:            ${nodes:=1}"
echo "gpus:             ${gpus:=8}"

params="--model_name_or_path $model \
    --tokenizer $tokenizer \
    --use_fast_tokenizer \
    --do_train \
    --dataset $data \
    --context_length $context \
    --preprocessing_num_workers $preprocessing \
    --dataloader_num_workers $workers \
    --dataloader_prefetch_factor $prefetch \
    --output_dir $path \
    --overwrite_output_dir \
    --logging_steps $logging \
    --include_num_input_tokens_seen \
    --save_steps $save \
    --save_total_limit $limit \
    --learning_rate $lr \
    --lr_scheduler_type $scheduler \
    --warmup_steps $warmup \
    --optim $optim \
    --weight_decay $decay \
    --adam_beta1=$beta1 \
    --adam_beta2=$beta2 \
    --max_grad_norm $norm \
    --num_train_epochs $epochs \
    --per_device_train_batch_size $batch \
    --gradient_accumulation_steps $update \
    --seed $seed \
    --logging_steps $logging \
    --log_level info \
    --bf16"

if [ $steps -gt 0 ]; then
    params+=" --max_steps $steps"
fi

if [ "$name" != "" ]; then
  params+=" --dataset_name $name"
fi
if [ "$cache" != "" ]; then
  params+=" --cache_dir $cache"
fi
if [ "$varlen" == "true" ]; then
  params+=" --varlen"
fi
if [ "$checkpoint" != "" ]; then
  params+=" --resume_from_checkpoint $checkpoint"
echo '*****************************************'$checkpoint
fi
# if [ "$WANDB_DISABLED" != "true" ]; then
#   params+=" --report_to wandb \
#   --run_name $type.$(basename $path)"
# else
params+=" --report_to none"
# fi


set -x
mkdir -p $path
cp * $path
cp -r configs $path
cp -r flame   $path
cp -r fla2 $path
cp -r fla3 $path
# export WANDB_DISABLED=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
if [ "$date" == "" ]; then
  date=$(date +%Y%m%d%H%M)
fi

/mnt/jfzn/miniconda3/envs/msj_triton36/bin/accelerate launch \
 --config_file $config \
 --num_processes $((SENSECORE_PYTORCH_NNODES*SENSECORE_ACCELERATE_DEVICE_COUNT)) \
 --num_machines $SENSECORE_PYTORCH_NNODES \
 --machine_rank $SENSECORE_PYTORCH_NODE_RANK \
 --main_process_ip $MASTER_ADDR \
 --main_process_port $MASTER_PORT \
 --mixed_precision=bf16 \
 --use_deepspeed \
 -- \
 run.py $params

echo "RUNNING DONE!"