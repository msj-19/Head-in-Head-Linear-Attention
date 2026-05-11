# -*- coding: utf-8 -*-

from datasets import load_from_disk,concatenate_datasets, load_dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          Trainer)
# import fla  # noqa
from flame.data import DataCollatorForLanguageModeling
from flame.logging import LogCallback, get_logger
from flame.parser import get_train_args
import sys
sys.path.append('/mnt/jfzn/msj/flash-linear-attention/legacy/training')
sys.path.append('/mnt/jfzn/msj/flash-linear-attention/legacy/training/fla2')
# logger = get_logger(__name__)

from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM, GatedDeltaNetModel
logger = get_logger(__name__)
# breakpoint()
print(GatedDeltaNetConfig.model_type)
AutoConfig.register("gated_deltanet",GatedDeltaNetConfig)
AutoModelForCausalLM.register(GatedDeltaNetConfig,GatedDeltaNetForCausalLM)

from fla.models import DeltaNetConfig, DeltaNetForCausalLM, DeltaNetModel
print(DeltaNetConfig.model_type)
AutoConfig.register("delta_net",DeltaNetConfig)
AutoModelForCausalLM.register(DeltaNetConfig,DeltaNetForCausalLM)

from fla3.models import RWKV7ForCausalLM,RWKV7Model,RWKV7Config
logger = get_logger(__name__)
print(RWKV7Config.model_type)
AutoConfig.register("rwkv7",RWKV7Config)
AutoModelForCausalLM.register(RWKV7Config,RWKV7ForCausalLM)

from fla3.models import mask_RWKV7ForCausalLM,mask_RWKV7Model,mask_RWKV7Config
logger = get_logger(__name__)
print(mask_RWKV7Config.model_type)
AutoConfig.register("mask_rwkv7",mask_RWKV7Config)
AutoModelForCausalLM.register(mask_RWKV7Config,mask_RWKV7ForCausalLM)

from fla2.models import mask_gdnForCausalLM,mask_gdnModel,mask_gdnConfig
print(mask_gdnConfig.model_type)
AutoConfig.register("mask_gdn",mask_gdnConfig)
AutoModelForCausalLM.register(mask_gdnConfig,mask_gdnForCausalLM)


def main():
    args = get_train_args()
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=True,
        add_bos_token=True,
        add_eos_token=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))
    if args.from_config:
        logger.info("All model params are randomly initialized for from-scratch training.")
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(args.model_name_or_path))
    else:
        logger.info(f"Loading pretrained checkpoint {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.train()

    trainable_params, all_param = model.num_parameters(only_trainable=True), model.num_parameters()
    logger.info(f"% of trainable params: {trainable_params:d} / {all_param:d} = {trainable_params / all_param:.2%}")
    logger.info(f"{tokenizer}\n{model}\n{model.config}")
    

    print(f"% of trainable params: {trainable_params:d} / {all_param:d} = {trainable_params / all_param:.2%}")
    logger.info(f"Loading the `{args.split}` split directly from the cache {args.cache_dir}...")
    
    cache_dir = args.cache_dir.split(',')
    if len(cache_dir)>1:  
        dataset = [load_from_disk(path) for path in cache_dir]
        dataset = concatenate_datasets(dataset)
    else:
        dataset = load_from_disk(cache_dir[0])    
    logger.info(f"{dataset}")
    logger.info(f"Shuffling the dataset with seed {args.seed}")
    dataset = dataset.shuffle(seed=args.seed)
    logger.info("Creating the data collator")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, varlen=args.varlen)
    logger.info(f"{data_collator}")

    if args.lr_scheduler_type == 'cosine_with_min_lr':
        args.lr_scheduler_kwargs = {'min_lr_rate': 0.1}
    if args.lr_scheduler_type == 'warmup_stable_decay':
        args.lr_scheduler_kwargs = {
            'num_stable_steps': args.max_steps * 0.9 - args.warmup_steps,
            'num_decay_steps': args.max_steps * 0.1
        }

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCallback()],
        train_dataset=dataset
    )
    results = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(trainer.args.output_dir)

    trainer.log_metrics("train", results.metrics)
    trainer.save_metrics("train", results.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
