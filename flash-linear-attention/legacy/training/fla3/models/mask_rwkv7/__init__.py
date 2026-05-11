# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_mask_rwkv7 import mask_RWKV7Config
from .modeling_mask_rwkv7 import mask_RWKV7ForCausalLM, mask_RWKV7Model

AutoConfig.register(mask_RWKV7Config.model_type, mask_RWKV7Config, True)
AutoModel.register(mask_RWKV7Config, mask_RWKV7Model, True)
AutoModelForCausalLM.register(mask_RWKV7Config, mask_RWKV7ForCausalLM, True)


__all__ = ['mask_RWKV7Config', 'mask_RWKV7ForCausalLM', 'mask_RWKV7Model']
