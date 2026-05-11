# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_mask_gdn import mask_gdnConfig
from .modeling_mask_gdn import mask_gdnForCausalLM, mask_gdnModel

AutoConfig.register(mask_gdnConfig.model_type, mask_gdnConfig)
AutoModel.register(mask_gdnConfig, mask_gdnModel)
AutoModelForCausalLM.register(mask_gdnConfig, mask_gdnForCausalLM)

__all__ = ['mask_gdnConfig', 'mask_gdnForCausalLM', 'mask_gdnModel']
