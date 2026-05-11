# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_mask_path_attention import MaskPathAttentionConfig
from .modeling_mask_path_attention import MaskPathAttentionForCausalLM, MaskPathAttentionModel

AutoConfig.register(MaskPathAttentionConfig.model_type, MaskPathAttentionConfig)
AutoModel.register(MaskPathAttentionConfig, MaskPathAttentionModel)
AutoModelForCausalLM.register(MaskPathAttentionConfig, MaskPathAttentionForCausalLM)


__all__ = ['MaskPathAttentionConfig', 'MaskPathAttentionForCausalLM', 'MaskPathAttentionModel']
