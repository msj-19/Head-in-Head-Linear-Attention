# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_emdeltanet import emdeltanetConfig
from .modeling_emdeltanet import emdeltanetForCausalLM, emdeltanetModel

AutoConfig.register(emdeltanetConfig.model_type, emdeltanetConfig)
AutoModel.register(emdeltanetConfig, emdeltanetModel)
AutoModelForCausalLM.register(emdeltanetConfig, emdeltanetForCausalLM)

__all__ = ['emdeltanetConfig', 'emdeltanetForCausalLM', 'emdeltanetModel']
