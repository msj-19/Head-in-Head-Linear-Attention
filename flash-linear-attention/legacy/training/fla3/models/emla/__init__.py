# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.emla.configuration_emla import emlaConfig
from fla.models.emla.modeling_emla import emlaForCausalLM, emlaModel

AutoConfig.register(emlaConfig.model_type, emlaConfig)
AutoModel.register(emlaConfig, emlaModel)
AutoModelForCausalLM.register(emlaConfig, emlaForCausalLM)

__all__ = ['emlaConfig', 'emlaForCausalLM', 'emlaModel']
