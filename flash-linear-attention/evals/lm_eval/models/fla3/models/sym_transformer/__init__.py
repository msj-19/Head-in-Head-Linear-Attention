# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM


from .configuration_transformer import sym_TransformerConfig
from .modeling_transformer import (
    sym_TransformerForCausalLM, sym_TransformerModel)

AutoConfig.register(sym_TransformerConfig.model_type, sym_TransformerConfig)
AutoModel.register(sym_TransformerConfig, sym_TransformerModel)
AutoModelForCausalLM.register(sym_TransformerConfig, sym_TransformerForCausalLM)


__all__ = ['sym_TransformerConfig', 'sym_TransformerForCausalLM', 'sym_TransformerModel']


# from fla.models.transformer.configuration_transformer import TransformerConfig
# from fla.models.transformer.modeling_transformer import TransformerForCausalLM, TransformerModel

# AutoConfig.register(TransformerConfig.model_type, TransformerConfig)
# AutoModel.register(TransformerConfig, TransformerModel)
# AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM)


# __all__ = ['TransformerConfig', 'TransformerForCausalLM', 'TransformerModel']
