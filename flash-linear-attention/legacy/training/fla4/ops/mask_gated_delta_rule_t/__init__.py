# -*- coding: utf-8 -*-

from .chunk import mask_gated_chunk_delta_rule
from .recurrent_fuse import mask_fused_recurrent_gated_delta_rule

__all__ = [
    # 'mask_fused_chunk_delta_rule',
    'mask_fused_recurrent_gated_delta_rule',
    'mask_gated_chunk_delta_rule',
]
