"""
Some of P2P modified from "https://github.com/google/prompt-to-prompt/".
"""

from .dit_utils import (
    register_qkv_control,
    create_qkv_controller,
    QKVControl,
    LocalBlend,
    p2p_callback,
    PROJECTION_METHODS,
    ATTENTION_REGION_METHODS,
)

from .unet_utils import (
    register_attention_control as register_unet_attention_control,
    create_controller as create_unet_controller,
    AttentionControl as UNetAttentionControl,
    LocalBlend as UNetLocalBlend,
    p2p_callback as unet_p2p_callback,
)

from .dit_attn_viz import (
    create_attention_inspector,
    register_attention_inspector,
    inspector_callback,
    AttentionInspector,
    visualize_attention_grid,
    visualize_attention_single,
)

from .common import (
    get_word_inds,
    get_refinement_mapper,
    get_equalizer,
    gaussian_smoothing,
    TOP5_BLOCKS,
)

__all__ = [
    # DiT utilities
    "register_qkv_control",
    "create_qkv_controller",
    "QKVControl",
    "LocalBlend",
    "p2p_callback",
    "PROJECTION_METHODS",
    "ATTENTION_REGION_METHODS",
    # U-Net utilities
    "register_unet_attention_control",
    "create_unet_controller",
    "UNetAttentionControl",
    "UNetLocalBlend",
    "unet_p2p_callback",
    # Visualization
    "create_attention_inspector",
    "register_attention_inspector",
    "inspector_callback",
    "AttentionInspector",
    "visualize_attention_grid",
    "visualize_attention_single",
    # Common
    "get_word_inds",
    "get_refinement_mapper",
    "get_equalizer",
    "gaussian_smoothing",
    "TOP5_BLOCKS",
]
