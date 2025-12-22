"""
Custom attention processors for MM-DiT models.

Based on diffusers attention processors with modifications for P2P-style attention manipulation.
- JointAttnProcessor2_0 (diffusers) -> P2P_JointAttnProcessor (for SD3)
- FluxAttnProcessor (diffusers) -> P2P_FluxAttnProcessor (for Flux)
- AttnProcessor (diffusers) -> P2P_SDAttnProcessor (for SD 1.x/2.x U-Net)

Modifications are marked with # ===== P2P ===== and # ===== END P2P =====
"""

import torch
import torch.nn.functional as F
from typing import Optional


# ===== P2P =====
def get_attention_scores_wo_softmax(query, key, attention_mask=None, upcast_attention=False, scale=1.0):
    """Compute attention scores without softmax (for manual attention computation)."""
    if upcast_attention:
        query = query.float()
        key = key.float()

    if attention_mask is None:
        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
    else:
        baddbmm_input = attention_mask
        beta = 1

    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=scale,
    )
    return attention_scores
# ===== END P2P =====


class P2P_JointAttnProcessor:
    """
    Attention processor for SD3 with P2P control.
    Based on diffusers JointAttnProcessor2_0 (lines 1420-1503 in attention_processor.py).

    In SD3: Image tokens come FIRST [image | text], then text tokens are concatenated.
    """

    # ===== P2P =====
    def __init__(self, controller, place_in_transformer, use_sdpa=True):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("P2P_JointAttnProcessor requires PyTorch 2.0+")
        self.controller = controller
        self.place_in_transformer = place_in_transformer
        self.use_sdpa = use_sdpa
    # ===== END P2P =====

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # SD3: [image | text]
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        # ===== P2P =====
        query = query.reshape(batch_size * attn.heads, -1, head_dim)
        key = key.reshape(batch_size * attn.heads, -1, head_dim)
        value = value.reshape(batch_size * attn.heads, -1, head_dim)

        if self.use_sdpa:
            # Projection replacement mode: controller modifies q, k, v in-place
            self.controller(query, key, value, self.place_in_transformer)
            query = query.reshape(batch_size, attn.heads, -1, head_dim)
            key = key.reshape(batch_size, attn.heads, -1, head_dim)
            value = value.reshape(batch_size, attn.heads, -1, head_dim)
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        else:
            # Attention region replacement mode: compute attention scores manually
            attention_scores = get_attention_scores_wo_softmax(query, key, attention_mask, getattr(attn, 'upcast_attention', False), attn.scale)
            attention_probs, value = self.controller.handle_attention(attention_scores, value, self.place_in_transformer)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = hidden_states.reshape(batch_size, attn.heads, -1, head_dim)
        # ===== END P2P =====

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class P2P_FluxAttnProcessor:
    """
    Attention processor for Flux with P2P control.
    Based on diffusers FluxAttnProcessor (lines 75-139 in transformer_flux.py).

    In Flux: Text tokens come FIRST [text | image], then image tokens.
    """

    # ===== P2P =====
    def __init__(self, controller, place_in_transformer, use_sdpa=True):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("P2P_FluxAttnProcessor requires PyTorch 2.0+")
        self.controller = controller
        self.place_in_transformer = place_in_transformer
        self.use_sdpa = use_sdpa
    # ===== END P2P =====

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # Flux: [text | image]
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # ===== P2P =====
        # Apply P2P controller for ALL blocks (both dual and single)
        query = query.reshape(batch_size * attn.heads, -1, head_dim)
        key = key.reshape(batch_size * attn.heads, -1, head_dim)
        value = value.reshape(batch_size * attn.heads, -1, head_dim)

        if self.use_sdpa:
            # Projection replacement mode: controller modifies q, k, v in-place
            self.controller(query, key, value, self.place_in_transformer)
            query = query.reshape(batch_size, attn.heads, -1, head_dim)
            key = key.reshape(batch_size, attn.heads, -1, head_dim)
            value = value.reshape(batch_size, attn.heads, -1, head_dim)
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        else:
            # Attention region replacement mode: compute attention scores manually
            # Flux doesn't have attn.scale, compute manually
            scale = head_dim ** -0.5
            attention_scores = get_attention_scores_wo_softmax(query, key, attention_mask, getattr(attn, 'upcast_attention', False), scale)
            attention_probs, value = self.controller.handle_attention(attention_scores, value, self.place_in_transformer)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = hidden_states.reshape(batch_size, attn.heads, -1, head_dim)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Single branch (FluxSingleTransformerBlock) - no output projections
        if encoder_hidden_states is None:
            return hidden_states

        # Dual branch - split and apply output projections
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )
        # ===== END P2P =====

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class P2P_SDAttnProcessor:
    """
    Attention processor for SD 1.x/2.x U-Net with P2P control.
    Based on diffusers AttnProcessor (without SDPA, to expose attention weights).
    """

    # ===== P2P =====
    def __init__(self, controller, place_in_unet):
        self.controller = controller
        self.place_in_unet = place_in_unet
    # ===== END P2P =====

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # ===== P2P =====
        # P2P attention control - controller modifies attention_probs in-place
        self.controller(attention_probs, attn.is_cross_attention, self.place_in_unet)
        # ===== END P2P =====

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# ===== Attention Inspection Processors (for visualization) =====

class InspectJointAttnProcessor:
    """
    Attention processor for SD3 that captures attention for visualization.
    Computes attention manually (no SDPA) to expose attention weights.
    """

    def __init__(self, inspector, place_in_transformer):
        self.inspector = inspector
        self.place_in_transformer = place_in_transformer

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args, **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # SD3: [image | text]
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        # Compute attention manually to capture weights
        query = query.reshape(batch_size * attn.heads, -1, head_dim)
        key = key.reshape(batch_size * attn.heads, -1, head_dim)
        value = value.reshape(batch_size * attn.heads, -1, head_dim)

        attention_scores = get_attention_scores_wo_softmax(
            query, key, attention_mask,
            getattr(attn, 'upcast_attention', False), attn.scale
        )
        attention_probs = self.inspector(attention_scores, self.place_in_transformer)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = hidden_states.view(batch_size, attn.heads, -1, head_dim)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


class InspectFluxAttnProcessor:
    """
    Attention processor for Flux that captures attention for visualization.
    Processes ALL blocks (dual and single) with manual attention computation.
    For single blocks, sequence is already [txt|img], T2I/I2T regions still exist.
    """

    def __init__(self, inspector, place_in_transformer):
        self.inspector = inspector
        self.place_in_transformer = place_in_transformer

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Dual branch: concatenate encoder hidden states
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # Flux: [text | image]
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Compute attention manually for ALL blocks (including single blocks)
        query = query.reshape(batch_size * attn.heads, -1, head_dim)
        key = key.reshape(batch_size * attn.heads, -1, head_dim)
        value = value.reshape(batch_size * attn.heads, -1, head_dim)

        # Flux doesn't have attn.scale, compute manually
        scale = head_dim ** -0.5
        attention_scores = get_attention_scores_wo_softmax(
            query, key, attention_mask,
            getattr(attn, 'upcast_attention', False), scale
        )
        attention_probs = self.inspector(attention_scores, self.place_in_transformer)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = hidden_states.view(batch_size, attn.heads, -1, head_dim)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states

        return hidden_states
