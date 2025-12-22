"""
DiT (Diffusion Transformer) utilities for prompt-based image editing.

This module implements the attention manipulation methods from:
"Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing"
Tested with SD3 and Flux architectures.
"""

import torch
import torch.nn.functional as F
from typing import Any, List, Optional
from diffusers import StableDiffusion3Pipeline
from diffusers.models.attention_processor import JointAttnProcessor2_0

from .common import (
    get_multi_word_inds,
    gaussian_smoothing,
    TOP5_BLOCKS,
)
from .attn_processors import P2P_JointAttnProcessor, P2P_FluxAttnProcessor


# Projection replacement methods (modify Q/K before attention), works with SDPA.
PROJECTION_METHODS = ["qiki", "qk", "qi", "ki", "qikt", "qtki", "qtkt", "kivi"]

# Attention region replacement methods (modify attention matrix directly)
# These require disabling SDPA for direct attention matrix access
ATTENTION_REGION_METHODS = ["I2I", "T2I", "I2T", "T2T"]


def register_qkv_control(pipeline, controller):
    """
    Register QKV control processor with the pipeline.

    Args:
        pipeline: SD3 or Flux pipeline instance
        controller: QKVControl instance
    """
    attn_procs = {}
    att_count = 0

    # Determine if we need SDPA (attention region methods require non-SDPA)
    use_sdpa = controller.method not in ATTENTION_REGION_METHODS

    # Store num_heads for CFG auto-detection
    controller.num_heads = pipeline.transformer.config.num_attention_heads

    for idx, name in enumerate(pipeline.transformer.attn_processors.keys()):
        # Skip attn2 (dedicated self-attention in SD3.5-M) - use default processor
        if "attn2" in name:
            attn_procs[name] = JointAttnProcessor2_0()
            continue

        att_count += 1
        place_in_transformer = name.split(".attn.")[0]
        controller.place_to_idx[place_in_transformer] = idx

        if isinstance(pipeline, StableDiffusion3Pipeline):
            attn_procs[name] = P2P_JointAttnProcessor(controller, place_in_transformer, use_sdpa=use_sdpa)
        else:
            attn_procs[name] = P2P_FluxAttnProcessor(controller, place_in_transformer, use_sdpa=use_sdpa)

    pipeline.controller = controller
    controller.num_att_layer = att_count
    pipeline.transformer.set_attn_processor(attn_procs)
    print(f"Registered {att_count} attention processors (SDPA: {use_sdpa}).")


def create_qkv_controller(
    pipeline_type: str,
    prompts: List[str],
    num_inference_steps: int,
    tokenizer: List[Any],
    device,
    dtype,
    # Replacement control
    method: str = "qiki",
    replace_steps: float = 0.2,
    replace_blocks: Optional[List[int]] = None,
    # Local blend control
    local_blend_words: Optional[List] = None,
    local_blend_threshold: float = 0.3,
    local_blend_steps: float = 0.5,
    local_blend_n_prev_steps: Optional[int] = None,
    tokenizer_choice: str = "both",
    # Attention aggregation control
    aggregate_blocks: Optional[List[int]] = None,
    # Model config
    t5_max_seq_len: Optional[int] = None,
    # Image dimensions (for non-square local blending)
    height: int = 1024,
    width: int = 1024,
) -> "QKVControl":
    """
    Create a QKVControl for MM-DiT attention manipulation.

    Args:
        pipeline_type: "sd3" or "flux"
        prompts: List of [source_prompt, target_prompt]
        num_inference_steps: Number of diffusion steps
        tokenizer: List of tokenizers (CLIP, T5)
        device: Computation device
        dtype: Data type

        # Replacement control
        method: Replacement method. Two types available:

            Projection methods (modify Q/K before attention computation):
            - "qiki": Replace q_i and k_i (paper's main method, DEFAULT)
                      Affects: I2I (fully), I2T (via q_i), T2I (via k_i)
            - "qk": Full Q,K replacement from source
            - "qi": Replace only q_i
            - "ki": Replace only k_i
            - "qikt": Replace q_i and k_t
            - "qtki": Replace q_t and k_i
            - "qtkt": Replace q_t and k_t
            - "kivi": Replace k_i and v_i

            Attention region methods (modify attention matrix after computation):
            - "I2I": Replace Image-to-Image attention region
            - "T2I": Replace Text-to-Image attention region
            - "I2T": Replace Image-to-Text attention region
            - "T2T": Replace Text-to-Text attention region

        replace_steps: Fraction of steps for replacement (τ in Algorithm 1, default: 0.2)
        replace_blocks: Block indices for replacement (None = all blocks)

        # Local blend control
        local_blend_words: Words to use for local blending (per prompt)
        local_blend_threshold: Threshold for binary mask (θ, default: 0.3)
        local_blend_steps: Fraction of steps for blending (η in Algorithm 1, default: 0.5)
        local_blend_n_prev_steps: Number of previous steps to use for mask aggregation (default: all)
        tokenizer_choice: SD3 only - "both", "clip", or "t5" for alpha layer computation, Flux only uses "t5".

        # Attention aggregation control
        aggregate_blocks: Block indices for attention storage/aggregation (None = all blocks)

        # Model config
        t5_max_seq_len: T5 sequence length (auto-detected: 256 for SD3, 512 for Flux)
        height: Image height (for non-square local blending, default: 1024)
        width: Image width (for non-square local blending, default: 1024)

    Returns:
        QKVControl instance
    """
    if pipeline_type not in ["sd3", "flux"]:
        raise ValueError(f"Pipeline type '{pipeline_type}' not recognized. Use 'sd3' or 'flux'.")

    # Auto-detect T5 sequence length if not provided
    if t5_max_seq_len is None:
        t5_max_seq_len = 256 if pipeline_type == "sd3" else 512
    if len(prompts) != 2:
        raise ValueError("Only two prompts are supported.")

    all_methods = PROJECTION_METHODS + ATTENTION_REGION_METHODS
    if method not in all_methods:
        raise ValueError(f"method '{method}' not recognized. Use one of: {all_methods}")
    if tokenizer_choice not in ["both", "clip", "t5"]:
        raise ValueError(f"tokenizer_choice '{tokenizer_choice}' not recognized. Use 'both', 'clip', or 't5'.")

    # Default local_blend_n_prev_steps to all steps
    if local_blend_n_prev_steps is None:
        local_blend_n_prev_steps = num_inference_steps

    # Setup local blend if words provided
    local_blend = None
    if local_blend_words is not None:
        local_blend = LocalBlend(
            pipeline_type=pipeline_type,
            prompts=prompts,
            words=local_blend_words,
            tokenizer=tokenizer,
            device=device,
            threshold=local_blend_threshold,
            t5_max_seq_len=t5_max_seq_len,
            tokenizer_choice=tokenizer_choice,
        )

    controller = QKVControl(
        pipeline_type=pipeline_type,
        prompts=prompts,
        local_blend=local_blend,
        num_inference_steps=num_inference_steps,
        tokenizer=tokenizer,
        method=method,
        replace_steps=replace_steps,
        replace_blocks=replace_blocks,
        local_blend_steps=local_blend_steps,
        local_blend_n_prev_steps=local_blend_n_prev_steps,
        t5_max_seq_len=t5_max_seq_len,
        aggregate_blocks=aggregate_blocks,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
    )
    return controller


class QKVControl:
    """
    QKV-based attention control for MM-DiT models.

    Supports two types of methods:
    1. Projection replacement: modify Q/K projections before attention (qiki, qk, etc.)
    2. Attention region replacement: modify attention matrix directly (I2I, T2I, etc.)
    """

    def __init__(
        self,
        pipeline_type: str,
        prompts: List[str],
        local_blend: Optional["LocalBlend"],
        num_inference_steps: int,
        tokenizer: List[Any],
        method: str,
        replace_steps: float,
        replace_blocks: Optional[List[int]],
        local_blend_steps: float,
        local_blend_n_prev_steps: int,
        t5_max_seq_len: int,
        aggregate_blocks: List[int],
        height: int,
        width: int,
        device,
        dtype,
    ):
        self.pipeline_type = pipeline_type
        self.prompts = prompts
        self.local_blend = local_blend
        self.num_inference_steps = num_inference_steps
        self.tokenizer = tokenizer
        self.height = height
        self.width = width
        self.device = device
        self.dtype = dtype

        # Replacement settings
        self.method = method
        self.replace_blocks = replace_blocks  # None means all blocks

        # Aggregation settings (for local blend)
        self.aggregate_blocks = aggregate_blocks
        self.local_blend_n_prev_steps = local_blend_n_prev_steps

        # Text sequence length: SD3 uses CLIP(77) + T5, Flux uses only T5
        self.text_seq_len = 77 + t5_max_seq_len if pipeline_type == "sd3" else t5_max_seq_len
        self.t5_max_seq_len = t5_max_seq_len
        self.alpha_layers = local_blend.alpha_layers if local_blend is not None else None

        # Step and block tracking
        self.cur_step = 0
        self.cur_att_layer = 0
        self.num_heads = -1  # Set by register_qkv_control
        self.num_att_layer = -1
        self.attention_store = {}
        self.batch_size = len(prompts)
        self.place_to_idx = {}

        # Convert replace_steps fraction to step indices
        if isinstance(replace_steps, float):
            replace_steps = (0.0, replace_steps)
        self.replace_steps = (
            int(num_inference_steps * replace_steps[0]),
            int(num_inference_steps * replace_steps[1]),
        )

        # Convert local_blend_steps fraction to step indices (η in Algorithm 1)
        if isinstance(local_blend_steps, float):
            local_blend_steps = (0.0, local_blend_steps)
        self.local_blend_steps = (
            int(num_inference_steps * local_blend_steps[0]),
            int(num_inference_steps * local_blend_steps[1]),
        )

    def __call__(self, query, key, value, place_in_transformer: str):
        """Apply projection replacement (for SDPA mode)."""
        # Auto-detect CFG on first call (SD3 with guidance_scale > 1 doubles batch)
        if not hasattr(self, '_uses_cfg'):
            h = query.shape[0]
            expected_with_cfg = self.batch_size * 2 * self.num_heads
            self._uses_cfg = (h == expected_with_cfg) and (self.pipeline_type == "sd3")

        # Split conditional and unconditional for classifier-free guidance
        if self._uses_cfg:
            q_cond = query[query.shape[0] // 2:]
            k_cond = key[key.shape[0] // 2:]
            v_cond = value[value.shape[0] // 2:]
        else:
            q_cond, k_cond, v_cond = query, key, value

        h_ = q_cond.shape[0] // self.batch_size

        q_cond = q_cond.reshape(self.batch_size, h_, *q_cond.shape[1:])
        k_cond = k_cond.reshape(self.batch_size, h_, *k_cond.shape[1:])
        v_cond = v_cond.reshape(self.batch_size, h_, *v_cond.shape[1:])

        q_base, q_replace = q_cond[0], q_cond[1]
        k_base, k_replace = k_cond[0], k_cond[1]
        v_base, v_replace = v_cond[0], v_cond[1]

        block_idx = self.place_to_idx.get(place_in_transformer, -1)

        # Store attention for local blending
        should_store = (self.aggregate_blocks is None or block_idx in self.aggregate_blocks) and \
                       (self.local_blend is not None) and \
                       (self.local_blend_steps[0] <= self.cur_step < self.local_blend_steps[1])
        if should_store:
            self._store_attention(q_cond, k_cond, place_in_transformer)

        # Apply projection replacement
        should_replace_block = self.replace_blocks is None or block_idx in self.replace_blocks
        if self.replace_steps[0] <= self.cur_step < self.replace_steps[1] and should_replace_block:
            q_new, k_new, v_new = self._apply_projection_replacement(
                q_base, q_replace, k_base, k_replace, v_base, v_replace
            )
            q_cond[1] = q_new
            k_cond[1] = k_new
            v_cond[1] = v_new

        q_cond = q_cond.reshape(self.batch_size * h_, *q_cond.shape[2:])
        k_cond = k_cond.reshape(self.batch_size * h_, *k_cond.shape[2:])
        v_cond = v_cond.reshape(self.batch_size * h_, *v_cond.shape[2:])

        if self._uses_cfg:
            query[query.shape[0] // 2:] = q_cond
            key[key.shape[0] // 2:] = k_cond
            value[value.shape[0] // 2:] = v_cond
        else:
            query, key, value = q_cond, k_cond, v_cond

        self.cur_att_layer += 1
        return query, key, value

    def handle_attention(self, attn, value, place_in_transformer: str):
        """Apply attention region replacement (for non-SDPA mode).

        Called from attn_processors.py when use_sdpa=False (I2I, T2I, I2T, T2T methods).
        """
        # Auto-detect CFG on first call if not already done
        if not hasattr(self, '_uses_cfg'):
            h = attn.shape[0]
            expected_with_cfg = self.batch_size * 2 * self.num_heads
            self._uses_cfg = (h == expected_with_cfg) and (self.pipeline_type == "sd3")

        h = attn.shape[0]
        if self._uses_cfg:
            attn_cond = attn[h // 2:]
        else:
            attn_cond = attn

        block_idx = self.place_to_idx.get(place_in_transformer, -1)
        should_replace_block = self.replace_blocks is None or block_idx in self.replace_blocks

        if self.replace_steps[0] <= self.cur_step < self.replace_steps[1] and should_replace_block:
            h_ = attn_cond.shape[0] // self.batch_size
            attn_cond = attn_cond.reshape(self.batch_size, h_, *attn_cond.shape[1:])
            attn_base, attn_replace = attn_cond[0], attn_cond[1]

            attn_new = self._apply_attention_replacement(attn_base, attn_replace)
            attn_cond[1] = attn_new
            attn_cond = attn_cond.reshape(self.batch_size * h_, *attn_cond.shape[2:])

            if self._uses_cfg:
                attn[h // 2:] = attn_cond
            else:
                attn = attn_cond

        # Store attention for local blending 
        should_store = (self.aggregate_blocks is None or block_idx in self.aggregate_blocks) and \
                       (self.local_blend is not None) and \
                       (self.local_blend_steps[0] <= self.cur_step < self.local_blend_steps[1])
        if should_store:
            self._store_attention_from_matrix(attn.softmax(-1), place_in_transformer)

        self.cur_att_layer += 1
        return attn.softmax(-1), value

    def _apply_projection_replacement(self, q_base, q_replace, k_base, k_replace, v_base, v_replace):
        """Apply projection replacement based on method."""
        S_txt = self.text_seq_len
        S_img = q_base.shape[1] - S_txt

        if self.pipeline_type == "sd3":
            img_slice = slice(0, S_img)
            txt_slice = slice(S_img, None)
        else:
            txt_slice = slice(0, S_txt)
            img_slice = slice(S_txt, None)

        q_new = q_replace.clone()
        k_new = k_replace.clone()
        v_new = v_replace

        if self.method == "qk":
            q_new = q_base
            k_new = k_base
        elif self.method == "qiki":
            q_new[:, img_slice, :] = q_base[:, img_slice, :]
            k_new[:, img_slice, :] = k_base[:, img_slice, :]
        elif self.method == "qi":
            q_new[:, img_slice, :] = q_base[:, img_slice, :]
        elif self.method == "ki":
            k_new[:, img_slice, :] = k_base[:, img_slice, :]
        elif self.method == "qikt":
            q_new[:, img_slice, :] = q_base[:, img_slice, :]
            k_new[:, txt_slice, :] = k_base[:, txt_slice, :]
        elif self.method == "qtki":
            q_new[:, txt_slice, :] = q_base[:, txt_slice, :]
            k_new[:, img_slice, :] = k_base[:, img_slice, :]
        elif self.method == "qtkt":
            q_new[:, txt_slice, :] = q_base[:, txt_slice, :]
            k_new[:, txt_slice, :] = k_base[:, txt_slice, :]
        elif self.method == "kivi":
            k_new[:, img_slice, :] = k_base[:, img_slice, :]
            v_new = v_replace.clone()
            v_new[:, img_slice, :] = v_base[:, img_slice, :]

        return q_new, k_new, v_new

    def _apply_attention_replacement(self, attn_base, attn_replace):
        """Apply attention region replacement based on method."""
        S_txt = self.text_seq_len
        S_img = attn_base.shape[2] - S_txt
        result = attn_replace.clone()

        if self.pipeline_type == "sd3":
            # SD3: [image | text]
            if self.method == "I2I":
                result[:, :S_img, :S_img] = attn_base[:, :S_img, :S_img]
            elif self.method == "T2I":
                result[:, :S_img, S_img:] = attn_base[:, :S_img, S_img:]
            elif self.method == "I2T":
                result[:, S_img:, :S_img] = attn_base[:, S_img:, :S_img]
            elif self.method == "T2T":
                result[:, S_img:, S_img:] = attn_base[:, S_img:, S_img:]
        else:
            # Flux: [text | image]
            if self.method == "I2I":
                result[:, S_txt:, S_txt:] = attn_base[:, S_txt:, S_txt:]
            elif self.method == "T2I":
                result[:, S_txt:, :S_txt] = attn_base[:, S_txt:, :S_txt]
            elif self.method == "I2T":
                result[:, :S_txt, S_txt:] = attn_base[:, :S_txt, S_txt:]
            elif self.method == "T2T":
                result[:, :S_txt, :S_txt] = attn_base[:, :S_txt, :S_txt]

        return result

    def _store_attention(self, q_cond, k_cond, place_in_transformer: str):
        """
        Store attention maps for local blending (from Q/K projections).
        """
        if self.alpha_layers is None:
            return

        # Compute full attention scores
        head_dim = q_cond.shape[-1]
        scale = head_dim ** -0.5

        # q_cond, k_cond: (batch, heads, seq_len, dim)
        # Full attention: (batch, heads, seq_len, seq_len)
        attn = torch.matmul(q_cond, k_cond.transpose(-2, -1)) * scale
        attn = attn.softmax(-1)

        # Extract attention to selected text tokens for local blending
        S_img = attn.shape[2] - self.text_seq_len
        attn = attn.mean(1)  # Average over heads: (batch, seq_len, seq_len)

        local_blend_store = []
        for i in range(self.batch_size):
            batch_indices = torch.where(self.alpha_layers[i])[0]
            if len(batch_indices) == 0:
                local_blend_store.append(torch.zeros(S_img, device=attn.device, dtype=attn.dtype))
                continue
            if self.pipeline_type == "sd3":
                col_indices = batch_indices + S_img  # Text columns
                local_blend_store.append(attn[i, :, col_indices].mean(-1)[:S_img])  # Image rows only
            else:
                col_indices = batch_indices  # Text columns (at start for Flux)
                local_blend_store.append(attn[i, :, col_indices].mean(-1)[self.text_seq_len:])  # Image rows only

        local_blend_store = torch.stack(local_blend_store).unsqueeze(0)  # (1, batch, S_img)

        if place_in_transformer not in self.attention_store:
            self.attention_store[place_in_transformer] = local_blend_store
        else:
            self.attention_store[place_in_transformer] = torch.cat(
                [self.attention_store[place_in_transformer], local_blend_store], dim=0
            )

    def _store_attention_from_matrix(self, attn, place_in_transformer: str):
        """Store attention maps for local blending (from attention matrix)."""
        if self.alpha_layers is None:
            return

        S_img = attn.shape[2] - self.text_seq_len
        attn = attn.reshape(self.batch_size, attn.shape[0] // self.batch_size, *attn.shape[1:])
        attn = attn.mean(1)  # Average over heads

        local_blend_store = []
        for i in range(self.batch_size):
            batch_indices = torch.where(self.alpha_layers[i])[0]
            if self.pipeline_type == "sd3":
                col_indices = batch_indices + S_img
                local_blend_store.append(attn[i, :, col_indices].mean(-1)[:S_img])
            else:
                col_indices = batch_indices
                local_blend_store.append(attn[i, :, col_indices].mean(-1)[self.text_seq_len:])

        local_blend_store = torch.stack(local_blend_store).unsqueeze(0)

        if place_in_transformer not in self.attention_store:
            self.attention_store[place_in_transformer] = local_blend_store
        else:
            self.attention_store[place_in_transformer] = torch.cat(
                [self.attention_store[place_in_transformer], local_blend_store], dim=0
            )

    def aggregate_attention(self, to_cpu=True):
        """
        Aggregate attention from stored blocks.

        - Iterates over all stored blocks (storage already filtered by aggregate_blocks)
        - Uses last N steps (local_blend_n_prev_steps)
        - Concatenates across blocks and time, then averages

        Returns:
            Tensor of shape (2, S_img) - attention maps for both batch items.
        """
        if not self.attention_store:
            return None

        # Collect from all stored blocks (storage already filtered by aggregate_blocks)
        # Both _store_attention and _store_attention_from_matrix produce (T, batch, S_img)
        b1_list, b2_list = [], []
        for key, val in self.attention_store.items():
            # Use last N steps: val[-self.local_blend_n_prev_steps:]
            val = val[-self.local_blend_n_prev_steps:]
            if to_cpu:
                val = val.cpu()
            # val shape: (T, batch, S_img)
            b1_list.append(val[:, 0, :])  # (T, S_img)
            b2_list.append(val[:, 1, :])  # (T, S_img)

        # Concatenate across blocks and time, then average
        b1_mask = torch.cat(b1_list, dim=0).mean(dim=0, keepdim=True)  # (1, S_img)
        b2_mask = torch.cat(b2_list, dim=0).mean(dim=0, keepdim=True)  # (1, S_img)

        return torch.cat([b1_mask, b2_mask], dim=0)  # (2, S_img)

    def step_callback(self, x_t):
        """
        Apply local blending during specified steps (η in Algorithm 1).
        Step increment happens here AFTER local blend.
        """
        if self.local_blend is not None and self.local_blend_steps[0] <= self.cur_step < self.local_blend_steps[1]:
            attn = self.aggregate_attention(to_cpu=False)
            if attn is not None:
                # Storage already produces image-only data (S_img), no slicing needed
                # Apply gaussian smoothing
                attn = gaussian_smoothing(attn, kernel_size=7, sigma=1.0, dim=(self.height // 16, self.width // 16))
                x_t = self.local_blend(x_t, attn)

        # Step increment happens AFTER local blend
        if self.cur_att_layer == self.num_att_layer:
            self.cur_att_layer = 0
            self.cur_step += 1
        return x_t

    def reset(self):
        """Reset controller state for new generation."""
        self.cur_step = 0
        self.cur_att_layer = 0
        self.attention_store = {}


class LocalBlend:
    """Local blending based on attention maps."""

    def __init__(
        self,
        pipeline_type: str,
        prompts: List[str],
        words: List,
        tokenizer,
        device,
        threshold: float = 0.3,
        t5_max_seq_len: int = 256,
        tokenizer_choice: str = "both",
    ):
        self.pipeline_type = pipeline_type
        self.threshold = threshold
        self.t5_max_seq_len = t5_max_seq_len
        self.tokenizer_choice = tokenizer_choice

        if pipeline_type == "sd3":
            alpha_layers1 = torch.zeros(len(prompts), 77, device=device)
            alpha_layers3 = torch.zeros(len(prompts), t5_max_seq_len, device=device)

            for i, (prompt, words_) in enumerate(zip(prompts, words)):
                if isinstance(words_, str):
                    words_ = [words_]
                for word in words_:
                    if tokenizer_choice in ["both", "clip"]:
                        ind1 = get_multi_word_inds(prompt, word, tokenizer[0])
                        alpha_layers1[i, ind1] = 1
                    if tokenizer_choice in ["both", "t5"]:
                        ind3 = get_multi_word_inds(prompt, word, tokenizer[2])
                        alpha_layers3[i, ind3] = 1

            if tokenizer_choice == "both":
                self.alpha_layers = torch.cat([alpha_layers1, alpha_layers3], dim=-1)
            elif tokenizer_choice == "clip":
                self.alpha_layers = torch.cat([alpha_layers1, torch.zeros_like(alpha_layers3)], dim=-1)
            else:
                self.alpha_layers = torch.cat([torch.zeros_like(alpha_layers1), alpha_layers3], dim=-1)

            self.text_seq_len = 77 + t5_max_seq_len
        else:
            alpha_layers = torch.zeros(len(prompts), t5_max_seq_len, device=device)
            for i, (prompt, words_) in enumerate(zip(prompts, words)):
                if isinstance(words_, str):
                    words_ = [words_]
                for word in words_:
                    ind = get_multi_word_inds(prompt, word, tokenizer[1])
                    alpha_layers[i, ind] = 1
            self.alpha_layers = alpha_layers
            self.text_seq_len = t5_max_seq_len

    def __call__(self, x_t, attention_store):
        """Apply local blending mask to latents."""
        if attention_store is None:
            return x_t

        if self.pipeline_type == "sd3":
            attention_store = attention_store.view(
                x_t.shape[0], 1, x_t.shape[2] // 2, x_t.shape[3] // 2
            )
            mask = F.max_pool2d(attention_store, (3, 3), stride=1, padding=1)
            mask = F.interpolate(mask, size=x_t.shape[2:])
            mask = mask / mask.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
            mask = mask.gt(self.threshold)
            mask = (mask[:1] + mask[1:]).to(x_t.dtype)
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        else:
            # Flux: latent is (batch, seq_len, dim), just normalize and threshold
            mask = attention_store / attention_store.max(dim=-1, keepdim=True)[0]
            mask = mask.gt(self.threshold)
            mask = mask.unsqueeze(-1)
            mask = (mask[:1] + mask[1:]).to(x_t.dtype)
            x_t = x_t[:1] + mask * (x_t - x_t[:1])

        return x_t


def p2p_callback(pipe, step, timestep, callback_kwargs):
    """Callback for diffusers pipeline to apply step-wise blending."""
    callback_kwargs["latents"] = pipe.controller.step_callback(callback_kwargs["latents"])
    return callback_kwargs
