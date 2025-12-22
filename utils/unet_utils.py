"""
U-Net utilities for prompt-based image editing (P2P).

This module implements the original Prompt-to-Prompt method for
U-Net based models (e.g., SD 1.x, SDXL).
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

from .common import (
    get_word_inds,
    get_refinement_mapper,
    get_equalizer,
)
from .attn_processors import P2P_SDAttnProcessor


def register_attention_control(pipeline, controller):
    """
    Register P2P attention control with U-Net pipeline.

    Args:
        pipeline: StableDiffusionPipeline instance
        controller: AttentionControl instance
    """
    attn_procs = {}
    att_count = 0

    for name in pipeline.unet.attn_processors.keys():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue

        att_count += 1
        attn_procs[name] = P2P_SDAttnProcessor(
            controller=controller, place_in_unet=place_in_unet
        )

    pipeline.controller = controller
    pipeline.unet.set_attn_processor(attn_procs)
    controller.num_att_layer = att_count
    print(f"Registered {att_count} attention processors.")


def _get_time_words_attention_alpha(
    prompts, num_steps, cross_replace_steps, tokenizer, max_num_words=77
):
    """Generate alpha tensor for time-word attention control."""
    if not isinstance(cross_replace_steps, dict):
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0.0, 1.0)

    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)

    for i in range(len(prompts) - 1):
        bounds = cross_replace_steps["default_"]
        if isinstance(bounds, float):
            bounds = (0.0, bounds)
        start, end = int(bounds[0] * num_steps), int(bounds[1] * num_steps)
        alpha_time_words[start:end, i, :] = 1

    for key, bounds in cross_replace_steps.items():
        if key != "default_":
            if isinstance(bounds, float):
                bounds = (0.0, bounds)
            start, end = int(bounds[0] * num_steps), int(bounds[1] * num_steps)
            inds = [
                get_word_inds(prompts[i + 1], key, tokenizer)
                for i in range(len(prompts) - 1)
            ]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words[:start, i, ind] = 0
                    alpha_time_words[start:end, i, ind] = 1
                    alpha_time_words[end:, i, ind] = 0

    alpha_time_words = alpha_time_words.reshape(
        num_steps + 1, len(prompts) - 1, 1, 1, max_num_words
    )
    return alpha_time_words


def _get_replacement_mapper(prompts, tokenizer, max_len=77):
    """Create mapper for attention replacement between prompts."""
    x_prompt = prompts[0]
    mappers = []
    for y_prompt in prompts[1:]:
        mapper = _get_replacement_mapper_single(x_prompt, y_prompt, tokenizer, max_len)
        mappers.append(mapper)
    return torch.stack(mappers)


def _get_replacement_mapper_single(x: str, y: str, tokenizer, max_len=77):
    """Create replacement mapper between two prompts."""
    import numpy as np

    words_x = x.strip().split()
    words_y = y.strip().split()
    if len(words_x) != len(words_y):
        raise ValueError(
            "Prompts must have the same number of words for 'replace' edit."
        )

    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]
    inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]

    mapper = np.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0
    while i < max_len and j < max_len:
        if (
            cur_inds < len(inds_source)
            and len(inds_source[cur_inds]) > 0
            and inds_source[cur_inds][0] == i
        ):
            inds_source_ = inds_source[cur_inds]
            inds_target_ = inds_target[cur_inds]
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio
            cur_inds += 1
            i += len(inds_source_)
            j += len(inds_target_)
        else:
            mapper[i, j] = 1
            i += 1
            j += 1
    return torch.from_numpy(mapper).float()


def create_controller(
    prompts: List[str],
    num_inference_steps: int,
    tokenizer,
    device,
    dtype,
    edit_type: str = "replace",
    n_cross_replace: float = 0.4,
    n_self_replace: float = 0.4,
    local_blend_words: Optional[List] = None,
    local_blend_threshold: float = 0.5,
    attn_res: int = 16,
    equalizer_words: Optional[List] = None,
    equalizer_strengths: Optional[List] = None,
) -> "AttentionControl":
    """
    Create attention controller for U-Net P2P editing.

    Args:
        prompts: List of [source_prompt, target_prompt]
        num_inference_steps: Number of diffusion steps
        tokenizer: Tokenizer for text encoding
        device: Computation device
        dtype: Data type
        edit_type: "replace", "refine", or "reweight"
        n_cross_replace: Cross-attention replacement steps fraction
        n_self_replace: Self-attention replacement steps fraction
        local_blend_words: Words for local blending
        local_blend_threshold: Threshold for local blend mask
        attn_res: Resolution of attention maps for self-attention
        equalizer_words: Words for reweighting (required for reweight)
        equalizer_strengths: Strengths for reweighting (required for reweight)

    Returns:
        AttentionControl instance
    """
    if edit_type not in ["replace", "refine", "reweight"]:
        raise ValueError(f"Edit type '{edit_type}' not recognized.")

    attn_res = (attn_res, attn_res) if isinstance(attn_res, int) else attn_res

    local_blend = None
    if local_blend_words is not None:
        local_blend = LocalBlend(
            prompts,
            local_blend_words,
            tokenizer=tokenizer,
            device=device,
            attn_res=attn_res,
            threshold=local_blend_threshold,
        )

    controller = AttentionControl(
        prompts,
        num_inference_steps,
        n_cross_replace,
        n_self_replace,
        edit_type=edit_type,
        local_blend=local_blend,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        attn_res=attn_res,
        equalizer_words=equalizer_words,
        equalizer_strengths=equalizer_strengths,
    )

    return controller


class AttentionControl:
    """
    Attention control for Prompt-to-Prompt editing in U-Net models.

    Handles replace, refine, and reweight edit types.
    """

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: Union[float, Tuple[float, float]],
        self_replace_steps: Union[float, Tuple[float, float]],
        edit_type: str,
        local_blend: Optional["LocalBlend"],
        tokenizer,
        device,
        dtype,
        attn_res,
        equalizer_words: Optional[List] = None,
        equalizer_strengths: Optional[List] = None,
    ):
        self.prompts = prompts
        self.num_steps = num_steps
        self.edit_type = edit_type
        self.local_blend = local_blend
        self.tokenizer = tokenizer
        self.device = device
        self.attn_res = attn_res
        self.alpha_layers = (
            local_blend.alpha_layers if local_blend is not None else None
        )

        self.cur_step = 0
        self.cur_att_layer = 0
        self.num_att_layer = -1
        self.num_uncond_att_layers = 0
        self.batch_size = len(prompts)
        self.step_store = self._get_empty_store()
        self.attention_store = {}

        # Cross-attention alpha
        self.cross_replace_alpha = _get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, tokenizer
        ).to(device, dtype=dtype)

        # Self-attention replacement range
        if isinstance(self_replace_steps, float):
            self_replace_steps = (0.0, self_replace_steps)
        self.num_self_replace = (
            int(num_steps * self_replace_steps[0]),
            int(num_steps * self_replace_steps[1]),
        )

        # Edit-type specific setup
        if edit_type == "replace":
            self.mapper = _get_replacement_mapper(prompts, tokenizer).to(
                device, dtype=dtype
            )
        elif edit_type == "refine":
            self.mapper, alphas = get_refinement_mapper(prompts, tokenizer)
            self.mapper = self.mapper.to(device)
            self.alphas = alphas.to(device, dtype=dtype).reshape(
                alphas.shape[0], 1, 1, alphas.shape[1]
            )
        elif edit_type == "reweight":
            if equalizer_words is None or equalizer_strengths is None:
                raise ValueError(
                    "For 'reweight' edit, specify 'equalizer_words' and 'equalizer_strengths'."
                )
            self.equalizer = get_equalizer(
                prompts[1], equalizer_words, equalizer_strengths, tokenizer
            ).to(device)

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        """Apply attention control."""
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn_uncond, attn_cond = attn[: h // 2], attn[h // 2 :]

            # Store before modification
            self._store_attention(attn_cond, is_cross, place_in_unet)

            if is_cross or (
                self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]
            ):
                h_ = attn_cond.shape[0] // self.batch_size
                attn_cond = attn_cond.reshape(self.batch_size, h_, *attn_cond.shape[1:])
                attn_base, attn_replace = attn_cond[0], attn_cond[1:]

                if is_cross:
                    alpha_words = self.cross_replace_alpha[self.cur_step]
                    if self.edit_type == "replace":
                        attn_replace_new = (
                            self._replace_cross_attention(attn_base, attn_replace)
                            * alpha_words
                            + (1 - alpha_words) * attn_replace
                        )
                    elif self.edit_type == "refine":
                        attn_replace_new = (
                            self._refine_cross_attention(attn_base, attn_replace)
                            * alpha_words
                            + (1 - alpha_words) * attn_replace
                        )
                    elif self.edit_type == "reweight":
                        attn_replace_new = (
                            self._reweight_cross_attention(attn_base, attn_replace)
                            * alpha_words
                            + (1 - alpha_words) * attn_replace
                        )
                    attn_cond[1:] = attn_replace_new
                else:
                    attn_cond[1:] = self._replace_self_attention(
                        attn_base, attn_replace
                    )

                attn_cond = attn_cond.reshape(
                    self.batch_size * h_, *attn_cond.shape[2:]
                )

            attn[h // 2 :] = attn_cond

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layer + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self._between_steps()

        return attn

    def _replace_cross_attention(self, attn_base, attn_replace):
        return torch.einsum("hpw,bwn->bhpn", attn_base, self.mapper)

    def _refine_cross_attention(self, attn_base, attn_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        return attn_base_replace * self.alphas + attn_replace * (1 - self.alphas)

    def _reweight_cross_attention(self, attn_base, attn_replace):
        return attn_base[None, :, :, :] * self.equalizer[:, None, None, :]

    def _replace_self_attention(self, attn_base, attn_replace):
        if attn_replace.shape[2] <= self.attn_res[0] * self.attn_res[1]:
            return attn_base.unsqueeze(0).expand(
                attn_replace.shape[0], *attn_base.shape
            )
        return attn_replace

    def _store_attention(self, attn, is_cross: bool, place_in_unet: str):
        """Store attention maps for local blending."""
        if is_cross and attn.shape[1] == self.attn_res[0] * self.attn_res[1]:
            if self.alpha_layers is not None:
                attn = attn.reshape(
                    self.batch_size,
                    attn.shape[0] // self.batch_size,
                    attn.shape[1],
                    attn.shape[2],
                )
                alpha_layers = self.alpha_layers[:, None, None, :]
                masked_attn = (attn * alpha_layers).sum(-1)
                key = f"{place_in_unet}_cross"
                if self.step_store[key] is None:
                    self.step_store[key] = masked_attn
                else:
                    self.step_store[key] += masked_attn

    def _between_steps(self):
        """Update attention store between steps."""
        if not self.attention_store:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                if self.step_store[key] is not None:
                    self.attention_store[key] += self.step_store[key]
        self.step_store = self._get_empty_store()

    def step_callback(self, x_t):
        """Apply local blending at each step."""
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def reset(self):
        """Reset controller state."""
        self.cur_step = 0
        self.cur_att_layer = 0
        self.step_store = self._get_empty_store()
        self.attention_store = {}

    def _get_empty_store(self):
        return {"down_cross": None, "mid_cross": None, "up_cross": None}


class LocalBlend:
    """Local blending for U-Net based models."""

    def __init__(
        self,
        prompts: List[str],
        words: List,
        tokenizer,
        device,
        attn_res,
        threshold: float = 0.5,
        max_num_words: int = 77,
    ):
        self.max_num_words = max_num_words
        self.device = device
        self.threshold = threshold
        self.attn_res = attn_res

        alpha_layers = torch.zeros(len(prompts), max_num_words, device=device)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if isinstance(words_, str):
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, ind] = 1
        self.alpha_layers = alpha_layers

    def __call__(self, x_t, attention_store):
        """Apply local blending mask to latents."""
        maps = [
            attention_store[key]
            for key in ["down_cross", "mid_cross", "up_cross"]
            if attention_store[key] is not None
        ]
        if not maps:
            return x_t

        maps = torch.cat(maps, dim=1)
        maps = maps.view(
            self.alpha_layers.shape[0], -1, 1, self.attn_res[0], self.attn_res[1]
        )
        maps = maps.mean(1)

        k = 1
        mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), stride=1, padding=k)
        mask = F.interpolate(mask, size=x_t.shape[2:])
        mask = mask / mask.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).to(x_t.dtype)
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t


def p2p_callback(pipe, step, timestep, callback_kwargs):
    """Callback for diffusers pipeline."""
    callback_kwargs["latents"] = pipe.controller.step_callback(
        callback_kwargs["latents"]
    )
    return callback_kwargs
