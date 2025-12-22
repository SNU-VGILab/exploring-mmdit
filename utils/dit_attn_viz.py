"""
Attention visualization utilities for MM-DiT models.

This module provides:
- AttentionInspector: Lightweight controller for capturing T2I and I2T attention
- Visualization functions for per-block and per-timestep attention maps
- Word-level attention extraction

For block-level evaluation (finding top-N blocks), see the commented GSAM2 section below.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
from diffusers.models.attention_processor import JointAttnProcessor2_0

from .attn_processors import InspectJointAttnProcessor, InspectFluxAttnProcessor
from .common import get_multi_word_inds


def register_attention_inspector(pipeline, inspector):
    """Register AttentionInspector with a pipeline."""
    from diffusers import StableDiffusion3Pipeline, FluxPipeline

    attn_procs = {}
    att_count = 0

    for name in pipeline.transformer.attn_processors.keys():
        if "attn2" in name:
            attn_procs[name] = JointAttnProcessor2_0()
            continue
        att_count += 1
        place_in_transformer = name.split(".attn.")[0]

        # Use model-specific inspection processor
        if isinstance(pipeline, StableDiffusion3Pipeline):
            attn_procs[name] = InspectJointAttnProcessor(inspector, place_in_transformer)
        else:
            attn_procs[name] = InspectFluxAttnProcessor(inspector, place_in_transformer)

    pipeline.controller = inspector
    pipeline.transformer.set_attn_processor(attn_procs)
    inspector.num_att_layer = att_count


def create_attention_inspector(
    pipeline,
    prompt: str,
    num_inference_steps: int = 28,
    t5_max_seq_len: Optional[int] = None,
):
    """
    Create an AttentionInspector for visualization.

    Args:
        pipeline: SD3 or Flux pipeline
        prompt: Single prompt for generation
        num_inference_steps: Number of denoising steps
        t5_max_seq_len: T5 sequence length (auto-detected if None)

    Returns:
        AttentionInspector instance
    """
    from diffusers import StableDiffusion3Pipeline, FluxPipeline

    if isinstance(pipeline, StableDiffusion3Pipeline):
        pipeline_type = "sd3"
        tokenizer = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
        if t5_max_seq_len is None:
            t5_max_seq_len = 256
    elif isinstance(pipeline, FluxPipeline):
        pipeline_type = "flux"
        tokenizer = [None, pipeline.tokenizer_2, None]
        if t5_max_seq_len is None:
            t5_max_seq_len = 512
    else:
        raise ValueError(f"Unsupported pipeline type: {type(pipeline)}")

    inspector = AttentionInspector(
        pipeline_type=pipeline_type,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        tokenizer=tokenizer,
        t5_max_seq_len=t5_max_seq_len,
    )
    return inspector


class AttentionInspector:
    """
    Lightweight controller for capturing attention maps (no editing).

    Stores both T2I (image queries attending to text) and I2T (text queries attending to image).
    Called by InspectJointAttnProcessor / InspectFluxAttnProcessor.
    """

    def __init__(
        self,
        pipeline_type: str,
        prompt: str,
        num_inference_steps: int,
        tokenizer: List[Any],
        t5_max_seq_len: int,
    ):
        self.pipeline_type = pipeline_type
        self.prompt = prompt
        self.num_inference_steps = num_inference_steps
        self.tokenizer = tokenizer
        self.t5_max_seq_len = t5_max_seq_len
        self.dtype = torch.float32

        self.text_seq_len = 77 + t5_max_seq_len if pipeline_type == "sd3" else t5_max_seq_len

        self.cur_step = 0
        self.cur_att_layer = 0
        self.num_att_layer = -1

        # Store T2I and I2T separately
        self.T2I = {}  # Image attending to text
        self.I2T = {}  # Text attending to image
        self.T2I_cpu = {}
        self.I2T_cpu = {}

    def __call__(self, attention_scores, place_in_transformer: str):
        """
        Called by attention processors with raw attention scores.
        Applies softmax, stores T2I/I2T regions, returns attention_probs.
        """
        self.dtype = attention_scores.dtype
        attention_probs = attention_scores.softmax(-1).to(self.dtype)
        self.store_attention(attention_probs, place_in_transformer)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layer:
            self.cur_att_layer = 0
            self.cur_step += 1
        return attention_probs

    def store_attention(self, attn, place_in_transformer: str):
        """Store T2I and I2T attention regions."""
        # attn shape: (batch*heads, seq, seq)
        # Infer batch_size (should be 1 for visualization)
        batch_size = 1
        num_heads = attn.shape[0] // batch_size

        S_txt = self.text_seq_len
        S_img = attn.shape[1] - S_txt

        # Reshape and average over heads: (batch, seq, seq)
        attn = attn.view(batch_size, num_heads, attn.shape[1], attn.shape[2]).mean(1)

        if self.pipeline_type == "sd3":
            # SD3: [img | txt] ordering
            T2I = attn[:, :S_img, S_img:]   # (batch, S_img, S_txt)
            I2T = attn[:, S_img:, :S_img]   # (batch, S_txt, S_img)
        else:
            # Flux: [txt | img] ordering
            T2I = attn[:, S_txt:, :S_txt]   # (batch, S_img, S_txt)
            I2T = attn[:, :S_txt, S_txt:]   # (batch, S_txt, S_img)

        if place_in_transformer not in self.T2I:
            self.T2I[place_in_transformer] = T2I
            self.I2T[place_in_transformer] = I2T
        else:
            self.T2I[place_in_transformer] = torch.cat([self.T2I[place_in_transformer], T2I], dim=0)
            self.I2T[place_in_transformer] = torch.cat([self.I2T[place_in_transformer], I2T], dim=0)

    def step_callback(self, x_t):
        """Move attention to CPU periodically to save memory."""
        if self.cur_step == self.num_inference_steps // 2:
            self.T2I_cpu = {k: v.cpu() for k, v in self.T2I.items()}
            self.I2T_cpu = {k: v.cpu() for k, v in self.I2T.items()}
            self.T2I, self.I2T = {}, {}

        if self.cur_step == self.num_inference_steps:
            self.T2I_cpu = {k: torch.cat([self.T2I_cpu[k], v.cpu()], dim=0) for k, v in self.T2I.items()}
            self.I2T_cpu = {k: torch.cat([self.I2T_cpu[k], v.cpu()], dim=0) for k, v in self.I2T.items()}

        return x_t

    def get_attn_map(self, timestep: int, block_id: int, region: str = "T2I"):
        """
        Get attention map for a specific timestep and block.

        Args:
            timestep: Timestep index (0 to num_inference_steps-1)
            block_id: Block index
            region: "T2I" or "I2T"

        Returns:
            Tensor of shape (1, S_img, S_txt) for T2I or (1, S_txt, S_img) for I2T
        """
        if self.pipeline_type == "sd3":
            name = f"transformer_blocks.{block_id}"
        else:
            if block_id >= 19:
                name = f"single_transformer_blocks.{block_id - 19}"
            else:
                name = f"transformer_blocks.{block_id}"

        store = self.T2I_cpu if region == "T2I" else self.I2T_cpu
        return store[name][timestep].unsqueeze(0)

    def aggregate_attention(
        self,
        timesteps: Optional[List[int]] = None,
        block_ids: Optional[List[int]] = None,
        region: str = "T2I",
    ):
        """
        Aggregate attention across timesteps and blocks.

        Args:
            timesteps: List of timestep indices (None = all)
            block_ids: List of block indices (None = all)
            region: "T2I" or "I2T"

        Returns:
            Tensor of shape (1, S_img, S_txt) for T2I or (1, S_txt, S_img) for I2T
        """
        store = self.T2I_cpu if region == "T2I" else self.I2T_cpu

        if timesteps is None:
            timesteps = list(range(self.num_inference_steps))
        if block_ids is None:
            block_ids = list(range(self.num_att_layer))

        attn_maps = []
        for t in timesteps:
            for block_id in block_ids:
                attn_maps.append(self.get_attn_map(t, block_id, region))

        return torch.cat(attn_maps, dim=0).mean(dim=0, keepdim=True)

    def get_word_attention(self, attn, word: str, tokenizer_choice: str = "both"):
        """
        Extract attention for a specific word.

        Args:
            attn: Attention tensor from get_attn_map or aggregate_attention
            word: Word to extract attention for
            tokenizer_choice: "both", "clip", or "t5" (SD3 only)

        Returns:
            Tensor of shape (batch, S_img) - attention map for the word
        """
        is_t2i = attn.shape[1] > attn.shape[2]  # T2I has more rows (S_img > S_txt)

        if self.pipeline_type == "sd3":
            if tokenizer_choice == "clip":
                inds = get_multi_word_inds(self.prompt, word, self.tokenizer[0])
            elif tokenizer_choice == "t5":
                inds = get_multi_word_inds(self.prompt, word, self.tokenizer[2])
                inds = inds + 77
            else:
                ind1 = get_multi_word_inds(self.prompt, word, self.tokenizer[0])
                ind3 = get_multi_word_inds(self.prompt, word, self.tokenizer[2])
                ind3 = ind3 + 77
                inds = np.concatenate([ind1, ind3])
        else:
            inds = get_multi_word_inds(self.prompt, word, self.tokenizer[1])

        if is_t2i:
            return attn[:, :, inds].mean(dim=2)  # (batch, S_img)
        else:
            return attn[:, inds, :].mean(dim=1)  # (batch, S_img)

    def reset(self):
        """Reset for new generation."""
        self.cur_step = 0
        self.cur_att_layer = 0
        self.T2I = {}
        self.I2T = {}
        self.T2I_cpu = {}
        self.I2T_cpu = {}


def inspector_callback(pipe, step, timestep, callback_kwargs):
    """Callback for diffusers pipeline."""
    callback_kwargs["latents"] = pipe.controller.step_callback(callback_kwargs["latents"])
    return callback_kwargs


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_attention_grid(
    inspector: AttentionInspector,
    word: str,
    region: str = "T2I",
    image_size: Tuple[int, int] = (1024, 1024),
    tokenizer_choice: str = "both",
    cmap: str = "viridis",
    save_path: Optional[str] = None,
):
    """
    Visualize attention maps per-block and per-timestep (Fig. A14 style).

    Args:
        inspector: AttentionInspector with stored attention
        word: Word to visualize attention for
        region: "T2I" or "I2T"
        image_size: (height, width) of generated image
        tokenizer_choice: "both", "clip", or "t5" for SD3
        cmap: Matplotlib colormap
        save_path: If provided, saves figures

    Returns:
        Tuple of (fig_blocks, fig_timesteps) matplotlib figures
    """
    h, w = image_size[0] // 16, image_size[1] // 16
    num_blocks = inspector.num_att_layer
    num_steps = inspector.num_inference_steps

    # Per-block: average across timesteps
    block_maps = []
    for block_id in range(num_blocks):
        attn = inspector.aggregate_attention(
            timesteps=list(range(num_steps)),
            block_ids=[block_id],
            region=region,
        )
        word_attn = inspector.get_word_attention(attn, word, tokenizer_choice)
        block_maps.append(word_attn.squeeze().float().numpy())

    # Per-timestep: average across blocks
    time_maps = []
    for t in range(num_steps):
        attn = inspector.aggregate_attention(
            timesteps=[t],
            block_ids=list(range(num_blocks)),
            region=region,
        )
        word_attn = inspector.get_word_attention(attn, word, tokenizer_choice)
        time_maps.append(word_attn.squeeze().float().numpy())

    def normalize(x):
        x_min, x_max = x.min(), x.max()
        return (x - x_min) / (x_max - x_min + 1e-6)

    # Figure 1: Per-block
    cols = 5
    rows = (num_blocks + cols - 1) // cols
    fig1, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    axes = axes.flatten() if num_blocks > 1 else [axes]
    for idx in range(rows * cols):
        ax = axes[idx]
        if idx < num_blocks:
            ax.imshow(normalize(block_maps[idx]).reshape(h, w), cmap=cmap)
            ax.set_title(f"Block {idx}", fontsize=8)
        ax.axis("off")
    fig1.suptitle(f"Per-block {region} attention for '{word}'", fontsize=10)
    plt.tight_layout()

    # Figure 2: Per-timestep
    rows = (num_steps + cols - 1) // cols
    fig2, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    axes = axes.flatten() if num_steps > 1 else [axes]
    for idx in range(rows * cols):
        ax = axes[idx]
        if idx < num_steps:
            ax.imshow(normalize(time_maps[idx]).reshape(h, w), cmap=cmap)
            ax.set_title(f"Step {idx}", fontsize=8)
        ax.axis("off")
    fig2.suptitle(f"Per-timestep {region} attention for '{word}'", fontsize=10)
    plt.tight_layout()

    if save_path:
        fig1.savefig(save_path.replace(".png", f"_{region}_blocks.png"), dpi=150, bbox_inches="tight")
        fig2.savefig(save_path.replace(".png", f"_{region}_timesteps.png"), dpi=150, bbox_inches="tight")

    return fig1, fig2


def visualize_attention_single(
    attention: torch.Tensor,
    image_size: Tuple[int, int] = (1024, 1024),
    title: str = "Attention",
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
):
    """
    Visualize a single attention map.

    Args:
        attention: 1D or 2D tensor (will be flattened)
        image_size: (height, width) of generated image
        title: Plot title
        cmap: Matplotlib colormap
        ax: Matplotlib axes (creates new figure if None)

    Returns:
        Matplotlib axes
    """
    if isinstance(attention, torch.Tensor):
        attention = attention.cpu().float().numpy()

    attention = attention.flatten()
    att_min, att_max = attention.min(), attention.max()
    attention = (attention - att_min) / (att_max - att_min + 1e-6)

    h, w = image_size[0] // 16, image_size[1] // 16
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(attention.reshape(h, w), cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    return ax


# =============================================================================
# GSAM2 Block Evaluation (Reference Implementation)
# =============================================================================
#
# The following code was used to evaluate block-level attention quality
# by comparing attention maps against ground-truth segmentation from GSAM2.
# This requires setting up the Grounded-SAM-2 environment:
# https://github.com/IDEA-Research/Grounded-SAM-2
#
# Note: The inspector.num_att_layer is set after pipeline runs, so you must
# run inference first to populate the attention maps before using this code.
#
# The prompts in data/block-inspect.json are curated samples from PartiPrompts
# that produce reasonable attention maps (indices correspond to PartiPrompts).
#
# To use this code:
# 1. Install Grounded-SAM-2 following their instructions
# 2. Uncomment the code below
# 3. Update DEFAULT_PARAMS with your checkpoint paths
#
# DEFAULT_PARAMS = {
#     "grounding_model": "IDEA-Research/grounding-dino-base",
#     "sam2_checkpoint": "/path/to/sam2.1_hiera_large.pt",
#     "sam2_model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
# }
#
#
# class GSAM2:
#     """Grounded SAM2 for generating ground-truth segmentation masks."""
#
#     def __init__(self, device="cuda"):
#         from sam2.build_sam import build_sam2
#         from sam2.sam2_image_predictor import SAM2ImagePredictor
#         from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
#
#         self.device = device
#         self.sam2_model = build_sam2(
#             DEFAULT_PARAMS["sam2_model_config"],
#             DEFAULT_PARAMS["sam2_checkpoint"],
#             device=self.device
#         )
#         self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
#         self.processor = AutoProcessor.from_pretrained(DEFAULT_PARAMS["grounding_model"])
#         self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
#             DEFAULT_PARAMS["grounding_model"]
#         ).to(self.device)
#
#     def __call__(self, image, text, res=None, flatten=False):
#         """Generate segmentation mask for text query."""
#         with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
#             text = text + "."
#             image = image.convert("RGB")
#             self.sam2_predictor.set_image(np.array(image))
#             inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
#
#             with torch.no_grad():
#                 outputs = self.grounding_model(**inputs)
#
#             results = self.processor.post_process_grounded_object_detection(
#                 outputs, inputs.input_ids,
#                 box_threshold=0.4, text_threshold=0.3,
#                 target_sizes=[image.size[::-1]],
#             )
#             input_boxes = results[0]["boxes"].cpu().numpy()
#             masks, scores, logits = self.sam2_predictor.predict(
#                 point_coords=None, point_labels=None,
#                 box=input_boxes, multimask_output=False,
#             )
#             if masks.ndim == 4:
#                 masks = masks.squeeze(1)
#             masks = torch.tensor(masks).sum(0, keepdim=True).clamp(0, 1)
#             if res is not None:
#                 masks = F.interpolate(masks.unsqueeze(0), size=res, mode="nearest").squeeze(0)
#             if flatten:
#                 masks = masks.flatten(start_dim=1)
#         return masks
#
#
# def evaluate_attention_map(attention_map, gt_seg, epsilon=1e-7):
#     """Evaluate attention map against ground-truth segmentation."""
#     att_normalized = attention_map / (torch.max(attention_map) + epsilon)
#
#     bce_loss = F.binary_cross_entropy(att_normalized, gt_seg.float(), reduction="mean")
#     intersection = torch.sum(att_normalized * gt_seg)
#     union = torch.sum(att_normalized) + torch.sum(gt_seg) - intersection
#     soft_iou = intersection / (union + epsilon)
#     mse = F.mse_loss(att_normalized, gt_seg.float())
#
#     return {"bce_loss": bce_loss.item(), "soft_iou": soft_iou.item(), "mse": mse.item()}
#
#
# def analyze_attention_blocks(
#     image, words, gsam2, inspector,
#     num_blocks=24, region="T2I",
# ):
#     """
#     Analyze which blocks have best attention localization.
#
#     Returns:
#         avg_metrics: Dict with BCE, IoU, MSE per block
#         top_blocks: Dict with top-10 blocks for each metric
#     """
#     block_metrics = {i: {"bce": [], "iou": [], "mse": []} for i in range(num_blocks)}
#
#     for word in words:
#         mask = gsam2(image=image, text=word, res=(64, 64), flatten=True)
#
#         for block_id in range(num_blocks):
#             attn = inspector.aggregate_attention(
#                 timesteps=list(range(inspector.num_inference_steps)),
#                 block_ids=[block_id],
#                 region=region,
#             )
#             word_attn = inspector.get_word_attention(attn, word)
#             metrics = evaluate_attention_map(word_attn.float(), mask.float())
#             block_metrics[block_id]["bce"].append(metrics["bce_loss"])
#             block_metrics[block_id]["iou"].append(metrics["soft_iou"])
#             block_metrics[block_id]["mse"].append(metrics["mse"])
#
#     avg_metrics = {
#         "bce": [np.mean(block_metrics[i]["bce"]) for i in range(num_blocks)],
#         "iou": [np.mean(block_metrics[i]["iou"]) for i in range(num_blocks)],
#         "mse": [np.mean(block_metrics[i]["mse"]) for i in range(num_blocks)],
#     }
#     top_blocks = {
#         "bce": np.argsort(avg_metrics["bce"])[:10].tolist(),
#         "iou": np.argsort(avg_metrics["iou"])[-10:][::-1].tolist(),
#         "mse": np.argsort(avg_metrics["mse"])[:10].tolist(),
#     }
#
#     return avg_metrics, top_blocks
#
#
# def plot_block_metrics(metrics):
#     """Plot BCE, IoU, MSE per block."""
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#     titles = ["BCE Loss", "IoU", "MSE"]
#
#     for ax, (metric, title) in zip(axes, zip(["bce", "iou", "mse"], titles)):
#         ax.plot(metrics[metric], marker="o")
#         ax.set_title(title)
#         ax.set_xlabel("Block ID")
#         ax.grid(True, alpha=0.3)
#
#     axes[0].set_ylabel("Value")
#     plt.tight_layout()
#     return fig
