"""Common utilities for attention manipulation in diffusion models."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Union

# Default seed used throughout experiments
DEFAULT_SEED = 864

# Top-5 blocks for attention aggregation (from paper analysis)
# Few-step models share architecture with their base models
TOP5_BLOCKS = {
    "sd3-m": [7, 8, 5, 4, 9],
    "sd3.5-m": [7, 9, 8, 5, 10],
    "sd3.5-l": [18, 21, 20, 24, 16],
    "sd3.5-l-turbo": [18, 21, 20, 24, 16],  # Same architecture as sd3.5-l
    "flux-dev": [18, 17, 12, 14, 11],
    "flux-schnell": [18, 17, 12, 14, 11],  # Same architecture as flux-dev
}

# Default parameters
DEFAULT_REPLACE_STEPS = 0.2  # τ in Algorithm 1
DEFAULT_LOCAL_BLEND_STEPS = 0.5  # η in Algorithm 1
DEFAULT_LOCAL_BLEND_THRESHOLD = 0.3


def get_word_inds(text: str, word: Union[str, int], tokenizer) -> np.ndarray:
    """
    Get token indices for a word in the text.

    Args:
        text: Input text
        word: Word to find (string) or word index (int)
        tokenizer: Tokenizer for encoding

    Returns:
        Array of token indices
    """
    split_text = text.strip().split()
    if isinstance(word, str):
        word_indices = [i for i, w in enumerate(split_text) if w == word]
    elif isinstance(word, int):
        word_indices = [word]
    else:
        raise ValueError("Word must be a string or an integer index.")

    token_indices = []

    if "T5" in tokenizer.__class__.__name__:
        if word_indices:
            tokens = tokenizer.encode(text)
            words_encode = [tokenizer.decode([token]).strip("#") for token in tokens][:-1]
            ptr = 0
            cur_len = 0
            for i, token_str in enumerate(words_encode):
                cur_len += len(token_str)
                if ptr in word_indices:
                    token_indices.append(i)
                if cur_len >= len(split_text[ptr]):
                    ptr += 1
                    cur_len = 0
    else:
        if word_indices:
            tokens = tokenizer.encode(text)
            words_encode = [tokenizer.decode([token]).strip("#") for token in tokens][1:-1]
            ptr = 0
            cur_len = 0
            for i, token_str in enumerate(words_encode):
                cur_len += len(token_str)
                if ptr in word_indices:
                    token_indices.append(i + 1)
                if cur_len >= len(split_text[ptr]):
                    ptr += 1
                    cur_len = 0

    return np.array(token_indices)


def get_multi_word_inds(text: str, words: str, tokenizer) -> np.ndarray:
    """Get token indices for multiple words (space-separated)."""
    texts = text.split()
    words_list = words.split()
    for word in words_list:
        if word not in texts:
            print(f"WARNING: Word '{word}' not found in text.")
    if len(words_list) == 1:
        return get_word_inds(text, words_list[0], tokenizer)
    else:
        return np.concatenate([get_word_inds(text, w, tokenizer) for w in words_list])


def get_refinement_mapper(prompts: List[str], tokenizer, max_len: int = 77) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create mapper and alphas for refining attention.

    Args:
        prompts: List of prompts [source, target]
        tokenizer: Tokenizer for encoding
        max_len: Maximum sequence length

    Returns:
        Tuple of (mapper, alphas) tensors
    """
    x_prompt = prompts[0]
    mappers, alphas = [], []
    for y_prompt in prompts[1:]:
        mapper, alpha = _get_mapper(x_prompt, y_prompt, tokenizer, max_len)
        mappers.append(mapper)
        alphas.append(alpha)
    return torch.stack(mappers), torch.stack(alphas)


def _get_mapper(x: str, y: str, tokenizer, max_len: int = 77) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create mapping between tokens of two prompts using sequence alignment."""
    x_seq = tokenizer.encode(x)
    y_seq = tokenizer.encode(y)
    score = _ScoreParams(gap=-1, match=2, mismatch=-1)
    _, trace_back = _global_align(x_seq, y_seq, score)
    mapper_base = _get_aligned_sequences(x_seq, y_seq, trace_back)[-1]

    alphas = torch.ones(max_len)
    alphas[:mapper_base.shape[0]] = (mapper_base[:, 1] != -1).float()
    mapper = torch.zeros(max_len, dtype=torch.int64)
    mapper[:mapper_base.shape[0]] = mapper_base[:, 1]
    mapper[mapper_base.shape[0]:] = len(y_seq) + torch.arange(max_len - len(y_seq))
    return mapper, alphas


class _ScoreParams:
    def __init__(self, gap, match, mismatch):
        self.gap = gap
        self.match = match
        self.mismatch = mismatch

    def mis_match_char(self, x, y):
        return self.match if x == y else self.mismatch


def _global_align(x, y, score):
    """Needleman-Wunsch sequence alignment."""
    size_x, size_y = len(x), len(y)
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    trace_back = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)

    for i in range(1, size_x + 1):
        matrix[i, 0] = matrix[i - 1, 0] + score.gap
        trace_back[i, 0] = 2
    for j in range(1, size_y + 1):
        matrix[0, j] = matrix[0, j - 1] + score.gap
        trace_back[0, j] = 1

    for i in range(1, size_x + 1):
        for j in range(1, size_y + 1):
            match = matrix[i - 1, j - 1] + score.mis_match_char(x[i - 1], y[j - 1])
            delete = matrix[i - 1, j] + score.gap
            insert = matrix[i, j - 1] + score.gap
            max_score = max(match, delete, insert)
            matrix[i, j] = max_score
            if max_score == match:
                trace_back[i, j] = 3
            elif max_score == delete:
                trace_back[i, j] = 2
            else:
                trace_back[i, j] = 1
    return matrix, trace_back


def _get_aligned_sequences(x, y, trace_back):
    """Retrieve aligned sequences from traceback matrix."""
    x_aligned, y_aligned = [], []
    i, j = len(x), len(y)
    mapper_y_to_x = []
    while i > 0 or j > 0:
        if trace_back[i, j] == 3:
            x_aligned.append(x[i - 1])
            y_aligned.append(y[j - 1])
            mapper_y_to_x.append((j - 1, i - 1))
            i -= 1
            j -= 1
        elif trace_back[i, j] == 2:
            x_aligned.append(x[i - 1])
            y_aligned.append('-')
            i -= 1
        elif trace_back[i, j] == 1:
            x_aligned.append('-')
            y_aligned.append(y[j - 1])
            mapper_y_to_x.append((j - 1, -1))
            j -= 1
    mapper_y_to_x.reverse()
    return x_aligned[::-1], y_aligned[::-1], torch.tensor(mapper_y_to_x, dtype=torch.int64)


def get_equalizer(
    text: str,
    word_select: Union[List[Union[str, int]], str, int],
    values: Union[List[float], Tuple[float, ...]],
    tokenizer,
    max_len: int = 77,
) -> torch.Tensor:
    """
    Create equalizer tensor for reweighting attention.

    Args:
        text: Input text
        word_select: Words or indices to select
        values: Reweighting values
        tokenizer: Tokenizer for encoding
        max_len: Maximum sequence length

    Returns:
        Equalizer tensor
    """
    if isinstance(word_select, (str, int)):
        word_select = [word_select]
    equalizer = torch.ones(len(values), max_len)
    values = torch.tensor(values, dtype=torch.float32)
    for i, word in enumerate(word_select):
        inds = get_word_inds(text, word, tokenizer)
        equalizer[i, inds] = values[i]
    return equalizer


def gaussian_smoothing(tensor: torch.Tensor, kernel_size: int = 7, sigma: float = 1.0, dim: Tuple[int, int] = None) -> torch.Tensor:
    """
    Apply Gaussian smoothing to flattened attention map.

    Args:
        tensor: Input tensor of shape (batch, flat_dim)
        kernel_size: Gaussian kernel size
        sigma: Gaussian sigma
        dim: Spatial dimensions to reshape to (auto-computed if None)

    Returns:
        Smoothed tensor of same shape
    """
    batch, flat_dim = tensor.shape
    if dim is None:
        # Assume square spatial dimensions
        h = w = int(np.sqrt(flat_dim))
        assert h * w == flat_dim, f"flat_dim {flat_dim} is not a perfect square"
    else:
        h, w = dim
    tensor = tensor.view(batch, 1, h, w)

    x = torch.arange(kernel_size) - kernel_size // 2
    x_grid = x.repeat(kernel_size, 1).float()
    y_grid = x_grid.t()
    gaussian_kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))
    gaussian_kernel /= gaussian_kernel.sum()

    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.to(tensor.device, dtype=tensor.dtype)

    smoothed_tensor = F.conv2d(tensor, gaussian_kernel, padding=kernel_size // 2)
    return smoothed_tensor.view(batch, flat_dim)


def get_alpha_layers(
    pipeline_type: str,
    prompts: List[str],
    words: List,
    tokenizer,
    max_t5_seq_len: int,
) -> torch.Tensor:
    """
    Compute alpha layers for local blending based on word indices.

    Args:
        pipeline_type: "sd3" or "flux"
        prompts: List of prompts
        words: List of words to blend for each prompt
        tokenizer: List of tokenizers
        max_t5_seq_len: Maximum T5 sequence length

    Returns:
        Alpha layers tensor
    """
    if pipeline_type == "sd3":
        alpha_layers1 = torch.zeros(len(prompts), 77)
        alpha_layers3 = torch.zeros(len(prompts), max_t5_seq_len)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if isinstance(words_, str):
                words_ = [words_]
            for word in words_:
                ind1 = get_multi_word_inds(prompt, word, tokenizer[0])
                ind3 = get_multi_word_inds(prompt, word, tokenizer[2])
                alpha_layers1[i, ind1] = 1
                alpha_layers3[i, ind3] = 1
        alpha_layers = torch.cat([alpha_layers1, alpha_layers3], dim=-1)
    else:
        alpha_layers = torch.zeros(len(prompts), max_t5_seq_len)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if isinstance(words_, str):
                words_ = [words_]
            for word in words_:
                ind = get_multi_word_inds(prompt, word, tokenizer[1])
                alpha_layers[i, ind] = 1
    return alpha_layers
