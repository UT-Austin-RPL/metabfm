"""Fourier conditioning helpers for the hybrid α-β ZSRL agent.

``fourier_encode`` expands an α or β scalar in ``[0, 1]`` into sin/cos
features at linearly-spaced frequencies plus the raw scalar, so a downstream
MLP can distinguish interior α/β values without the scalar being drowned out
by the much wider ``policy_ctx`` / ``ext_ctx`` features in the same input.

Linear spacing ``freqs = [pi, 2pi, ..., n_freq * pi]`` (default
``n_freq=16``, ~8 full cycles over ``[0, 1]``) separates ~16 α values while
the lowest frequency stays monotonic for smooth interpolation between α=0
and α=1.  Geometric / NeRF-style spacing (``2^k * pi``) would alias adjacent
α values into noise.

The raw-scalar tail lets the MLP recover ``x`` linearly instead of learning
an arccos-style inverse from the sin/cos channels.

Output dim per scalar is ``2 * n_freq + 1`` (33 by default; 66 for α + β).
"""

from __future__ import annotations

import math

import torch


DEFAULT_N_FREQ = 16


def fourier_encoded_dim(n_freq: int = DEFAULT_N_FREQ) -> int:
    """Output dim of :func:`fourier_encode` for a scalar input."""
    return 2 * n_freq + 1


def fourier_encode(x: torch.Tensor, n_freq: int = DEFAULT_N_FREQ) -> torch.Tensor:
    """Fourier-encode a scalar in ``[0, 1]`` with a raw-scalar safety tail.

    Args:
        x: Tensor with last dim = 1.  Values are assumed to lie in ``[0, 1]``
            (the supported α and β range).
        n_freq: Number of sin/cos frequency pairs.  Defaults to
            :data:`DEFAULT_N_FREQ` (16).

    Returns:
        Tensor with the same leading shape as ``x`` and last dim
        ``2 * n_freq + 1`` =
        ``[sin(k*pi*x) for k=1..n_freq, cos(k*pi*x) for k=1..n_freq, x]``.
    """
    if x.shape[-1] != 1:
        raise ValueError(
            f"fourier_encode expected scalar input with last dim=1, got "
            f"shape {tuple(x.shape)}."
        )
    freqs = torch.arange(1, n_freq + 1, device=x.device, dtype=x.dtype) * math.pi
    angles = x * freqs  # (..., n_freq)
    return torch.cat([angles.sin(), angles.cos(), x], dim=-1)  # (..., 2*n_freq + 1)
