"""
Geometric Attention Bias (GAB) module for HRM-GAB Chess.

Generates dynamic, position-dependent attention biases from the current z_H
hidden state. Unlike Chessformer's static GAB computed once from input, this
version is recomputed each H/L cycle, so geometric understanding evolves as
the model "thinks deeper."

Architecture:
    z_H [B, 65, hidden] → compress → project → [B, heads, 65, 65] + static templates

Reference: Chessformer (2026), Section 3.2 — adapted for recurrent use.
Implements: PLAN.md step 1.1
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GeometricAttentionBias(nn.Module):
    """
    Generates per-head additive attention biases from the board representation.

    Args:
        hidden_size:   Dimension of z_H per token (e.g. 512).
        num_heads:     Number of attention heads (e.g. 8).
        seq_len:       Sequence length including CLS (65).
        compress_dim:  Intermediate compression dimension (e.g. 128).
        use_static_templates: Whether to add learnable static bias templates.

    Input:
        z_H: [B, seq_len, hidden_size]

    Output:
        bias: [B, num_heads, seq_len, seq_len]
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        seq_len: int = 65,
        compress_dim: int = 128,
        use_static_templates: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.seq_len = seq_len

        # Compress full board representation to a compact vector
        # Use a small MLP rather than flattening the full hidden*seq_len
        # to keep parameter count and memory manageable.
        self.pool = nn.AdaptiveAvgPool1d(1)  # [B, hidden, seq_len] → [B, hidden, 1]
        self.compress = nn.Sequential(
            nn.Linear(hidden_size, compress_dim),
            nn.GELU(),
            nn.LayerNorm(compress_dim),
        )

        # Project compressed representation to per-head bias matrices
        # Output: [B, num_heads * seq_len * seq_len]
        # For seq_len=65, heads=8: 65*65*8 = 33,800 outputs per sample
        self.bias_proj = nn.Linear(compress_dim, num_heads * seq_len * seq_len)

        # Learnable static templates capture fixed geometric patterns
        # (e.g. knight movement patterns, diagonal relationships)
        self.use_static_templates = use_static_templates
        if use_static_templates:
            self.static_templates = nn.Parameter(
                torch.zeros(num_heads, seq_len, seq_len)
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize bias projection near zero for stable training start."""
        nn.init.zeros_(self.bias_proj.weight)
        nn.init.zeros_(self.bias_proj.bias)

    def forward(self, z_H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_H: [B, seq_len, hidden_size]

        Returns:
            bias: [B, num_heads, seq_len, seq_len]
        """
        B = z_H.shape[0]

        # Pool across sequence: [B, seq_len, hidden] → [B, hidden]
        pooled = self.pool(z_H.transpose(1, 2)).squeeze(-1)  # [B, hidden]

        # Compress: [B, hidden] → [B, compress_dim]
        compressed = self.compress(pooled.float())

        # Project to bias: [B, compress_dim] → [B, H*S*S]
        dynamic = self.bias_proj(compressed)
        dynamic = dynamic.view(B, self.num_heads, self.seq_len, self.seq_len)

        # Cast to match input dtype
        dynamic = dynamic.to(z_H.dtype)

        # Add static templates
        if self.use_static_templates:
            dynamic = dynamic + self.static_templates.unsqueeze(0).to(z_H.dtype)

        return dynamic
