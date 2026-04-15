"""
2D Rotary Position Embedding for an 8x8 chess board + 1 CLS token.

Sequence order: [CLS, sq(r=0,f=0), sq(r=0,f=1), ..., sq(r=7,f=7)] = 65 tokens

The head_dim cos/sin tensor is split into two halves:
  first  head_dim//2  dims  →  row-axis RoPE  (positions 0–7)
  second head_dim//2  dims  →  col-axis RoPE  (positions 0–7)

This produces a (cos, sin) CosSin tuple of shape [65, head_dim] that drops
directly into the upstream apply_rotary_pos_emb without modifying layers.py.
"""
import torch
import torch.nn as nn

from chessmodels.layers import CosSin


class ChessRoPE2D(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for 2D RoPE"
        half = head_dim // 2  # each axis gets head_dim//2 dims

        # Square coordinates: row varies slowly, file varies fast
        rows = torch.arange(8).repeat_interleave(8)  # [64]: 0,0,...,0,1,1,...,7
        cols = torch.arange(8).repeat(8)             # [64]: 0,1,...,7,0,1,...,7

        # CLS token: assign (row=0, col=0) — neutral position
        all_rows = torch.cat([torch.zeros(1, dtype=torch.long), rows])  # [65]
        all_cols = torch.cat([torch.zeros(1, dtype=torch.long), cols])  # [65]

        # Inverse frequencies for half dimensions (standard RoPE formula)
        inv_freq = 1.0 / (base ** (torch.arange(0, half, 2, dtype=torch.float32) / half))

        # Row embedding: [65, half]
        row_freqs = torch.outer(all_rows.float(), inv_freq)   # [65, half//2]
        row_emb = torch.cat([row_freqs, row_freqs], dim=-1)   # [65, half]

        # Col embedding: [65, half]
        col_freqs = torch.outer(all_cols.float(), inv_freq)   # [65, half//2]
        col_emb = torch.cat([col_freqs, col_freqs], dim=-1)   # [65, half]

        # Concatenate to [65, head_dim]
        emb = torch.cat([row_emb, col_emb], dim=-1)

        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self) -> CosSin:
        return self.cos_cached, self.sin_cached
