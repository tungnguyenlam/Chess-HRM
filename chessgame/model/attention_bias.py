"""
GAB-aware attention and reasoning module wrappers.

These subclasses inject additive GAB bias into the upstream attention mechanism
WITHOUT modifying hrm_act_v1.py. The upstream Attention.forward() doesn't accept
an attn_bias kwarg, so we override the forward to add it before softmax.

Implements: PLAN.md steps 1.2 and 1.3
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from chessmodels.layers import (
    Attention,
    CosSin,
    apply_rotary_pos_emb,
    rms_norm,
    SwiGLU,
    CastedLinear,
)
from chessmodels.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1Block,
    HierarchicalReasoningModel_ACTV1ReasoningModule,
    HierarchicalReasoningModel_ACTV1Config,
)


class AttentionWithBias(Attention):
    """
    Extends upstream Attention to accept an optional additive attention bias.

    The bias is added to the attention logits before softmax:
        attn_weights = softmax(QK^T / sqrt(d) + gab_bias)

    This does NOT modify the upstream class — it subclasses and overrides forward.
    """

    def forward(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            cos_sin: (cos, sin) for RoPE
            hidden_states: [B, seq_len, hidden_size]
            attn_bias: Optional [B, num_heads, seq_len, seq_len] additive bias
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Manual attention with bias injection (can't use flash_attn with custom bias)
        q = query.transpose(1, 2)  # [B, heads, seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        if attn_bias is not None:
            # Use manual attention to inject GAB bias
            scale = self.head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = attn_weights + attn_bias
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v)
        else:
            # No bias — use PyTorch's optimized SDPA
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)

        attn_output = attn_output.transpose(1, 2)  # [B, seq_len, heads, head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)


class BlockWithBias(nn.Module):
    """
    Transformer block that uses AttentionWithBias instead of plain Attention.
    Mirrors HierarchicalReasoningModel_ACTV1Block but with GAB support.
    """

    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.self_attn = AttentionWithBias(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Post-norm with GAB-aware attention
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states, attn_bias=attn_bias),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )
        return hidden_states


class ReasoningModuleWithBias(nn.Module):
    """
    Reasoning module (H-level or L-level) that passes GAB bias through to blocks.
    Mirrors HierarchicalReasoningModel_ACTV1ReasoningModule.
    """

    def __init__(self, layers: list[BlockWithBias]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attn_bias=attn_bias,
                **kwargs,
            )
        return hidden_states
