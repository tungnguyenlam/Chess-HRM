"""
Chess adapter for HierarchicalReasoningModel_ACTV1 with Geometric Attention Bias.

HRMChessInner  — subclasses the upstream inner model, replaces token embedding
                 with board projection + 2D RoPE, overrides forward to expose z_H.
                 GAB is computed per H-cycle from evolving z_H.

HRMChess       — standalone nn.Module that owns HRMChessInner and re-implements
                 the ACT outer loop (copied verbatim from the upstream outer wrapper)
                 so we keep full control without double-instantiation.

Implements: PLAN.md steps 1.2, 1.3
"""

from typing import Dict, Tuple, List

import torch
import torch.nn as nn

from chessmodels.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1_Inner,
    HierarchicalReasoningModel_ACTV1InnerCarry,
    HierarchicalReasoningModel_ACTV1Carry,
)
from chessmodels.layers import CastedLinear

from chessgame.model.hrm_chess_config import HRMChessConfig
from chessgame.model.rope_2d import ChessRoPE2D
from chessgame.model.gab import GeometricAttentionBias
from chessgame.model.attention_bias import BlockWithBias, ReasoningModuleWithBias


class HRMChessInner(HierarchicalReasoningModel_ACTV1_Inner):
    """
    Replaces embed_tokens + lm_head with board projection + chess heads.
    Overrides forward entirely to use 2D RoPE and expose z_H directly.
    Replaces H/L levels with GAB-aware ReasoningModuleWithBias.

    H_init, L_init, q_head are inherited unchanged.
    """

    def __init__(self, config: HRMChessConfig) -> None:
        super().__init__(config)

        # Remove unused parent modules to save memory
        del self.embed_tokens
        del self.lm_head
        if hasattr(self, "rotary_emb"):
            del self.rotary_emb
        if hasattr(self, "embed_pos"):
            del self.embed_pos

        # Replace upstream H/L levels with GAB-aware versions
        del self.H_level
        del self.L_level
        self.H_level = ReasoningModuleWithBias(
            [BlockWithBias(config) for _ in range(config.H_layers)]
        )
        self.L_level = ReasoningModuleWithBias(
            [BlockWithBias(config) for _ in range(config.L_layers)]
        )

        # Board input: project each of 64 squares from 119 planes to hidden_size
        self.board_proj = CastedLinear(
            config.board_input_dim, config.hidden_size, False
        )

        # 2D positional encoding for 8x8 grid + CLS token
        self.rope_2d = ChessRoPE2D(config.hidden_size // config.num_heads)

        # CLS token (prepended at position 0; q_head already uses z_H[:,0,:])
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, config.hidden_size, dtype=self.forward_dtype)
        )

        # Chess output heads
        self.policy_head = CastedLinear(config.hidden_size, config.policy_size, False)
        self.value_head = nn.Sequential(
            CastedLinear(config.hidden_size, 256, True),
            nn.GELU(),
            CastedLinear(256, 1, True),
            nn.Tanh(),
        )

        # Geometric Attention Bias module
        gab_compress = getattr(config, "gab_compress_dim", 128)
        gab_static = getattr(config, "gab_static_templates", True)
        gab_enabled = getattr(config, "gab_enabled", True)

        self.gab_enabled = gab_enabled
        if gab_enabled:
            self.gab = GeometricAttentionBias(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                seq_len=config.seq_len,
                compress_dim=gab_compress,
                use_static_templates=gab_static,
            )

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[
        HierarchicalReasoningModel_ACTV1InnerCarry,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        List[torch.Tensor],
    ]:
        board = batch["inputs"].to(self.forward_dtype)  # [B, 8, 8, 119]
        B = board.shape[0]

        # Board projection → [B, 64, hidden]
        x = board.view(B, 64, self.config.board_input_dim)
        x = self.board_proj(x)

        # Prepend CLS token → [B, 65, hidden]
        cls = self.cls_token.expand(B, -1, -1)
        input_embeddings = torch.cat([cls, x], dim=1)

        seq_info = dict(cos_sin=self.rope_2d())

        # ---------------------------------------------------------------
        # H/L reasoning cycles — one-step gradient trick (from upstream)
        # All iterations except the last run inside torch.no_grad().
        # GAB is recomputed from z_H at the start of each H-cycle.
        # ---------------------------------------------------------------
        gab_biases = []
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H in range(self.config.H_cycles):
                # Compute GAB from current z_H (evolves each cycle)
                gab_bias = self.gab(z_H) if self.gab_enabled else None
                if gab_bias is not None:
                    gab_biases.append(gab_bias)

                for _L in range(self.config.L_cycles):
                    last_iter = (_H == self.config.H_cycles - 1) and (
                        _L == self.config.L_cycles - 1
                    )
                    if not last_iter:
                        z_L = self.L_level(
                            z_L, z_H + input_embeddings, attn_bias=gab_bias, **seq_info
                        )

                if _H != self.config.H_cycles - 1:
                    z_H = self.H_level(z_H, z_L, attn_bias=gab_bias, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad (final iteration, gradients flow here)
        # Recompute GAB with gradient tracking
        gab_bias_grad = self.gab(z_H) if self.gab_enabled else None
        if gab_bias_grad is not None:
            gab_biases.append(gab_bias_grad)

        z_L = self.L_level(
            z_L, z_H + input_embeddings, attn_bias=gab_bias_grad, **seq_info
        )
        z_H = self.H_level(z_H, z_L, attn_bias=gab_bias_grad, **seq_info)

        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
        )

        # CLS token hidden state — used by policy/value heads and q_head
        cls_hidden = z_H[:, 0, :]  # [B, hidden]

        q_logits = self.q_head(cls_hidden).to(torch.float32)  # [B, 2]

        # Return cls_hidden as "output"; outer HRMChess applies chess heads
        return new_carry, cls_hidden, (q_logits[..., 0], q_logits[..., 1]), gab_biases


class HRMChess(nn.Module):
    """
    Full chess model: wraps HRMChessInner with the ACT outer loop.

    batch dict expected by forward():
      "inputs"              float32 [B, 8, 8, 119]  board tensor
      "puzzle_identifiers"  int32   [B]              dummy zeros (required by carry init)
    """

    def __init__(self, config: HRMChessConfig) -> None:
        super().__init__()
        self.config = config
        self.inner = HRMChessInner(config)

    def initial_carry(
        self, batch: Dict[str, torch.Tensor]
    ) -> HierarchicalReasoningModel_ACTV1Carry:
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        inner_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len,
                self.config.hidden_size,
                dtype=self.inner.forward_dtype,
                device=device,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.seq_len,
                self.config.hidden_size,
                dtype=self.inner.forward_dtype,
                device=device,
            ),
        )
        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=inner_carry,
            steps=torch.zeros(batch_size, dtype=torch.int32, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Reset carry for sequences that halted in the previous step
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, cls_hidden, (q_halt_logits, q_continue_logits), gab_biases = (
            self.inner(new_inner_carry, new_current_data)
        )

        # Apply chess output heads
        policy = self.inner.policy_head(cls_hidden)  # [B, 4672]
        value = self.inner.value_head(cls_hidden)  # [B, 1]

        outputs: Dict[str, torch.Tensor] = {
            "policy": policy,
            "value": value,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "gab_biases": gab_biases,
            "halt_steps": new_steps + 1,
        }

        # ACT halting logic (copied verbatim from upstream outer wrapper)
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and self.config.halt_max_steps > 1:
                halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                ) * torch.randint_like(
                    new_steps, low=2, high=self.config.halt_max_steps + 1
                )
                halted = halted & (new_steps >= min_halt_steps)

                # Bootstrap Q-continue target
                next_q_halt, next_q_cont = self.inner(
                    new_inner_carry, new_current_data
                )[-2]
                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt,
                        torch.maximum(next_q_halt, next_q_cont),
                    )
                )

        new_carry = HierarchicalReasoningModel_ACTV1Carry(
            new_inner_carry, new_steps, halted, new_current_data
        )
        return new_carry, outputs
