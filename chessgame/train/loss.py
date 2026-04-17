"""
Pure loss functions for chess HRM training.
No training loop logic — just the math.
"""

from typing import Optional, Dict
import torch
import torch.nn.functional as F


def policy_hard(logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss against a single best-move index.  Phase 2 supervised."""
    return F.cross_entropy(logits.float(), target_idx)


def policy_soft(logits: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
    """
    Soft cross-entropy (= KL divergence up to a constant) against a
    probability distribution over moves.  Phase 3 distillation.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(soft_target * log_probs).sum(dim=-1).mean()


def value_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between predicted value [B,1] and target [B]."""
    return F.mse_loss(pred.squeeze(-1), target)


def act_q_loss(
    outputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    ACT Q-learning loss combining:
      - q_halt_loss     (BCE on halt decision vs sequence correctness)
      - q_continue_loss (BCE on bootstrapped Q-continue target)

    Should only be included during Phase 4 RL when game outcomes provide
    a meaningful reward signal.  Pass outputs dict from HRMChess.forward().
    """
    q_halt_logits = outputs["q_halt_logits"]
    q_continue_logits = outputs["q_continue_logits"]

    # q_halt target: 1 when halting is correct (proxy: use halted flag)
    # During RL the outer model computes target_q_continue via bootstrapping.
    q_continue_loss = torch.tensor(0.0, device=q_halt_logits.device)
    if "target_q_continue" in outputs:
        q_continue_loss = F.binary_cross_entropy_with_logits(
            q_continue_logits, outputs["target_q_continue"], reduction="mean"
        )

    return q_continue_loss


def total_supervised(
    outputs: Dict[str, torch.Tensor],
    move_target: torch.Tensor,
    value_target: torch.Tensor,
) -> torch.Tensor:
    """Phase 2 loss: policy CE + value MSE.  ACT excluded."""
    return policy_hard(outputs["policy"], move_target) + value_mse(
        outputs["value"], value_target
    )


def total_distill(
    outputs: Dict[str, torch.Tensor],
    soft_policy: torch.Tensor,
    value_target: torch.Tensor,
) -> torch.Tensor:
    """Phase 3 loss: soft policy KL + value MSE.  ACT excluded."""
    return policy_soft(outputs["policy"], soft_policy) + value_mse(
        outputs["value"], value_target
    )


def total_rl(
    outputs: Dict[str, torch.Tensor],
    mcts_pi: torch.Tensor,
    outcome: torch.Tensor,
    act_weight: float = 0.1,
    kl_sf_loss: Optional[torch.Tensor] = None,
    kl_weight: float = 0.0,
) -> torch.Tensor:
    """
    Phase 4 RL loss:
      soft_policy + value_mse + act_weight * act_q_loss
      + kl_weight * kl_sf_loss  (Stockfish KL regularizer, annealed to 0)
    """
    loss = (
        policy_soft(outputs["policy"], mcts_pi)
        + value_mse(outputs["value"], outcome)
        + act_weight * act_q_loss(outputs)
    )

    if kl_sf_loss is not None and kl_weight > 0.0:
        loss = loss + kl_weight * kl_sf_loss

    return loss
