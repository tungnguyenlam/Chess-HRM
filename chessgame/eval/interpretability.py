"""
Interpretability analysis scaffolding.

Placeholder for:
- GAB attention bias visualization across recurrent cycles
- ACT depth histogram by game phase and position complexity
- Concept probing on z_H representations

Implements: PLAN.md step 0.6 (scaffolding only — full implementation in Phase 5)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class GABSnapshot:
    """GAB bias at a specific H/L cycle."""
    cycle: int
    bias: torch.Tensor  # [num_heads, seq_len, seq_len]


@dataclass
class ACTDepthRecord:
    """Halting step count for one position."""
    fen: str
    halt_step: int
    phase: str  # "opening", "middlegame", "endgame"
    sf_eval_variance: float  # proxy for position complexity


def classify_game_phase(fullmove_number: int, num_pieces: int) -> str:
    """Simple heuristic for game phase classification."""
    if fullmove_number <= 10:
        return "opening"
    elif num_pieces <= 10:
        return "endgame"
    else:
        return "middlegame"
