"""
Interpretability analysis tools for HRM-GAB Chess.

Includes:
- GAB attention bias visualization
- ACT depth (halt step) analysis
- Game phase classification

Implements: PLAN.md step 5.4
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import chess

from chessgame.encoding.board_encoder import encode_board


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
    sf_eval: float


def classify_game_phase(board: chess.Board) -> str:
    """Simple heuristic for game phase classification."""
    fullmove_number = board.fullmove_number
    num_pieces = len(board.piece_map())

    if fullmove_number <= 10:
        return "opening"
    elif num_pieces <= 12:
        return "endgame"
    else:
        return "middlegame"


def visualize_gab_bias(
    snapshot: GABSnapshot,
    save_path: str,
    head_idx: int = 0,
):
    """
    Plot the GAB attention bias matrix for a specific head.
    The sequence is [CLS, sq0, sq1, ..., sq63].
    """
    bias_matrix = snapshot.bias[head_idx].cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(bias_matrix, cmap="viridis")
    plt.title(f"GAB Attention Bias - Head {head_idx} - Cycle {snapshot.cycle}")
    plt.xlabel("Key position")
    plt.ylabel("Query position")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_act_histogram(
    records: List[ACTDepthRecord],
    save_path: str,
):
    """Plot histogram of halt steps grouped by game phase."""
    phases = ["opening", "middlegame", "endgame"]
    data = {p: [r.halt_step for r in records if r.phase == p] for p in phases}

    plt.figure(figsize=(10, 6))
    for p in phases:
        if data[p]:
            plt.hist(data[p], bins=np.arange(1, 10) - 0.5, alpha=0.5, label=p)

    plt.title("ACT Halt Step Distribution by Game Phase")
    plt.xlabel("Halt Step")
    plt.ylabel("Frequency")
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


@torch.no_grad()
def record_interpretability_metrics(
    model: any,  # HRMChess
    board: chess.Board,
    device: torch.device,
) -> Dict:
    """
    Run one forward pass and extract GAB biases and ACT halt steps.
    Note: Requires model to be instrumented to return these.
    """
    board_t = encode_board(board, history=[]).unsqueeze(0).to(device)
    batch = {
        "inputs": board_t,
        "puzzle_identifiers": torch.zeros(1, dtype=torch.int32, device=device),
    }

    carry = model.initial_carry(batch)
    # Forward pass - we assume the model returns 'gab_biases' and 'halt_steps' in outputs
    carry, outputs = model(carry, batch)

    # Extract metrics
    # Note: These keys must match what HRMChess.forward returns
    metrics = {
        "halt_step": outputs.get("halt_steps", torch.tensor([1]))[0].item(),
        "gab_biases": outputs.get("gab_biases", []),  # List of [B, H, S, S]
        "phase": classify_game_phase(board),
        "fen": board.fen(),
    }

    return metrics
