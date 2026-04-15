"""
Fixed-capacity circular replay buffer for self-play RL (Phase 4).

Stores (board_tensor, mcts_pi, outcome) tuples.
"""
from typing import Tuple
import random

import torch


class ReplayBuffer:
    """
    Circular buffer of (board_tensor [8,8,119], mcts_pi [4672], outcome float) tuples.

    Args:
        capacity: Maximum number of samples stored.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._buf: list[tuple] = []
        self._pos = 0

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, samples: list[tuple]) -> None:
        """Add a list of (board_tensor, mcts_pi, outcome) samples."""
        for s in samples:
            if len(self._buf) < self.capacity:
                self._buf.append(s)
            else:
                self._buf[self._pos] = s
            self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a batch of randomly sampled data.

        Returns:
            board_tensors  [B, 8, 8, 119]
            mcts_pi        [B, 4672]
            outcomes       [B]
        """
        batch = random.sample(self._buf, min(batch_size, len(self._buf)))
        boards, pis, outcomes = zip(*batch)
        return (
            torch.stack(boards),
            torch.stack(pis),
            torch.tensor(outcomes, dtype=torch.float32),
        )
