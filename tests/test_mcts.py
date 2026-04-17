import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import chess
import torch
import pytest
from chessgame.train.mcts import MCTS, MCTSNode
from chessgame.model.hrm_chess import HRMChess
from chessgame.model.hrm_chess_config import HRMChessConfig
from chessgame.encoding.move_encoder import encode_move


def test_mcts_node_is_leaf():
    board = chess.Board()
    node = MCTSNode(board=board)
    assert node.is_leaf()

    # Expand
    policy = torch.zeros(4672)
    policy[encode_move(chess.Move.from_uci("e2e4"))] = 1.0
    node.expand(policy)
    assert not node.is_leaf()
    assert len(node.children) > 0


def test_mcts_backup():
    board = chess.Board()
    root = MCTSNode(board=board)
    child = MCTSNode(board=board.copy(), parent=root)
    root.children[chess.Move.from_uci("e2e4")] = child

    # Value from perspective of player at child (Black)
    # If child evaluates to 0.5 for Black, it should be -0.5 for White (root)
    child.backup(0.5)

    assert child.N == 1
    assert child.W == 0.5
    assert child.Q == 0.5

    assert root.N == 1
    assert root.W == -0.5
    assert root.Q == -0.5


def test_mcts_select_child():
    board = chess.Board()
    root = MCTSNode(board=board)
    root.N = 10

    m1 = chess.Move.from_uci("e2e4")
    m2 = chess.Move.from_uci("d2d4")

    c1 = MCTSNode(board=board.copy(), parent=root, move=m1)
    c2 = MCTSNode(board=board.copy(), parent=root, move=m2)

    root.children = {m1: c1, m2: c2}

    # Set artificial policy: m1 is very likely
    root.P = torch.zeros(4672)
    root.P[encode_move(m1)] = 0.9
    root.P[encode_move(m2)] = 0.1

    # c1 has 0 visits, c2 has 0 visits. PUCT should pick c1 because of P.
    move, node = root.select_child(c_puct=1.0)
    assert move == m1
    assert node == c1

    # Now give c1 many visits but poor Q
    c1.N = 100
    c1.Q = -1.0
    # c2 has 0 visits, it should definitely be picked now for exploration
    move, node = root.select_child(c_puct=1.0)
    assert move == m2


@pytest.mark.parametrize("device", ["cpu"])
def test_mcts_search_smoke(device):
    config = HRMChessConfig.mac_mini()
    model = HRMChess(config).to(device)
    model.eval()

    mcts = MCTS(model, num_simulations=10, device=device)
    board = chess.Board()

    probs = mcts.search(board)

    assert isinstance(probs, torch.Tensor)
    assert probs.shape == (4672,)
    assert torch.isclose(probs.sum(), torch.tensor(1.0))

    # Verify that only legal moves have non-zero probability
    legal_indices = [encode_move(m) for m in board.legal_moves]
    for i in range(4672):
        if i not in legal_indices:
            assert probs[i] == 0
        elif probs[i] > 0:
            # At least some legal moves should be visited
            pass


if __name__ == "__main__":
    pytest.main([__file__])
