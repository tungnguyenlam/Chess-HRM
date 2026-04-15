# Backlog — HRM-GAB Chess

> **This is the lab notebook.** Every experiment, every decision, every failure goes here.
> Entries are chronological. Never delete entries — only append.
> Cross-reference `PLAN.md` step IDs.

---

## Naming Conventions

| Prefix | Meaning | Example |
|--------|---------|---------|
| `ENV-XXX` | Environment setup, dependency changes | `ENV-001` |
| `SMOKE-XXX` | Smoke tests, sanity checks | `SMOKE-001` |
| `TRAIN-SV-XXX` | Supervised training runs | `TRAIN-SV-001` |
| `TRAIN-DS-XXX` | Distillation training runs | `TRAIN-DS-001` |
| `TRAIN-RL-XXX` | RL self-play training runs | `TRAIN-RL-001` |
| `EVAL-SV-XXX` | Evaluation of supervised checkpoints | `EVAL-SV-001` |
| `EVAL-DS-XXX` | Evaluation of distilled checkpoints | `EVAL-DS-001` |
| `EVAL-RL-XXX` | Evaluation of RL checkpoints | `EVAL-RL-001` |
| `ABL-XXX-YYY` | Ablation studies | `ABL-ROPE-001` |
| `INTERP-XXX` | Interpretability experiments | `INTERP-001` |
| `BUG-XXX` | Bug discoveries and fixes | `BUG-001` |
| `DESIGN-XXX` | Architecture/design decisions | `DESIGN-001` |
| `DATA-XXX` | Dataset creation/processing | `DATA-001` |
| `PLAN-CHANGE-XXX` | Modifications to PLAN.md | `PLAN-CHANGE-001` |

---

## Template

Copy this template for each new entry:

```markdown
### [PREFIX-XXX]: [Title]
- **Date**: YYYY-MM-DD
- **PLAN.md step**: [step number]
- **Hypothesis**: What do you expect to happen and why?
- **Setup**:
  - Hardware: [GPU/CPU, memory]
  - Config: [yaml profile name, key overrides]
  - Seed: [random seed]
  - Commit: [git hash]
- **Command**: `[exact command to reproduce]`
- **Status**: PLANNED | RUNNING | DONE | FAILED | BLOCKED
- **Result**:
  - [metrics, numbers, observations]
- **Conclusion**: What did we learn? Does this change the plan?
- **Artifacts**:
  - Checkpoint: [path]
  - W&B run: [link]
  - Plots: [paths]
- **Follow-up**: [next experiment or action triggered by this result]
```

---

## Log

---

### DESIGN-001: Architecture Selection — HRM-GAB
- **Date**: 2026-04-16
- **PLAN.md step**: Novelty Statement
- **Hypothesis**: Combining HRM's adaptive computation with Chessformer's GAB will outperform either alone, because chess positions require variable depth AND position-specific geometry.
- **Setup**: N/A (design decision, not experiment)
- **Status**: DONE
- **Result**:
  - Reviewed 6 prior architectures (AlphaZero, Searchless Chess, Chessformer, HRM, SearchFormer, ALLIE)
  - No prior work combines adaptive compute + geometric attention + recurrence
  - Estimated parameter budget: ~32M (fits in 16GB unified memory)
- **Conclusion**: HRM-GAB is the chosen architecture. Key innovation: GAB is recomputed each H/L cycle from evolving z_H, creating "deepening geometric understanding."
- **Artifacts**:
  - Architecture debate: `ReinforcementLearninginChess/Chatdumps/ConcatPlan.md`
  - Chessformer notes: `ReinforcementLearninginChess/Papers/2026-CHESSFORNER A UNIFIED.md`
- **Follow-up**: Begin Phase 0 implementation

---

### DESIGN-002: Rejected Architectures
- **Date**: 2026-04-16
- **PLAN.md step**: Novelty Statement
- **Status**: DONE
- **Result**:

  | Architecture | Why Rejected |
  |-------------|-------------|
  | Pure HRM + RoPE | RoPE is chess-agnostic; ignores board geometry. Will serve as ablation baseline. |
  | Chessformer (no ACT) | No adaptive compute — wastes cycles on simple positions, underthinsk complex ones. Ablation baseline. |
  | HRM + CMS + Titans + Hope | "Franken-model" from Nested Learning paper. No chess evidence. Untested interaction between CMS update frequencies and ACT halting masks. |
  | SearchFormer replacing MCTS | Research-stage; not production-ready. Cannot replace MCTS for competitive play. |
  | CNN front-end | Duplicates HRM's hierarchical feature extraction. Destroys token-level ACT. Literature shows transformers + 2D PE match CNNs. |

- **Conclusion**: Rejected alternatives documented for paper's Related Work section. Three will serve as ablation baselines.
- **Follow-up**: None — design frozen.

---

### DESIGN-003: ACT Training Schedule
- **Date**: 2026-04-16
- **PLAN.md step**: Phase 2.1, Phase 3.1, Phase 4.2
- **Status**: DONE
- **Result**:
  - Phase 1 (supervised): ACT OFF — no reward signal to train halt/continue decisions
  - Phase 2 (distillation): ACT OFF — same reason
  - Phase 3+ (RL): ACT ON — game outcomes provide reward signal for Q-head bootstrapping
- **Conclusion**: This matches the consensus from Claude and ChatGPT in the architecture debate. Deepseek's suggestion to train ACT during supervised phase is incorrect.
- **Follow-up**: Implement ACT enable/disable flag in training loops

---

### ENV-001: Environment Setup
- **Date**: 2026-04-16
- **PLAN.md step**: 0.1
- **Setup**:
  - Hardware: M4 Mac Mini, 16GB unified memory
  - Python: 3.9.6 (system), venv at `.venv/`
  - torch=2.8.0, MPS=True, CUDA=False
  - chess=1.11.2, numpy=2.0.2, pytest=8.4.2, h5py=3.14.0
  - Commit: f2aac72
- **Status**: DONE
- **Result**: All core deps installed. MPS backend available. Stockfish NOT installed locally (SF-dependent tests skip gracefully).
- **Conclusion**: Ready for development. Stockfish installation deferred to Phase 0.4 benchmarking.
- **Follow-up**: Install Stockfish binary before running full SF annotation tests.

---

### SMOKE-BOARD-001: Board Encoder Tests
- **Date**: 2026-04-16
- **PLAN.md step**: 0.2
- **Command**: `.venv/bin/python -m pytest tests/test_board_encoder.py -x -v`
- **Status**: DONE
- **Result**: 21/21 passed in 0.70s
  - Shape, dtype, bounds: ✓
  - Piece placement (pawns, king, rooks, empty squares): ✓
  - Auxiliary planes (side to move, castling, halfmove, fullmove): ✓
  - History encoding (offset, zero-padding, 7-board limit): ✓
  - Repetition detection: ✓
- **Conclusion**: Board encoder matches AlphaZero 119-plane spec. Pre-existing code was already correct.

---

### SMOKE-MOVE-001: Move Encoder Tests
- **Date**: 2026-04-16
- **PLAN.md step**: 0.3
- **Command**: `.venv/bin/python -m pytest tests/test_move_encoder.py -x -v`
- **Status**: DONE
- **Result**: 17/17 passed in 0.50s
  - Round-trip (e2e4, Nf3, all starting legal, all Italian Game legal): ✓
  - Promotions (queen, knight, rook, bishop): ✓
  - Castling (kingside, queenside): ✓
  - Directional encoding (N, NE): ✓
  - Knight deltas (all 8 directions): ✓
  - Legal mask (20 moves at start, checkmate = 0): ✓
- **Conclusion**: Move encoder correctly implements 4672-dim AlphaZero action space.

---

### SMOKE-SF-001: Stockfish Annotator Tests
- **Date**: 2026-04-16
- **PLAN.md step**: 0.4
- **Command**: `.venv/bin/python -m pytest tests/test_stockfish_annotator.py -x -v`
- **Status**: DONE
- **Result**: 5 passed, 12 skipped in 0.05s
  - Score conversion (positive cp, negative cp, mate, mated, black perspective): ✓ (5/5)
  - Engine-dependent tests: SKIPPED (Stockfish not installed)
- **Conclusion**: Annotator code structure is correct. JSONL format and score conversion verified. Full integration test deferred until Stockfish binary is installed.
- **Follow-up**: Install Stockfish and re-run to verify engine integration.

---

<!-- NEW ENTRIES GO BELOW THIS LINE -->
