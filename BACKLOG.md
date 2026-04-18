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

### SMOKE-DATA-001: Dataset Pipeline Tests
- **Date**: 2026-04-16
- **PLAN.md step**: 0.5
- **Command**: `.venv/bin/python -m pytest tests/test_data_pipeline.py -x -v`
- **Status**: DONE
- **Result**: 16/16 passed in 0.93s
  - LichessEliteDataset (load, filter, shapes, value range, move validity): ✓ (6/6)
  - StockfishSoftDataset (load, shapes, soft policy distribution, bounds): ✓ (5/5)
  - ReplayBuffer (empty, add, circular overwrite, sample shapes, clamp): ✓ (5/5)
- **Conclusion**: All dataset components work correctly with synthetic JSONL fixtures.

---

### SMOKE-EVAL-001: Evaluation Infrastructure Tests
- **Date**: 2026-04-16
- **PLAN.md step**: 0.6
- **Command**: `.venv/bin/python -m pytest tests/test_eval.py -x -v`
- **Status**: DONE
- **Result**: 15/15 passed in 0.87s
  - Puzzle CSV parsing (valid, setup move, invalid, single move): ✓ (4/4)
  - PuzzleEvaluator (perfect model, wrong model, max_puzzles, rating buckets): ✓ (4/4)
  - PuzzleReport (empty, accuracy calc): ✓ (2/2)
  - Interpretability (game phase classifier, dataclasses): ✓ (5/5)
- **Conclusion**: Evaluation pipeline ready. Arena eval in evaluate_chess.py; puzzle eval in chessgame/eval/puzzles.py.

---

### SMOKE-GAB-001: GAB Module + Full Model Smoke Test
- **Date**: 2026-04-16
- **PLAN.md step**: 1.1, 1.2, 1.3, 1.5
- **Command**: `.venv/bin/python -m pytest tests/test_gab_model.py -x -v`
- **Status**: DONE
- **Result**: 18/18 passed in 1.55s
  - GAB module (shape, dtype, gradient, NaN, zero-init, static toggle, differentiation, batch independence): ✓ (8/8)
  - AttentionWithBias (bias changes output, zero-bias matches no-bias, gradient through bias): ✓ (3/3)
  - Full model smoke (forward shapes, backward, GAB gradients, no-GAB ablation, no NaN, param count, deterministic): ✓ (7/7)
- **Architecture decisions**:
  - GAB uses AdaptiveAvgPool1d + MLP (not full flatten) to keep params manageable
  - AttentionWithBias uses manual QK^T+bias+softmax when bias present; SDPA fallback when None
  - ReasoningModuleWithBias subclasses pass attn_bias through to blocks
  - Upstream hrm_act_v1.py NOT modified (hard invariant preserved)
- **Conclusion**: HRM-GAB architecture fully operational. One-step gradient trick works with GAB. Ready for Phase 2 training.
- **Follow-up**: Begin supervised training loop implementation.

---

### FULL-REGRESSION-001: Complete Test Suite
- **Date**: 2026-04-16
- **Command**: `.venv/bin/python -m pytest tests/ -x -v`
- **Status**: DONE
- **Result**: 92 passed, 12 skipped in 1.10s
- **Conclusion**: No regressions. All skips are Stockfish-dependent (expected).

---

<!-- NEW ENTRIES GO BELOW THIS LINE -->

### BUG-001: Stockfish Annotator Test Failure
- **Date**: 2026-04-16
- **PLAN.md step**: 0.4
- **Symptoms**: `test_missing_binary_raises` failed because `StockfishAnnotator` didn't check for file existence when a string path was provided.
- **Fix**: Added `not Path(stockfish_path).exists()` check in `__init__`.
- **Status**: DONE
- **Result**: `tests/test_stockfish_annotator.py` now passes (17/17 passed).
- **Conclusion**: Fixed initialization logic to be more robust.

### SMOKE-TRAIN-SV-001: Supervised Training Loop Verification
- **Date**: 2026-04-16
- **PLAN.md step**: 2.1
- **Command**: `.venv/bin/python tests/test_supervised_train.py`
- **Status**: DONE
- **Result**:
  - Model parameters: 5,833,163 (mac_mini)
  - Dataset indexing: 20 records in 0.05s
  - Training: 1 epoch, batch size 4
  - Checkpoint saved: `tmp_test_data/checkpoints/epoch_1.pt`
  - Resumption logic: Added `resume_from` and verified it loads model/optimizer/step/epoch.
  - Logging: Added gradient norm logging.
- **Conclusion**: Supervised training loop is fully functional and supports resumption.
- **Follow-up**: Ready for full-scale Phase 2 training on Lichess Elite dataset.

### SMOKE-TRAIN-DS-001: Distillation Training Loop Verification
- **Date**: 2026-04-16
- **PLAN.md step**: 3.1
- **Command**: `.venv/bin/python tests/test_distill_train.py`
- **Status**: DONE
- **Result**:
  - `scripts/generate_soft_labels.py`: Successfully generated MultiPV labels using 2 workers in 0.2s.
  - Distillation Training: Completed 1 epoch on 10 positions with soft-policy targets.
  - Checkpoint saved: `tmp_distill_test/checkpoints/epoch_1.pt`.
  - Resumption logic: Verified same as supervised loop.
  - Logging: Verified gradient norm and KL-divergence loss (via `total_distill`).
- **Conclusion**: Distillation pipeline is fully functional.
- **Follow-up**: Proceed to Phase 4 (MCTS implementation) after data generation.

### SMOKE-MCTS-001: MCTS Implementation Verification
- **Date**: 2026-04-16
- **PLAN.md step**: 4.1
- **Command**: `.venv/bin/python tests/test_mcts.py`
- **Status**: DONE
- **Result**:
  - `MCTSNode`: Correctly handles backup (value flipping) and selection (PUCT).
  - Batched evaluation: Verified stack of boards passed to model.
  - Search: 10 simulations on starting position produce a valid visit-count distribution.
- **Conclusion**: MCTS is functional and integrated with HRMChess.

### SMOKE-SELFPLAY-001: Self-Play Loop Verification
- **Date**: 2026-04-16
- **PLAN.md step**: 4.2
- **Command**: `.venv/bin/python tests/test_self_play.py`
- **Status**: DONE
- **Result**:
  - `play_game`: Successfully plays a full game (or up to 200 moves) using MCTS.
  - `ReplayBuffer`: Correctly stores (board, pi, outcome) and samples batches.
  - Training step: Loss decreases on a single batch of self-play data.
  - CLI: `python -m chessgame.train.self_play` successfully runs 2 games as a smoke test.
- **Conclusion**: RL training infrastructure is ready.
- **Follow-up**: Proceed to Phase 5 (Ablations & Interpretability).

### EXP-SF-001: Stockfish Annotator Benchmark
- **Date**: 2026-04-17
- **PLAN.md step**: 0.4
- **Hypothesis**: Multiprocessed annotation throughput and RAM usage estimation.
- **Setup**:
  - Scripts: `scripts/estimate_ram.py` and `scripts/estimate_ram_manual.py`
- **Status**: DONE
- **Result**: Scripts created to estimate RAM and benchmark throughput for generating the large Stockfish datasets.
- **Conclusion**: Bottlenecks identified and RAM usage estimated for generating Phase 1 & 2 datasets.
- **Follow-up**: Complete dataset generation for Phase 1 & Phase 2.

### INTERP-001: Interpretability Scaffolding
- **Date**: 2026-04-17
- **PLAN.md step**: 5.4
- **Hypothesis**: We need tools to visualize GAB bias and ACT depth.
- **Setup**:
  - Added `chessgame/eval/interpretability.py`
- **Status**: DONE
- **Result**: Implemented `GABSnapshot`, `ACTDepthRecord`, and game phase classifier.
- **Conclusion**: Scaffolding is ready for Phase 5 ablation and interpretability studies.
- **Follow-up**: Run model and log visualizations to proceed with interpretability experiments.

### DATA-001: Extract Lichess Elite Dataset (Memory & Storage Optimized)
- **Date**: 2026-04-17
- **PLAN.md step**: 0.5
- **Hypothesis**: The raw 209GB PGN file will exceed available SSD space when uncompressed. We must stream-process the 209GB PGN into compressed JSONL shards (.jsonl.zst), then compress the original PGN to `.zst` and remove the raw `.pgn` to reclaim storage.
- **Setup**:
  - Script: `scripts/convert_pgn.py`
  - Max Games Phase 1: 75,000 (~5M positions)
  - Max Games Phase 2: 30,000 (~2M positions)
  - Target: `data-lichess/phase1` and `data-lichess/phase2`
- **Command**: Python extraction followed by `zstd --rm`
- **Status**: DONE
- **Result**:
  - Phase 1 generated 5,480,742 records across 2 shards (`data-lichess/phase1`).
  - Phase 2 generated 2,179,981 records across 1 shard (`data-lichess/phase2`).
  - Compressed the 209GB raw Lichess PGN using `zstd --rm` to ~40GB to reclaim ~169GB of SSD storage.
- **Conclusion**: The large Lichess dataset is now processed into compressed shards suitable for `ShardedLichessDataset`, and storage space has been recovered successfully.
- **Follow-up**: Proceed to train on Phase 1 dataset.

### BUG-002: CPU Training Fails with Multi-Worker DataLoader
- **Date**: 2026-04-18
- **PLAN.md step**: 2.1, 3.1
- **Symptoms**:
  - `tests/test_supervised_train.py` fails on CPU when `DataLoader(num_workers=4)` tries to launch `torch_shm_manager`.
  - `tests/test_distill_train.py` fails the same way on CPU after soft-label generation.
- **Reproduction**:
  - `.venv/bin/python tests/test_supervised_train.py`
  - `.venv/bin/python tests/test_distill_train.py`
- **Status**: DONE
- **Result**:
  - Root cause was the hardcoded worker policy in the training loops: CPU fell into the multi-worker shared-memory path even when the environment only supported single-process loading.
  - Added a shared runtime helper that defaults CPU/MPS to `num_workers=0`, keeps CUDA at `4`, and warns when local CPU/MPS runs explicitly request spawned workers.
  - `tests/test_supervised_train.py`, `tests/test_distill_train.py`, and the full test suite now pass with the new runtime policy.
- **Conclusion**: Fixed by making worker selection device-aware and reusing that logic across training/eval entrypoints.

### TRAIN-SV-001: Phase 1 Supervised Epoch Validation on Real Shards
- **Date**: 2026-04-18
- **PLAN.md step**: 2.2
- **Hypothesis**: After fixing the runtime path, a conservative CPU run should complete one supervised epoch over `data-lichess/phase1` and emit a checkpoint without shared-memory worker failures.
- **Setup**:
  - Hardware: CPU-only sandbox, Apple Silicon host, MPS unavailable to this process
  - Config: `mac_mini` with `--device cpu --num_workers 0 --epochs 1`
  - Seed: Not fixed
  - Commit: `b62d1b4`
- **Command**: `.venv/bin/python scripts/s1_supervised.py --data data-lichess/phase1 --config mac_mini --epochs 1 --device cpu --num_workers 0 --checkpoint_dir /tmp/hrm-gab-supervised-smoke`
- **Status**: RUNNING
- **Result**:
  - Pending
- **Conclusion**: Pending
- **Artifacts**:
  - Checkpoint: `/tmp/hrm-gab-supervised-smoke`
- **Follow-up**: Run full regression tests after the runtime fix, then inspect the epoch logs for any additional training issues.

### BUG-003: ShardedLichessDataset Returned No Samples
- **Date**: 2026-04-18
- **PLAN.md step**: 2.1, 2.2
- **Symptoms**:
  - Real Phase 1 training appeared to hang after shard startup even with progress logging enabled.
  - Loader decoded tens of thousands of records without ever producing a first batch.
- **Root Cause**:
  - `ShardedLichessDataset._process_line()` called `torch.from_numpy(board_tensor)` even though `encode_board()` already returns a `torch.Tensor`.
  - The resulting exception was swallowed by the dataset’s broad `except Exception`, so every record was silently dropped.
- **Fix**:
  - Made `_process_line()` accept the existing tensor output from `encode_board()`.
  - Added a regression test covering `ShardedLichessDataset` iteration.
- **Status**: DONE
- **Result**:
  - `tests/test_data_pipeline.py` now includes a streaming-shard test and passes.
  - Real Phase 1 training now reaches live optimizer steps on `data-lichess/phase1`.
- **Conclusion**: The apparent training “hang” was primarily silent sample loss in the sharded loader, not model deadlock.

### SMOKE-TRAIN-SV-002: Live Phase 1 Progress Logging Validation
- **Date**: 2026-04-18
- **PLAN.md step**: 2.2
- **Command**: `.venv/bin/python scripts/s1_supervised.py --data data-lichess/phase1 --config mac_mini --epochs 1 --device cpu --num_workers 0 --batch_size 8 --accum_steps 1 --min_elo 1800 --checkpoint_dir /tmp/hrm-gab-supervised-smoke`
- **Status**: DONE
- **Result**:
  - Added immediate logging for runtime config, shard open, buffer warmup, first sample, first batch, and optimizer-step loss.
  - Fixed wrapper behavior so explicit `--accum_steps 1` is respected instead of silently reverting to the config value of 16.
  - Verified real training output through step 240 before manual interrupt, with per-step timing and loss visible.
- **Conclusion**: The supervised entrypoint now provides enough live output to distinguish data-loading work from actual stalls.
- **Follow-up**: If desired, mirror the same verbose progress reporting into the distillation loop.

### ENV-002: Apple Silicon Runtime Policy and Python Target Alignment
- **Date**: 2026-04-18
- **PLAN.md step**: 0.1, 2.1, 3.1
- **Hypothesis**: A unified runtime policy plus a Python 3.13 bootstrap script will make local Apple Silicon runs more predictable than the current stale Python 3.9 venv path.
- **Setup**:
  - Hardware: M4 Mac Mini, 16GB unified memory
  - Focus: local environment bootstrap, runtime auto-dtype, worker warnings
- **Command**: `.venv/bin/python --version`
- **Status**: DONE
- **Result**:
  - Active repo venv is still `Python 3.9.6`.
  - System `python3` is `Python 3.13.12`.
  - `.venv` reports `torch 2.8.0`, `mps_built=True`, `mps_available=False` for this process.
  - Added `scripts/setup_mac_env.sh` to rebuild or reuse the local venv from `python3`, install dependencies, and print the resolved Torch/MPS status.
- **Conclusion**: The codebase now has a reproducible Python 3.13 bootstrap path, but this sandboxed process still does not expose MPS even though the Torch build includes it.
- **Artifacts**:
  - Script: `scripts/setup_mac_env.sh`
- **Follow-up**: Run runtime tests and record the active interpreter situation.

### SMOKE-RUNTIME-001: Runtime Auto-Dtype and Worker Policy Validation
- **Date**: 2026-04-18
- **PLAN.md step**: 2.1, 3.1
- **Hypothesis**: The shared runtime helper should resolve `forward_dtype=auto` correctly for CPU/MPS/CUDA and warn when macOS-style worker settings are risky.
- **Setup**:
  - Test: `tests/test_runtime.py`
- **Command**: `.venv/bin/python -m pytest tests/test_runtime.py -x -v`
- **Status**: DONE
- **Result**:
  - `.venv/bin/python -m pytest tests/test_runtime.py -x -v` passed (`6/6`).
  - Verified `forward_dtype=auto` resolves to `float32` on CPU, `float16` on MPS, and `bfloat16` on CUDA by policy.
  - Verified explicit low-precision requests fall back safely on unsupported local targets and emit warnings.
- **Conclusion**: Runtime selection is now explicitly tested instead of being spread across entrypoints.
- **Follow-up**: Re-run supervised and distillation smokes after the runtime helper passes.

### FULL-REGRESSION-002: Full Test Suite After Runtime Unification
- **Date**: 2026-04-18
- **PLAN.md step**: 2.1, 3.1, 4.1, 4.2
- **Hypothesis**: Unifying runtime selection across training, evaluation, UCI, and RL should preserve the existing test suite behavior.
- **Setup**:
  - Scope: full `tests/` suite after runtime and CLI changes
- **Command**: `.venv/bin/python -m pytest tests/ -x -v`
- **Status**: DONE
- **Result**:
  - `.venv/bin/python -m pytest tests/ -x -v` passed (`118 passed`) in `233.27s`.
  - Existing warnings were limited to `urllib3` LibreSSL and `matplotlib` deprecations; no new test failures were introduced by the runtime unification.
- **Conclusion**: Runtime, CLI, evaluation, and RL path changes did not regress the repo test suite.
- **Follow-up**: If the suite passes, attempt an MPS availability check for the sandboxed environment.

### SMOKE-TRAIN-MPS-001: Apple Silicon MPS Supervised Startup Validation
- **Date**: 2026-04-18
- **PLAN.md step**: 2.2
- **Hypothesis**: The supervised wrapper should resolve `--forward_dtype auto` to `float16` on MPS, emit live progress quickly, and keep `num_workers=0` for a low-RAM local run.
- **Setup**:
  - Command target: `scripts/s1_supervised.py`
  - Config: `mac_mini`
- **Command**: `.venv/bin/python scripts/s1_supervised.py --data data-lichess/phase1 --config mac_mini --epochs 1 --device mps --forward_dtype auto --num_workers 0 --accum_steps 1 --wandb --checkpoint_dir /tmp/hrm-gab-mps-smoke`
- **Status**: FAILED
- **Result**:
  - The real user-side MPS run reached the first batch and first carry initialization, then aborted inside Apple’s MPS matmul path with:
    `/AppleInternal/.../MPSNDArrayMatrixMultiplication.mm:641: failed assertion 'LORADOWN GEMV Kernel - matrixRowPadElements will overflow its fc bit allocation.'`
  - The failing command resolved to `device=mps forward_dtype=float16 num_workers=0`.
- **Conclusion**: `forward_dtype=auto -> float16` is not stable for this project on the current Apple/PyTorch stack. The default MPS auto-dtype needs to fall back to float32.
- **Artifacts**:
  - Checkpoint: `/tmp/hrm-gab-mps-smoke`
  - W&B run: `https://wandb.ai/chess/hrm-gab-chess/runs/u64fwhzr`
- **Follow-up**: Patch runtime auto-dtype policy for MPS and ask for a rerun on the real machine.

### BUG-004: MPS Float16 Training Aborts in First Matmul
- **Date**: 2026-04-18
- **PLAN.md step**: 2.1, 2.2
- **Symptoms**:
  - Real MPS supervised training aborts after the first batch with an Apple MPS matmul assertion.
  - Failure appears on the first model forward pass after `Runtime: device=mps forward_dtype=float16 num_workers=0`.
- **Reproduction**:
  - `.venv/bin/python scripts/s1_supervised.py --data data-lichess/phase1 --config mac_mini --epochs 1 --device mps --forward_dtype auto --num_workers 0 --accum_steps 1 --wandb --checkpoint_dir /tmp/hrm-gab-mps-smoke`
- **Status**: DONE
- **Result**:
  - Updated the shared runtime policy so `forward_dtype=auto` now resolves to stable `float32` on MPS.
  - Kept `--forward_dtype float16` available as explicit opt-in and emit a warning that float16 MPS matmul can crash on some Apple/PyTorch stacks.
  - Updated runtime tests and reran the full test suite successfully after the policy change.
- **Conclusion**: The repo default now prioritizes MPS stability over risky float16 speedups. Real hardware rerun is still needed to confirm the original command no longer aborts.
- **Follow-up**: Make MPS `auto` resolve to stable `float32`, keep `float16` as explicit opt-in, then rerun validation.

### SMOKE-RUNTIME-002: MPS Stability Runtime Policy Regression
- **Date**: 2026-04-18
- **PLAN.md step**: 2.1
- **Hypothesis**: After the MPS crash, `forward_dtype=auto` should resolve to stable `float32` on MPS while keeping explicit `float16` available as opt-in.
- **Setup**:
  - Test: `tests/test_runtime.py`
- **Command**: `.venv/bin/python -m pytest tests/test_runtime.py -x -v`
- **Status**: DONE
- **Result**:
  - `.venv/bin/python -m pytest tests/test_runtime.py -x -v` passed (`7/7`).
  - Verified `forward_dtype=auto` now resolves to `float32` on MPS and explicit `float16` remains opt-in with a warning.
- **Conclusion**: The runtime helper now encodes the safer MPS default explicitly.
- **Follow-up**: Re-run the full suite after the runtime policy change.

### FULL-REGRESSION-003: Full Test Suite After MPS Stability Fallback
- **Date**: 2026-04-18
- **PLAN.md step**: 2.1, 3.1, 4.1, 4.2
- **Hypothesis**: Switching MPS auto-dtype from float16 to float32 should preserve repo behavior while avoiding the newly discovered Apple MPS matmul crash.
- **Setup**:
  - Scope: full `tests/` suite after the MPS runtime fallback
- **Command**: `.venv/bin/python -m pytest tests/ -x -v`
- **Status**: DONE
- **Result**:
  - `.venv/bin/python -m pytest tests/ -x -v` passed (`119 passed`) in `197.54s`.
  - The only warnings were the pre-existing `urllib3` LibreSSL and `matplotlib` deprecation warnings.
- **Conclusion**: The MPS stability fallback did not regress the repo test suite.

### BUG-005: setup_mac_env.sh Could Recreate .venv From the Active 3.9 Interpreter
- **Date**: 2026-04-18
- **PLAN.md step**: 0.1
- **Symptoms**:
  - Running `bash scripts/setup_mac_env.sh --recreate` from inside the active `.venv` could rebuild `.venv` using `.venv/bin/python3`, preserving the old Python 3.9 interpreter.
- **Fix**:
  - Updated `scripts/setup_mac_env.sh` to prefer an external Python 3.13 launcher (`python3.13`, Homebrew, then other `python3`) and to skip interpreters inside the active `VIRTUAL_ENV` unless `PYTHON_BIN` is set explicitly.
  - Verified the shell script syntax with `bash -n scripts/setup_mac_env.sh`.
- **Status**: DONE
- **Conclusion**: The environment bootstrap script is now robust against recreating the venv from the currently activated old venv.
- **Follow-up**: Ask for a user-side rerun of the MPS supervised smoke.
