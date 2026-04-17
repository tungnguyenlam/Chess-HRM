# Agent Guide — HRM-GAB Chess

> **Read this file FIRST before touching any code.**
> This document orients you to the codebase, establishes workflow rules, and defines how to behave like a rigorous researcher — not a code monkey.

# Reasoning & Problem-Solving Standards

1. **Decompose First**: For non-trivial tasks, break the problem into logical sub-steps.
2. **Verify Assumptions**: Explicitly state any assumptions made. If data is missing, note it.
3. **Iterative Refinement**: If generating code or architecture, consider edge cases and potential failures.
4. **Clarity Over Cleverness**: Prefer clear, maintainable solutions over overly complex ones.
5. **Adapt Depth**: Match the depth of your analysis to the complexity of the user’s request. Do not over-engineer simple answers.

---

## 1. Project Overview

**Goal**: Build a novel chess-playing model (**HRM-GAB**) that combines:
- **HRM** (Hierarchical Reasoning Model) — adaptive computation depth via ACT (Adaptive Computation Time)
- **GAB** (Geometric Attention Bias) — chess-specific dynamic positional encoding from Chessformer

**Why this is novel**: No prior work combines adaptive compute allocation with chess-specific geometric attention in a recurrent architecture. See `PLAN.md §Novelty` for the full argument.

**Target**: Publishable paper at NeurIPS/ICLR/AAAI-tier venue.

---

## 2. Repository Structure

```
Chess-HRM/
├── AGENTS.md                    ← YOU ARE HERE. Read first.
├── PLAN.md                      ← Implementation steps. Mark [V] when done.
├── BACKLOG.md                   ← Experiment log. Record EVERYTHING here.
├── README.md                    ← Public-facing project description (HRM upstream)
├── Modelparams.md               ← Parameter count breakdown
│
├── config/
│   ├── chess.yaml               ← Hyperparameters (full + mac_mini profiles)
│   └── arch/                    ← Upstream HRM architecture configs
│
├── chessmodels/                 ← MODEL IMPLEMENTATIONS (core)
│   ├── hrm/
│   │   ├── hrm_act_v1.py       ← ★ UPSTREAM HRM — DO NOT MODIFY
│   │   └── README.md           ← HRM internals documentation
│   ├── layers.py               ← CastedLinear, shared layer primitives
│   ├── losses.py               ← Loss functions (CE, MSE, ACT, KL)
│   ├── sparse_embedding.py     ← Sparse embedding utilities
│   └── common.py               ← Shared constants
│
├── chessgame/                   ← CHESS-SPECIFIC CODE
│   ├── model/
│   │   ├── hrm_chess.py        ← ★ HRMChess wrapper — OUR MAIN MODEL
│   │   ├── hrm_chess_config.py ← Model config dataclass
│   │   └── rope_2d.py          ← 2D RoPE for 8×8 board + CLS
│   ├── encoding/               ← Board → tensor encoding
│   ├── data/                   ← Dataset loaders
│   └── train/                  ← Training loops
│
├── scripts/                     ← One-off scripts (annotation, eval, etc.)
├── utils/                       ← General utilities
├── evaluate.py                  ← Evaluation entry point
├── evaluate_chess.py            ← Chess-specific evaluation
│
├── ReinforcementLearninginChess/ ← Research notes (read-only reference)
│   ├── Papers/                  ← PDF + summary notes of key papers
│   └── Chatdumps/               ← Prior architecture debates
│
└── archive/                     ← Deprecated code (do not use)
```

---

## 3. Mandatory Reading Order

**Before writing ANY code**, read these files in this exact order:

| # | File | Why |
|---|------|-----|
| 1 | `PLAN.md` | Understand what's been done and what's next |
| 2 | `BACKLOG.md` | Understand what's been tried and what failed |
| 3 | `config/chess.yaml` | Understand the hyperparameter space |
| 4 | `chessmodels/hrm/hrm_act_v1.py` | Understand the upstream HRM internals — ACT loop, Q-head, one-step gradient trick |
| 5 | `chessgame/model/hrm_chess.py` | Understand our chess adapter — how HRMChessInner and HRMChess wrap the upstream |
| 6 | `chessgame/model/hrm_chess_config.py` | Understand config fields |
| 7 | `chessmodels/layers.py` | Understand CastedLinear and shared primitives |
| 8 | `chessmodels/losses.py` | Understand loss functions |

**For research context** (optional but recommended):
- `ReinforcementLearninginChess/Papers/2026-CHESSFORNER A UNIFIED.md` — Chessformer/GAB details
- `ReinforcementLearninginChess/Papers/2024-Armortized Planing with Large-Scale.md` — Searchless chess baseline
- `ReinforcementLearninginChess/Chatdumps/ConcatPlan.md` — Architecture debate synthesis

---

## 4. Critical Invariants (DO NOT VIOLATE)

These are hard rules. Breaking them will silently corrupt the project.

### 4.1 Upstream HRM is READ-ONLY

```
chessmodels/hrm/hrm_act_v1.py  →  NEVER MODIFY THIS FILE
```

All chess-specific changes go in `chessgame/model/hrm_chess.py` or new files. If upstream HRM needs a patch, subclass or monkey-patch in `chessgame/`.

**Rationale**: We need to be able to pull upstream HRM updates without merge conflicts. Our paper's contribution is *on top of* HRM, not a fork of it.

### 4.2 One-Step Gradient Trick

The HRM training loop runs H/L cycles under `torch.no_grad()` and only tracks gradients for the **final** iteration. This is not a bug — it's the core memory optimization that makes HRM trainable.

```python
# CORRECT — in hrm_chess.py forward()
with torch.no_grad():
    for _H in range(H_cycles):
        for _L in range(L_cycles):
            if not last_iter:
                z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        if _H != H_cycles - 1:
            z_H = self.H_level(z_H, z_L, **seq_info)

# Final iteration WITH gradients
z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
z_H = self.H_level(z_H, z_L, **seq_info)
```

**DO NOT** remove `torch.no_grad()` to "get more gradient signal." You will OOM immediately.

### 4.3 ACT Training Schedule

| Phase | ACT Q-head | Reason |
|-------|:---:|--------|
| Supervised (Phase 1) | **OFF** | No reward signal to learn halt/continue |
| Distillation (Phase 2) | **OFF** | Same reason |
| RL Self-Play (Phase 3+) | **ON** | Game outcomes provide reward signal |

### 4.4 Board Encoding

Always use the **full AlphaZero 8×8×119 encoding**:
- 96 planes: 12 piece types × 8 history plies
- 23 auxiliary planes: castling, side to play, repetition, move counters

Simplified encodings (e.g., 8×8×14 without history) produce a fundamentally broken engine that cannot detect repetition or the 50-move rule.

### 4.5 Move Encoding

4672-dimensional action space (AlphaZero convention: 8×8×73 flattened). Do not invent a new encoding.

---

## 5. Workflow Rules

### 5.1 Before Starting ANY Task

1. Read `PLAN.md` — find the next unchecked `[ ]` item
2. Read `BACKLOG.md` — check if someone already attempted this and failed
3. Identify the files you'll touch
4. State your plan in a comment before writing code

### 5.2 Before Modifying PLAN.md

**PLAN.md is the source of truth.** Modifying it is a significant action.

1. Create a `PLAN-CHANGE-XXX` entry in `BACKLOG.md` explaining **what** you're changing and **why**.
2. Append a row to the `PLAN.md` Changelog table.
3. Never delete a step — mark it `[X]` with a reason.
4. Scope changes (adding/removing entire phases) require user approval.

```
# Example BACKLOG entry:
### PLAN-CHANGE-001: Add GAB memory optimization step
- **Date**: 2026-05-01
- **What changed**: Added step 1.1.1 "Implement GAB gradient checkpointing"
- **Why**: GAB module OOMs at batch_size=256 (discovered in SMOKE-001)
- **Impact**: No other steps affected; this is additive
```

### 5.3 During Development

1. **One logical change per commit.** No "fix everything" commits.
2. **Every new module gets a docstring** explaining:
   - What it does
   - What it expects as input
   - What it produces as output
   - Which paper/section it implements
3. **Every hyperparameter** must be in `config/chess.yaml`, not hardcoded.
4. **Every experiment** must be logged in `BACKLOG.md` before AND after running.

### 5.4 Testing Gate (MANDATORY)

**No step in `PLAN.md` may be marked `[V]` until its tests pass.**

1. Every implementation step must have a corresponding test in `tests/`.
2. Run `python -m pytest tests/ -x -v` after completing a step. The `-x` flag stops on first failure.
3. If tests fail, fix the code. Do NOT mark the step `[V]` and move on.
4. If a step has no natural test (e.g., "download dataset"), create a validation check (e.g., verify file exists, row count matches, sample decoding round-trips).

```
# Gate check before marking step done:
python -m pytest tests/ -x -v

# If targeting a specific step's tests:
python -m pytest tests/test_board_encoder.py -x -v
```

### 5.5 After Completing a Task

1. Verify tests pass (section 5.4)
2. Mark the item `[V]` in `PLAN.md`
3. Log results in `BACKLOG.md` (see format below)
4. Update `config/chess.yaml` if new hyperparameters were introduced

### 5.6 Experiment Logging Protocol

**Before running an experiment**, add an entry to `BACKLOG.md`:

```markdown
### EXP-XXX: [Title]
- **Date**: YYYY-MM-DD
- **Hypothesis**: What you expect to happen and why
- **Config**: Which yaml profile, key hyperparameters
- **Status**: RUNNING
```

**After the experiment**, update the entry:

```markdown
- **Status**: DONE
- **Result**: [metrics, observations]
- **Conclusion**: What did we learn? Does this change the plan?
- **Artifacts**: [checkpoint path, W&B run link, plot paths]
```

---

## 6. Scientific Rigor Checklist

Apply this checklist before claiming any result:

- [ ] **Reproducibility**: Can someone else run the exact same command and get the same result? (Record seeds, hardware, software versions)
- [ ] **Baseline comparison**: Did you compare against the correct baseline? (Not a strawman)
- [ ] **Ablation isolation**: Did you change only ONE variable? If multiple changed, the result is uninterpretable.
- [ ] **Statistical significance**: For Elo estimates, did you play enough games? (≥100 games per data point)
- [ ] **Failure documentation**: If something failed, did you record WHY in BACKLOG.md? Negative results are data.

---

## 7. Key Technical Decisions & Rationale

| Decision | Chosen | Rejected | Why |
|----------|--------|----------|-----|
| Positional encoding | GAB (dynamic) | RoPE, learned, sinusoidal | Chess geometry is position-dependent; GAB adapts per board state |
| Input tokenization | 64 squares + CLS | CNN front-end, FEN string | Preserves 2D structure for attention; aligns with Chessformer |
| Optimizer | AdamW | M3, SGD, CMS | Proven in HRM codebase; exotic optimizers are untested risk |
| Search | MCTS (batched) | SearchFormer, pure inference | MCTS is proven for strong play; SearchFormer is research-stage |
| Architecture base | HRM-ACT v1 | Nested Learning/Hope/Titans | HRM code is tested; mixing papers creates untestable "Franken-model" |

---

## 8. Common Pitfalls

| Pitfall | How to Avoid |
|---------|-------------|
| OOM during training | Never remove `torch.no_grad()` from H/L cycle loop |
| ACT learns garbage | Only enable ACT loss during RL phase (Phase 3+) |
| Illegal moves in play | Always mask policy output with legal move mask before selection |
| Board encoding bugs | Use AlphaZero 119-plane encoding; flip board for Black |
| Slow MCTS | Batch leaf evaluations (8-16 per GPU call); never evaluate 1 at a time |
| Loss divergence in RL | Anneal Stockfish KL regularizer from 0.3→0 over 50K steps |
| MPS bugs on Mac | If bfloat16 crashes on MPS, fall back to float16 or float32 |

---

## 9. File Creation Conventions

When creating new files:

```
chessgame/
├── model/
│   ├── gab.py              ← NEW: Geometric Attention Bias module
│   └── ...
├── encoding/
│   ├── board_encoder.py     ← Board → 8×8×119 tensor
│   ├── move_encoder.py      ← Move → 4672-dim index
│   └── ...
├── data/
│   ├── lichess_dataset.py   ← Lichess data loader
│   ├── stockfish_annotator.py ← SF annotation pipeline
│   └── ...
├── train/
│   ├── supervised.py        ← Phase 1 training loop
│   ├── distillation.py      ← Phase 2 training loop
│   ├── self_play.py         ← Phase 3-4 training loop
│   └── mcts.py              ← MCTS implementation
└── eval/
    ├── arena.py             ← Elo estimation via tournament
    ├── puzzles.py           ← Puzzle accuracy evaluation
    └── interpretability.py  ← GAB visualization, ACT depth analysis
```

---

## 10. Contact & References

- **Upstream HRM repo**: https://github.com/sapientinc/HRM
- **Chessformer paper**: See `ReinforcementLearninginChess/Papers/2026-CHESSFORNER A UNIFIED.md`
- **Architecture debate**: See `ReinforcementLearninginChess/Chatdumps/ConcatPlan.md`
- **Config reference**: `config/chess.yaml`
