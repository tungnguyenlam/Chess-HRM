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

<!-- NEW ENTRIES GO BELOW THIS LINE -->
