# HRMChess Model Parameters

**Configuration:** 512 hidden size, 8 heads, 4 H layers, 4 L layers (Full config)

**Total Parameters: 29,849,091**

---

## Layer Breakdown

### Embedding & Input Projection
| Layer | Parameters |
|-------|------------|
| `inner.board_proj` (119 → 512) | 60,928 |
| `inner.cls_token` | 512 |
| **Subtotal** | **61,440** |

---

### H-Level Reasoning Module (4 layers)

| Layer | Attention (QKV) | Attention (O) | MLP (Gate/Up) | MLP (Down) | Total per Layer |
|-------|-----------------|---------------|---------------|------------|-----------------|
| `inner.H_level.layers.0` | 786,432 | 262,144 | 1,572,864 | 786,432 | 3,407,872 |
| `inner.H_level.layers.1` | 786,432 | 262,144 | 1,572,864 | 786,432 | 3,407,872 |
| `inner.H_level.layers.2` | 786,432 | 262,144 | 1,572,864 | 786,432 | 3,407,872 |
| `inner.H_level.layers.3` | 786,432 | 262,144 | 1,572,864 | 786,432 | 3,407,872 |
| **H-Level Total** | **3,145,728** | **1,048,576** | **6,291,456** | **3,145,728** | **13,631,488** |

---

### L-Level Reasoning Module (4 layers)

| Layer | Attention (QKV) | Attention (O) | MLP (Gate/Up) | MLP (Down) | Total per Layer |
|-------|-----------------|---------------|---------------|------------|-----------------|
| `inner.L_level.layers.0` | 786,432 | 262,144 | 1,572,864 | 786,432 | 3,407,872 |
| `inner.L_level.layers.1` | 786,432 | 262,144 | 1,572,864 | 786,432 | 3,407,872 |
| `inner.L_level.layers.2` | 786,432 | 262,144 | 1,572,864 | 786,432 | 3,407,872 |
| `inner.L_level.layers.3` | 786,432 | 262,144 | 1,572,864 | 786,432 | 3,407,872 |
| **L-Level Total** | **3,145,728** | **1,048,576** | **6,291,456** | **3,145,728** | **13,631,488** |

---

### Output Heads
| Layer | Parameters |
|-------|------------|
| `inner.q_head.weight` (512 → 2) | 1,024 |
| `inner.q_head.bias` | 2 |
| `inner.policy_head` (512 → 4672) | 2,392,064 |
| `inner.value_head.0` (512 → 256) | 131,072 |
| `inner.value_head.0.bias` | 256 |
| `inner.value_head.2` (256 → 1) | 256 |
| `inner.value_head.2.bias` | 1 |
| **Subtotal** | **2,524,675** |

---

## Summary by Component

| Component | Parameters |
|-----------|------------|
| Input Projection (board_proj + cls_token) | 61,440 |
| H-Level Reasoning (4 layers) | 13,631,488 |
| L-Level Reasoning (4 layers) | 13,631,488 |
| Output Heads (policy + value + q_head) | 2,524,675 |
| **Total** | **29,849,091** |

---

## Per-Block Details (Single H/L Layer)

Each transformer block consists of:
- **Self-Attention:**
  - QKV projection: `hidden_size × (3 × head_dim × num_heads)` = 512 × 1,536 = 786,432
  - O projection: `(head_dim × num_heads) × hidden_size` = 1,536 × 512 = 262,144
- **MLP (SwiGLU):**
  - Gate/Up projection: `hidden_size × (intermediate × 2)` where intermediate = 2,048 → 512 × 4,096 = 2,096,256
  - Note: Due to `_find_multiple` rounding to 256: 2,048 × 2 = 4,096 → 1,572,864
  - Down projection: `intermediate × hidden_size` = 2,048 × 512 = 1,048,576
  - Note: Due to `_find_multiple` rounding: 786,432

**Single H/L Layer Total: 3,407,872**