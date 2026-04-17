"""Tests for Phase 1: GAB module, attention bias injection, and full model smoke test.

Covers PLAN.md steps 1.1, 1.2, 1.3, 1.4, 1.5.
"""

import torch
import pytest

from chessgame.model.gab import GeometricAttentionBias
from chessgame.model.attention_bias import AttentionWithBias
from chessgame.model.hrm_chess import HRMChess
from chessgame.model.hrm_chess_config import HRMChessConfig


# Use CPU + float32 for testing (MPS can have precision issues)
DEVICE = torch.device("cpu")
DTYPE = torch.float32


@pytest.fixture
def mac_cfg():
    return HRMChessConfig.mac_mini()


@pytest.fixture
def mac_cfg_no_gab():
    return HRMChessConfig.mac_mini_no_gab()


# ---------------------------------------------------------------------------
# Step 1.1: GAB Module
# ---------------------------------------------------------------------------


class TestGABModule:
    """GeometricAttentionBias unit tests."""

    def test_output_shape(self):
        gab = GeometricAttentionBias(
            hidden_size=256, num_heads=4, seq_len=65, compress_dim=64
        )
        z_H = torch.randn(2, 65, 256)
        out = gab(z_H)
        assert out.shape == (2, 4, 65, 65), f"Expected (2,4,65,65), got {out.shape}"

    def test_output_dtype_matches_input(self):
        gab = GeometricAttentionBias(hidden_size=256, num_heads=4, seq_len=65)
        z_H = torch.randn(2, 65, 256)
        out = gab(z_H)
        assert out.dtype == z_H.dtype

    def test_gradient_flow(self):
        """GAB should allow gradients to flow through."""
        gab = GeometricAttentionBias(hidden_size=256, num_heads=4, seq_len=65)
        z_H = torch.randn(2, 65, 256, requires_grad=True)
        out = gab(z_H)
        loss = out.sum()
        loss.backward()
        assert z_H.grad is not None
        assert z_H.grad.shape == z_H.shape

    def test_no_nan(self):
        gab = GeometricAttentionBias(hidden_size=256, num_heads=4, seq_len=65)
        z_H = torch.randn(2, 65, 256)
        out = gab(z_H)
        assert not torch.isnan(out).any(), "GAB output contains NaN"

    def test_zero_init(self):
        """GAB output should be near zero at initialization (zero init on bias_proj)."""
        gab = GeometricAttentionBias(hidden_size=256, num_heads=4, seq_len=65)
        z_H = torch.randn(2, 65, 256)
        out = gab(z_H)
        # With zero-init on bias_proj, dynamic bias should be 0 regardless of input
        # Only static_templates contribute (also initialized to 0)
        assert out.abs().max() < 1e-4, (
            f"GAB should be near-zero at init, max={out.abs().max()}"
        )

    def test_static_templates_disabled(self):
        gab = GeometricAttentionBias(
            hidden_size=256, num_heads=4, seq_len=65, use_static_templates=False
        )
        assert not hasattr(gab, "static_templates") or not gab.use_static_templates
        z_H = torch.randn(2, 65, 256)
        out = gab(z_H)
        assert out.shape == (2, 4, 65, 65)

    def test_different_inputs_different_outputs(self):
        """Different z_H should produce different biases (after training)."""
        gab = GeometricAttentionBias(hidden_size=256, num_heads=4, seq_len=65)
        # Manually set non-zero weights to ensure differentiation
        with torch.no_grad():
            gab.bias_proj.weight.normal_(0, 0.01)
        z_H1 = torch.randn(1, 65, 256)
        z_H2 = torch.randn(1, 65, 256)
        out1 = gab(z_H1)
        out2 = gab(z_H2)
        assert not torch.allclose(out1, out2), (
            "Different inputs should produce different biases"
        )

    def test_batch_independence(self):
        """Each sample in batch should get independent bias."""
        gab = GeometricAttentionBias(hidden_size=256, num_heads=4, seq_len=65)
        with torch.no_grad():
            gab.bias_proj.weight.normal_(0, 0.01)
        z_H = torch.randn(4, 65, 256)
        out = gab(z_H)
        # Sample 0 and sample 1 should have different biases
        assert not torch.allclose(out[0], out[1])


# ---------------------------------------------------------------------------
# Step 1.3: AttentionWithBias
# ---------------------------------------------------------------------------


class TestAttentionWithBias:
    """Verify attention bias injection changes output."""

    def test_with_and_without_bias(self):
        """Output should differ when GAB bias is applied."""
        attn = AttentionWithBias(
            hidden_size=256,
            head_dim=64,
            num_heads=4,
            num_key_value_heads=4,
            causal=False,
        )
        cos = torch.ones(65, 64)
        sin = torch.zeros(65, 64)
        x = torch.randn(2, 65, 256)
        bias = torch.randn(2, 4, 65, 65) * 0.1

        out_no_bias = attn(cos_sin=(cos, sin), hidden_states=x, attn_bias=None)
        out_with_bias = attn(cos_sin=(cos, sin), hidden_states=x, attn_bias=bias)

        assert not torch.allclose(out_no_bias, out_with_bias, atol=1e-5), (
            "Attention output should change with GAB bias"
        )

    def test_zero_bias_same_as_no_bias(self):
        """Zero bias should give approximately same result as no bias."""
        attn = AttentionWithBias(
            hidden_size=256,
            head_dim=64,
            num_heads=4,
            num_key_value_heads=4,
            causal=False,
        )
        cos = torch.ones(65, 64)
        sin = torch.zeros(65, 64)
        x = torch.randn(2, 65, 256)
        zero_bias = torch.zeros(2, 4, 65, 65)

        out_no_bias = attn(cos_sin=(cos, sin), hidden_states=x, attn_bias=None)
        out_zero_bias = attn(cos_sin=(cos, sin), hidden_states=x, attn_bias=zero_bias)

        assert torch.allclose(out_no_bias, out_zero_bias, atol=1e-4), (
            "Zero bias should match no-bias output"
        )

    def test_gradient_through_bias(self):
        """Gradients should flow through the attention bias."""
        attn = AttentionWithBias(
            hidden_size=256,
            head_dim=64,
            num_heads=4,
            num_key_value_heads=4,
            causal=False,
        )
        cos = torch.ones(65, 64)
        sin = torch.zeros(65, 64)
        x = torch.randn(2, 65, 256)
        bias = torch.randn(2, 4, 65, 65, requires_grad=True)

        out = attn(cos_sin=(cos, sin), hidden_states=x, attn_bias=bias)
        out.sum().backward()
        assert bias.grad is not None


# ---------------------------------------------------------------------------
# Step 1.5: Full Model Smoke Test
# ---------------------------------------------------------------------------


class TestHRMChessSmokeTest:
    """End-to-end model forward/backward on random input."""

    def test_forward_pass_shapes(self, mac_cfg):
        """Forward pass produces correct output shapes."""
        model = HRMChess(mac_cfg).to(DEVICE).to(DTYPE)
        model.eval()

        B = 2
        batch = {
            "inputs": torch.randn(B, 8, 8, 119, device=DEVICE, dtype=DTYPE),
            "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEVICE),
        }
        carry = model.initial_carry(batch)

        with torch.no_grad():
            carry, outputs = model(carry, batch)

        assert outputs["policy"].shape == (B, 4672), (
            f"Policy shape: {outputs['policy'].shape}"
        )
        assert outputs["value"].shape == (B, 1), (
            f"Value shape: {outputs['value'].shape}"
        )
        assert "q_halt_logits" in outputs
        assert "q_continue_logits" in outputs

    def test_backward_pass(self, mac_cfg):
        """Backward pass completes without error."""
        model = HRMChess(mac_cfg).to(DEVICE).to(DTYPE)
        model.train()

        B = 2
        batch = {
            "inputs": torch.randn(B, 8, 8, 119, device=DEVICE, dtype=DTYPE),
            "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEVICE),
        }
        carry = model.initial_carry(batch)
        carry, outputs = model(carry, batch)

        loss = outputs["policy"].sum() + outputs["value"].sum()
        loss.backward()

        # Check that parameters have gradients
        grad_params = [p for p in model.parameters() if p.grad is not None]
        assert len(grad_params) > 0, "No parameters received gradients"

    def test_gab_receives_gradients(self, mac_cfg):
        """GAB module parameters should receive gradients."""
        model = HRMChess(mac_cfg).to(DEVICE).to(DTYPE)
        model.train()

        B = 2
        batch = {
            "inputs": torch.randn(B, 8, 8, 119, device=DEVICE, dtype=DTYPE),
            "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEVICE),
        }
        carry = model.initial_carry(batch)
        carry, outputs = model(carry, batch)
        loss = outputs["policy"].sum()
        loss.backward()

        # Check GAB-specific params
        gab = model.inner.gab
        assert gab.bias_proj.weight.grad is not None, (
            "GAB bias_proj should have gradient"
        )

    def test_no_gab_ablation(self, mac_cfg_no_gab):
        """Model should work without GAB (ablation baseline)."""
        model = HRMChess(mac_cfg_no_gab).to(DEVICE).to(DTYPE)
        model.eval()

        B = 2
        batch = {
            "inputs": torch.randn(B, 8, 8, 119, device=DEVICE, dtype=DTYPE),
            "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEVICE),
        }
        carry = model.initial_carry(batch)

        with torch.no_grad():
            carry, outputs = model(carry, batch)

        assert outputs["policy"].shape == (B, 4672)
        assert not model.inner.gab_enabled

    def test_no_nan_in_outputs(self, mac_cfg):
        """No NaN in any output tensor."""
        model = HRMChess(mac_cfg).to(DEVICE).to(DTYPE)
        model.eval()

        B = 2
        batch = {
            "inputs": torch.randn(B, 8, 8, 119, device=DEVICE, dtype=DTYPE),
            "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEVICE),
        }
        carry = model.initial_carry(batch)

        with torch.no_grad():
            carry, outputs = model(carry, batch)

        for k, v in outputs.items():
            if isinstance(v, list):
                for i, item in enumerate(v):
                    assert not torch.isnan(item).any(), f"NaN in output '{k}' index {i}"
            else:
                assert not torch.isnan(v).any(), f"NaN in output '{k}'"

    def test_parameter_count(self, mac_cfg):
        """Parameter count sanity check for mac_mini config."""
        model = HRMChess(mac_cfg)
        total = sum(p.numel() for p in model.parameters())
        # mac_mini with GAB should be ~7-10M params
        assert 3_000_000 < total < 15_000_000, f"Unexpected param count: {total:,}"
        print(f"\nmac_mini params: {total:,}")

    def test_deterministic_eval(self, mac_cfg):
        """Same input should produce same output in eval mode."""
        model = HRMChess(mac_cfg).to(DEVICE).to(DTYPE)
        model.eval()

        B = 1
        x = torch.randn(B, 8, 8, 119, device=DEVICE, dtype=DTYPE)
        batch = {
            "inputs": x,
            "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEVICE),
        }

        carry1 = model.initial_carry(batch)
        carry2 = model.initial_carry(batch)

        with torch.no_grad():
            _, out1 = model(carry1, batch)
            _, out2 = model(carry2, batch)

        assert torch.allclose(out1["policy"], out2["policy"], atol=1e-5), (
            "Eval mode should be deterministic"
        )
