import torch

from chessgame.model.hrm_chess_config import HRMChessConfig
from chessgame.train.runtime import resolve_training_runtime


class TestRuntimeResolution:
    def test_auto_device_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        cfg = HRMChessConfig.mac_mini()
        runtime = resolve_training_runtime(cfg, device_str="auto", forward_dtype_str="auto")

        assert runtime.device.type == "cpu"
        assert runtime.forward_dtype == torch.float32
        assert runtime.forward_dtype_name == "float32"
        assert runtime.num_workers == 0
        assert cfg.forward_dtype == "float32"

    def test_auto_dtype_prefers_float32_on_mps_for_stability(self):
        cfg = HRMChessConfig.mac_mini()
        runtime = resolve_training_runtime(cfg, device_str="mps", forward_dtype_str="auto")

        assert runtime.device.type == "mps"
        assert runtime.forward_dtype == torch.float32
        assert runtime.forward_dtype_name == "float32"
        assert runtime.num_workers == 0
        assert cfg.forward_dtype == "float32"

    def test_explicit_bfloat16_on_mps_falls_back_to_float32(self):
        cfg = HRMChessConfig.full()
        runtime = resolve_training_runtime(
            cfg,
            device_str="mps",
            forward_dtype_str="bfloat16",
        )

        assert runtime.forward_dtype == torch.float32
        assert runtime.forward_dtype_name == "float32"
        assert any("bfloat16" in warning for warning in runtime.warnings)

    def test_explicit_float16_on_mps_is_opt_in_with_warning(self):
        cfg = HRMChessConfig.mac_mini()
        runtime = resolve_training_runtime(
            cfg,
            device_str="mps",
            forward_dtype_str="float16",
        )

        assert runtime.forward_dtype == torch.float16
        assert runtime.forward_dtype_name == "float16"
        assert any("experimental" in warning for warning in runtime.warnings)

    def test_explicit_low_precision_on_cpu_falls_back_to_float32(self):
        cfg = HRMChessConfig.full()
        runtime = resolve_training_runtime(
            cfg,
            device_str="cpu",
            forward_dtype_str="float16",
        )

        assert runtime.forward_dtype == torch.float32
        assert runtime.forward_dtype_name == "float32"
        assert any("float32" in warning for warning in runtime.warnings)

    def test_cpu_or_mps_workers_warn_when_explicitly_enabled(self):
        cfg = HRMChessConfig.mac_mini()
        runtime = resolve_training_runtime(
            cfg,
            device_str="cpu",
            num_workers=2,
            forward_dtype_str="auto",
        )

        assert runtime.num_workers == 2
        assert any("num_workers=2" in warning for warning in runtime.warnings)

    def test_cuda_defaults_to_four_workers(self):
        cfg = HRMChessConfig.full()
        runtime = resolve_training_runtime(
            cfg,
            device_str="cuda",
            forward_dtype_str="auto",
        )

        assert runtime.device.type == "cuda"
        assert runtime.num_workers == 4
        assert runtime.forward_dtype_name == "bfloat16"
