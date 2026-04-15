from chessmodels.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Config


class HRMChessConfig(HierarchicalReasoningModel_ACTV1Config):
    board_input_dim: int = 119
    policy_size: int = 4672

    @classmethod
    def full(cls) -> "HRMChessConfig":
        """Server-scale config (~28M params)."""
        return cls(
            board_input_dim=119,
            policy_size=4672,
            batch_size=256,
            seq_len=65,           # 64 squares + 1 CLS
            vocab_size=1,         # unused — chess has no token vocab
            num_puzzle_identifiers=1,
            puzzle_emb_ndim=0,
            pos_encodings="rope", # parent expects this; chess uses rope_2d override
            hidden_size=512,
            num_heads=8,
            expansion=4.0,
            H_layers=4,
            L_layers=4,
            H_cycles=3,
            L_cycles=6,
            halt_max_steps=8,
            halt_exploration_prob=0.1,
        )

    @classmethod
    def mac_mini(cls) -> "HRMChessConfig":
        """Reduced config for local development (~7M params)."""
        return cls(
            board_input_dim=119,
            policy_size=4672,
            batch_size=8,
            seq_len=65,
            vocab_size=1,
            num_puzzle_identifiers=1,
            puzzle_emb_ndim=0,
            pos_encodings="rope",
            hidden_size=256,
            num_heads=4,
            expansion=4.0,
            H_layers=2,
            L_layers=2,
            H_cycles=2,
            L_cycles=4,
            halt_max_steps=4,
            halt_exploration_prob=0.1,
        )
