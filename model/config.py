from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # Vocabulary & sequences
    vocab_size: int = 32000
    max_seq_len: int = 4096
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Architecture dimensions
    hidden_size: int = 2048
    num_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8          # GQA: fewer KV heads than Q heads
    head_dim: Optional[int] = None        # defaults to hidden_size // num_attention_heads
    intermediate_size: int = 5504         # FFN hidden dim (≈ 2.7 * hidden_size for SwiGLU)

    # Normalization
    rms_norm_eps: float = 1e-5

    # Positional encoding
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None  # scale factor for extended context

    # Regularization
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Architecture flags
    use_bias: bool = False                # No bias in linear layers (modern LLMs)
    tie_word_embeddings: bool = True      # Share input/output embedding weights

    # Precision & memory
    use_flash_attention: bool = True      # Use Flash Attention if available
    use_gradient_checkpointing: bool = False

    # Initialization
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size {self.hidden_size} must be divisible by num_attention_heads {self.num_attention_heads}"
        )
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"num_attention_heads {self.num_attention_heads} must be divisible by num_key_value_heads {self.num_key_value_heads}"
        )

    @classmethod
    def small(cls) -> "ModelConfig":
        """~125M parameters — for testing on CPU/single GPU."""
        return cls(
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
            intermediate_size=2048,
            max_seq_len=2048,
        )

    @classmethod
    def medium(cls) -> "ModelConfig":
        """~1B parameters."""
        return cls(
            hidden_size=2048,
            num_layers=24,
            num_attention_heads=16,
            num_key_value_heads=8,
            intermediate_size=5504,
            max_seq_len=4096,
        )

    @classmethod
    def large(cls) -> "ModelConfig":
        """~7B parameters."""
        return cls(
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=11008,
            max_seq_len=4096,
        )

    @classmethod
    def xl(cls) -> "ModelConfig":
        """~70B parameters — Claude Sonnet-scale."""
        return cls(
            hidden_size=8192,
            num_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,
            intermediate_size=28672,
            max_seq_len=8192,
        )


@dataclass
class TrainingConfig:
    # Optimization
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 2000
    max_steps: int = 100_000
    lr_decay_steps: Optional[int] = None  # defaults to max_steps

    # Batching
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    micro_batch_size: int = 4

    # Precision
    dtype: str = "bfloat16"              # "float32", "float16", "bfloat16"

    # Checkpointing
    save_every_steps: int = 1000
    eval_every_steps: int = 500
    checkpoint_dir: str = "checkpoints"
    keep_last_n_checkpoints: int = 3

    # Logging
    log_every_steps: int = 10
    wandb_project: Optional[str] = None

    # Data
    data_dir: str = "data/processed"
    val_split: float = 0.005

    # Distributed
    num_workers: int = 4

    def __post_init__(self):
        if self.lr_decay_steps is None:
            self.lr_decay_steps = self.max_steps


@dataclass
class RLHFConfig:
    # SFT
    sft_learning_rate: float = 2e-5
    sft_epochs: int = 3
    sft_batch_size: int = 16

    # Reward model
    rm_learning_rate: float = 2e-5
    rm_epochs: int = 2
    rm_batch_size: int = 16

    # PPO
    ppo_learning_rate: float = 1e-5
    ppo_epochs: int = 4
    ppo_batch_size: int = 64
    ppo_mini_batch_size: int = 8
    kl_coef: float = 0.1
    clip_range: float = 0.2
    vf_coef: float = 0.1
    target_kl: float = 0.1
    gamma: float = 1.0
    lam: float = 0.95

    # DPO (alternative to PPO)
    dpo_beta: float = 0.1
    dpo_learning_rate: float = 5e-7
    dpo_epochs: int = 3

    # Constitutional AI
    num_cai_revisions: int = 3
    cai_batch_size: int = 8
