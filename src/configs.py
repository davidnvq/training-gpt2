from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False


@dataclass
class TrainingArguments:
    num_epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    eval_freq: int = 15
    eval_iter: int = 1
