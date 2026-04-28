from dataclasses import asdict, dataclass

@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 65 # set by default to match tiny shakespeare
    context_length: int = 400
    n_layers: int = 10 # number of transformer layers
    n_heads: int = 8 # number of attention heads
    n_embd: int = 400 # embedding dimension, must be divisible by n_heads
    dropout: float = 0.1 # dropout rate for regularization

    def __post_init__(self) -> None:
        # sanity checks prior to training
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.n_embd % self.n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")

    def to_dict(self) -> dict[str, int | float]:
        # convert to dict for easier logging and checkpointing
        return asdict(self)