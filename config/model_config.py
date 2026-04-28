from dataclasses import asdict, dataclass
from typing import Iterable

import torch

@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 65 # set by default to match tiny shakespeare
    context_length: int = 400
    n_layers: int = 10 # number of transformer layers
    n_heads: int = 8 # number of attention heads
    n_embd: int = 400 # embedding dimension, must be divisible by n_heads
    dropout: float = 0.1 # dropout rate for regularization
    optimizer_name: str = "adamw" # optimizer, adamw is probably the best choice

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
        if self.optimizer_name.lower() not in {"adamw", "adam", "sgd", "rmsprop"}:
            raise ValueError("optimizer_name must be one of: adamw, adam, sgd, rmsprop")

    def to_dict(self) -> dict[str, int | float | str]:
        # convert to dict for easier logging and checkpointing
        return asdict(self)

    def build_optimizer(
        self,
        parameters: Iterable[torch.nn.Parameter],
        learning_rate: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        name = self.optimizer_name.lower()
        if name == "adamw":
            return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
        if name == "adam":
            return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        if name == "sgd":
            return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        if name == "rmsprop":
            return torch.optim.RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay)
        raise ValueError(f"Unsupported optimizer_name: {self.optimizer_name}")