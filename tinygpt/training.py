from __future__ import annotations

import json
from pathlib import Path

import torch

from config import ModelConfig, TrainConfig
from .model import TinyGPT


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(0, len(data) - context_length, (batch_size,))
    x = torch.stack([data[index : index + context_length] for index in indices])
    y = torch.stack([data[index + 1 : index + 1 + context_length] for index in indices])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: TrainConfig,
    context_length: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}

    for split_name, split_data in (("train", train_data), ("val", val_data)):
        split_losses = []
        for _ in range(config.eval_iters):
            x, y = get_batch(split_data, config.batch_size, context_length, device)
            _, loss = model(x, y)
            split_losses.append(float(loss.item()))
        losses[split_name] = sum(split_losses) / len(split_losses)

    model.train()
    return losses


def train_model(
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: str,
) -> dict[str, list[float] | list[int]]:
    history: dict[str, list[float] | list[int]] = {"steps": [], "train": [], "val": []}
    model.train()

    for step in range(train_config.max_iters + 1):
        x, y = get_batch(train_data, train_config.batch_size, model_config.context_length, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % train_config.eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                train_config,
                model_config.context_length,
                device,
            )
            history["steps"].append(step)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            print(
                f"Iter {step:4d} | Train loss: {losses['train']:.4f} | "
                f"Val loss: {losses['val']:.4f}"
            )

    return history


def save_checkpoint(
    path: str | Path,
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config.to_dict(),
            "train_config": train_config.to_dict(),
        },
        checkpoint_path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[TinyGPT, ModelConfig, dict]:
    checkpoint = torch.load(path, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = TinyGPT(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, model_config, checkpoint


def save_history(path: str | Path, history: dict[str, list[float] | list[int]]) -> None:
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")