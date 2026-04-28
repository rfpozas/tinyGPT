from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ModelConfig, TrainConfig
from tinygpt import load_prepared_data, resolve_device, set_seed
from tinygpt.model import TinyGPT
from tinygpt.training import save_checkpoint, save_history, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small GPT model")
    parser.add_argument("--prepared-dir", default="data/processed", help="Prepared dataset directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=400)
    parser.add_argument("--n-layers", type=int, default=10)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=400)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--optimizer", default="adamw", help="adamw, adam, sgd, or rmsprop")
    parser.add_argument("--checkpoint-path", default="artifacts/tinygpt.pt")
    parser.add_argument("--history-path", default="artifacts/train_history.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    dataset = load_prepared_data(args.prepared_dir)
    model_config = ModelConfig(
        vocab_size=dataset.vocab_size,
        context_length=args.context_length,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_embd=args.n_embd,
        dropout=args.dropout,
        optimizer_name=args.optimizer,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        history_path=args.history_path,
    )

    if len(dataset.train_data) <= model_config.context_length:
        raise ValueError("Prepared training data is shorter than context_length")
    if len(dataset.val_data) <= model_config.context_length:
        raise ValueError("Prepared validation data is shorter than context_length")

    model = TinyGPT(model_config).to(device)
    optimizer = model_config.build_optimizer(
        model.parameters(),
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    print(f"Device: {device}")
    print(f"Model parameters: {model.num_parameters()}")
    print(
        f"Config -> layers={model_config.n_layers}, heads={model_config.n_heads}, "
        f"embd={model_config.n_embd}, context={model_config.context_length}, "
        f"optimizer={model_config.optimizer_name}"
    )

    history = train_model(
        model=model,
        optimizer=optimizer,
        train_data=dataset.train_data,
        val_data=dataset.val_data,
        model_config=model_config,
        train_config=train_config,
        device=device,
    )
    save_checkpoint(train_config.checkpoint_path, model, optimizer, model_config, train_config)
    save_history(train_config.history_path, history)

    print(f"Saved checkpoint to {train_config.checkpoint_path}")
    print(f"Saved training history to {train_config.history_path}")


if __name__ == "__main__":
    main()