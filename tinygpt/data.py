from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(slots=True)
class DatasetBundle:
    train_data: torch.Tensor
    val_data: torch.Tensor
    stoi: dict[str, int]
    itos: dict[int, str]
    vocab_size: int
    split_ratio: float


def load_text_corpus(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def prepare_dataset(text: str, split_ratio: float = 0.9) -> DatasetBundle:
    if not text:
        raise ValueError("input text corpus is empty")

    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for char, index in stoi.items()}
    encoded = torch.tensor([stoi[char] for char in text], dtype=torch.long)

    split_index = int(len(encoded) * split_ratio)
    train_data = encoded[:split_index]
    val_data = encoded[split_index:]

    return DatasetBundle(
        train_data=train_data,
        val_data=val_data,
        stoi=stoi,
        itos=itos,
        vocab_size=len(chars),
        split_ratio=split_ratio,
    )


def save_prepared_data(bundle: DatasetBundle, out_dir: str | Path, source_path: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    torch.save(bundle.train_data, out_path / "train.pt")
    torch.save(bundle.val_data, out_path / "val.pt")

    meta = {
        "source_path": str(source_path),
        "vocab_size": bundle.vocab_size,
        "split_ratio": bundle.split_ratio,
        "stoi": bundle.stoi,
        "itos": [bundle.itos[index] for index in range(bundle.vocab_size)],
    }
    (out_path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_prepared_data(prepared_dir: str | Path) -> DatasetBundle:
    prepared_path = Path(prepared_dir)
    meta = json.loads((prepared_path / "meta.json").read_text(encoding="utf-8"))

    train_data = torch.load(prepared_path / "train.pt", map_location="cpu")
    val_data = torch.load(prepared_path / "val.pt", map_location="cpu")
    stoi = {str(key): int(value) for key, value in meta["stoi"].items()}
    itos = {index: char for index, char in enumerate(meta["itos"])}

    return DatasetBundle(
        train_data=train_data,
        val_data=val_data,
        stoi=stoi,
        itos=itos,
        vocab_size=int(meta["vocab_size"]),
        split_ratio=float(meta["split_ratio"]),
    )


def decode_tokens(token_ids: list[int], itos: dict[int, str]) -> str:
    return "".join(itos[int(token_id)] for token_id in token_ids)