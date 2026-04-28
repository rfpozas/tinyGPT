from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinygpt import load_text_corpus, prepare_dataset, save_prepared_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a character-level corpus for tinyGPT")
    parser.add_argument(
        "--input",
        default="data/tiny_shakespeare.txt",
        help="Path to the raw text corpus",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="Directory where train.pt, val.pt, and meta.json will be saved",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.9,
        help="Fraction of the corpus used for training",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    text = load_text_corpus(args.input)
    dataset = prepare_dataset(text, split_ratio=args.split_ratio)
    save_prepared_data(dataset, args.out_dir, args.input)

    print(f"Saved prepared dataset to {Path(args.out_dir).resolve()}")
    print(f"Corpus size: {len(text)} characters")
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Train tokens: {len(dataset.train_data)}")
    print(f"Val tokens: {len(dataset.val_data)}")


if __name__ == "__main__":
    main()