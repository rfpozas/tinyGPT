from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinygpt import generate_text, load_checkpoint, load_prepared_data, resolve_device

# This script loads a trained tinyGPT model checkpoint and generates text
# based on a given prompt.

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from a trained tinyGPT checkpoint")
    parser.add_argument("--prepared-dir", default="data/processed", help="Prepared dataset directory")
    parser.add_argument("--checkpoint-path", default="artifacts/tinygpt.pt", help="Model checkpoint path")
    parser.add_argument("--prompt", default="ROMEO:", help="Prompt to seed generation")
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    dataset = load_prepared_data(args.prepared_dir)
    model, model_config, _ = load_checkpoint(args.checkpoint_path, device=device)

    sample = generate_text(
        model=model,
        start_text=args.prompt,
        stoi=dataset.stoi,
        itos=dataset.itos,
        context_length=model_config.context_length,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(sample)


if __name__ == "__main__":
    main()