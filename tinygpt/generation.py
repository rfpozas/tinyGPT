from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import decode_tokens
from .model import TinyGPT


@torch.no_grad()
def generate_text(
    model: TinyGPT,
    start_text: str,
    stoi: dict[str, int],
    itos: dict[int, str],
    context_length: int,
    device: str,
    max_new_tokens: int = 300,
    temperature: float = 1.0,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    model.eval()
    prompt = start_text or " "
    start_ids = [stoi.get(char, 0) for char in prompt]
    idx = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return decode_tokens(idx[0].tolist(), itos)