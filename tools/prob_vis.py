import json
import os
from datetime import datetime

import torch

VIS_PROB_DIR = os.path.join(os.getcwd(), "vis", "prob")
_CURRENT_LOG_PATH = None
_SAMPLE_INDEX = 0
_LAST_STEP = None


def _ensure_log_path():
    global _CURRENT_LOG_PATH
    if _CURRENT_LOG_PATH is None:
        os.makedirs(VIS_PROB_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _CURRENT_LOG_PATH = os.path.join(VIS_PROB_DIR, f"prob_{timestamp}.jsonl")
        with open(_CURRENT_LOG_PATH, "w", encoding="utf-8") as f:
            f.write("")
    return _CURRENT_LOG_PATH


def _append_record(record):
    path = _ensure_log_path()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_token_distribution(
    logits,
    tokenizer,
    topk: int = 5,
    prefix: str = "[CD] Token distribution",
    step: int = None,
    meta: dict = None,
):
    """
    Print the top-k token probabilities for the provided logits tensor.

    Args:
        logits: Tensor of shape [1, vocab_size] (after logits processors).
        tokenizer: Hugging Face tokenizer used to decode token ids.
        topk: Number of tokens to display.
        prefix: Message prefix for the stdout log.
    """
    global _SAMPLE_INDEX, _LAST_STEP
    if logits is None or tokenizer is None:
        return

    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = probs.topk(topk, dim=-1)
    top_probs = top_probs[0].detach().cpu().tolist()
    top_ids = top_ids[0].detach().cpu().tolist()

    tokens = [
        tokenizer.decode([token_id], skip_special_tokens=True).strip() or "<blank>"
        for token_id in top_ids
    ]
    formatted = ", ".join(f"{tok}:{prob:.4f}" for tok, prob in zip(tokens, top_probs))
    # print(f"{prefix}: {formatted}")

    if step == 0 and (_LAST_STEP is None or _LAST_STEP != 0):
        _SAMPLE_INDEX += 1
        _append_record({"sample_index": _SAMPLE_INDEX, "event": "new_sample"})

    record = {
        "step": step,
        "tokens": tokens,
        "probs": top_probs,
        "prefix": prefix,
    }
    if _SAMPLE_INDEX > 0:
        record["sample_index"] = _SAMPLE_INDEX
    if meta:
        record["meta"] = meta
    _append_record(record)
    if step is not None:
        _LAST_STEP = step
