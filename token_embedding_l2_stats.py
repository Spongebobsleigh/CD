import argparse
import math

import torch
from transformers import AutoModel, AutoModelForCausalLM


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compute mean/min/max pairwise L2 distances of token embeddings."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name or local path (HF format).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Block size for pairwise distance computation.",
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        default=None,
        help="Limit vocab size for a quicker approximate run.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Embedding dtype used for distance calculation.",
    )
    return parser.parse_args()


def _get_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main():
    args = _parse_args()
    dtype = _get_dtype(args.dtype)

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model)
    except ValueError:
        # LLaVA uses LlavaConfig, which is not supported by AutoModelForCausalLM.
        model = AutoModel.from_pretrained(args.model)
    emb = model.get_input_embeddings().weight.detach()

    if args.max_vocab is not None:
        emb = emb[: args.max_vocab]

    emb = emb.to(device=args.device, dtype=dtype)
    vocab_size = emb.size(0)
    block = max(1, args.block_size)

    total_sum = 0.0
    total_count = 0
    min_dist = math.inf
    max_dist = -math.inf

    with torch.no_grad():
        for i in range(0, vocab_size, block):
            emb_i = emb[i : i + block]
            for j in range(i, vocab_size, block):
                emb_j = emb[j : j + block]
                dists = torch.cdist(emb_i, emb_j, p=2)

                if i == j:
                    # Exclude self-distances on the diagonal.
                    diag_len = min(dists.size(0), dists.size(1))
                    if diag_len > 0:
                        diag_idx = torch.arange(diag_len, device=dists.device)
                        dists[diag_idx, diag_idx] = float("nan")

                valid = dists[~torch.isnan(dists)]
                if valid.numel() == 0:
                    continue

                total_sum += valid.sum().item()
                total_count += valid.numel()
                min_dist = min(min_dist, valid.min().item())
                max_dist = max(max_dist, valid.max().item())

    mean_dist = total_sum / total_count if total_count > 0 else float("nan")
    print(f"vocab_size: {vocab_size}")
    print(f"pair_count: {total_count}")
    print(f"mean_l2: {mean_dist}")
    print(f"min_l2: {min_dist}")
    print(f"max_l2: {max_dist}")


if __name__ == "__main__":
    main()
