"""Compute final reconstruction MSE on a held-out activation set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from metageniuses.extraction.contracts import iter_layer_batches
from .model import BatchTopKSAE


def compute_test_mse(
    artifact_root: str,
    sae_checkpoint: str,
    layer: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 4096,
) -> float:
    scale_path = Path(sae_checkpoint).parent / "activation_scale.json"
    activation_scale = 1.0
    if scale_path.exists():
        activation_scale = json.loads(scale_path.read_text())["activation_scale"]

    sae = BatchTopKSAE.load(sae_checkpoint, device=device)
    sae = sae.to(device)
    sae.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch_vecs, _ in iter_layer_batches(artifact_root, layer, batch_size=batch_size):
        x = torch.tensor(batch_vecs, dtype=torch.float32, device=device)
        x = x / activation_scale
        with torch.no_grad():
            out = sae(x)
        total_loss += F.mse_loss(out.x_hat, x, reduction="sum").item()
        total_tokens += x.shape[0]

    mse = total_loss / total_tokens
    print(f"Test MSE: {mse:.5f}  ({total_tokens:,} tokens)")
    return mse


def main() -> None:
    p = argparse.ArgumentParser(description="Compute test reconstruction MSE")
    p.add_argument("--artifact_root", required=True)
    p.add_argument("--sae_checkpoint", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    compute_test_mse(
        artifact_root=args.artifact_root,
        sae_checkpoint=args.sae_checkpoint,
        layer=args.layer,
        device=args.device,
    )


if __name__ == "__main__":
    main()
