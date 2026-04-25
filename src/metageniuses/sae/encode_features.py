"""Encode raw residual activations into SAE feature vectors for probing.

Produces two files in the output directory:
  features.npy      float32 array [n_sequences, d_sae] — mean-pooled SAE features
  sequence_ids.json list of sequence_ids in row order
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from metageniuses.extraction.contracts import iter_layer_batches, load_manifest
from .model import BatchTopKSAE


def encode_features(
    artifact_root: str,
    sae_checkpoint: str,
    layer: int,
    output_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 4096,
) -> None:
    artifact_root = Path(artifact_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load activation scale used during training
    scale_path = Path(sae_checkpoint).parent / "activation_scale.json"
    activation_scale = 1.0
    if scale_path.exists():
        activation_scale = json.loads(scale_path.read_text())["activation_scale"]
    print(f"activation_scale={activation_scale:.4f}")

    sae = BatchTopKSAE.load(sae_checkpoint, device=device)
    sae = sae.to(device)
    sae.eval()
    print(f"SAE loaded: d_model={sae.d_model}  d_sae={sae.d_sae}  k={sae.k}")

    # Accumulate per-sequence feature sums and token counts for mean pooling
    seq_feature_sum: dict[str, np.ndarray] = {}
    seq_token_count: dict[str, int] = defaultdict(int)

    n_tokens = 0
    for batch_vecs, batch_meta in iter_layer_batches(artifact_root, layer, batch_size=batch_size):
        x = torch.tensor(batch_vecs, dtype=torch.float32, device=device)
        x = x / activation_scale

        with torch.no_grad():
            z = sae.encode(x)
            # Apply per-token TopK (not batch TopK) for inference so each token
            # gets exactly k active features independently.
            topk = torch.topk(z, sae.k, dim=-1)
            z_sparse = torch.zeros_like(z)
            z_sparse.scatter_(1, topk.indices, topk.values)

        z_np = z_sparse.cpu().numpy()

        for i, meta in enumerate(batch_meta):
            sid = meta["sequence_id"]
            if sid not in seq_feature_sum:
                seq_feature_sum[sid] = np.zeros(sae.d_sae, dtype=np.float32)
            seq_feature_sum[sid] += z_np[i]
            seq_token_count[sid] += 1

        n_tokens += len(batch_vecs)
        if n_tokens % 50000 == 0:
            print(f"  {n_tokens:,} tokens processed")

    print(f"Total tokens: {n_tokens:,}  Unique sequences: {len(seq_feature_sum):,}")

    # Build output matrix in a stable sequence order
    sequence_ids = sorted(seq_feature_sum.keys())
    features = np.stack(
        [seq_feature_sum[sid] / seq_token_count[sid] for sid in sequence_ids],
        axis=0,
    )  # [n_sequences, d_sae]

    np.save(output_dir / "features.npy", features)
    (output_dir / "sequence_ids.json").write_text(json.dumps(sequence_ids, indent=2))

    print(f"Saved features.npy {features.shape}  →  {output_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Encode activations → SAE feature vectors")
    p.add_argument("--artifact_root", required=True)
    p.add_argument("--sae_checkpoint", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=4096)
    args = p.parse_args()
    encode_features(
        artifact_root=args.artifact_root,
        sae_checkpoint=args.sae_checkpoint,
        layer=args.layer,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
