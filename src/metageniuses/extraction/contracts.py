from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterator


def load_manifest(artifact_root: str | Path) -> dict:
    path = Path(artifact_root) / "manifest.json"
    return json.loads(path.read_text())


def iter_layer_batches(
    artifact_root: str | Path,
    transformer_layer: int,
    batch_size: int = 512,
) -> Iterator[tuple[list[list[float]], list[dict]]]:
    root = Path(artifact_root)
    manifest = load_manifest(root)
    layer_key = str(transformer_layer)
    if layer_key not in manifest["layers"]:
        raise ValueError(f"Layer {transformer_layer} not found in manifest.")

    d_model = int(manifest["model"]["d_model"])
    row_size_bytes = d_model * 4
    row_unpack = struct.Struct("<" + ("f" * d_model))

    pending_vectors: list[list[float]] = []
    pending_meta: list[dict] = []

    shards = manifest["layers"][layer_key]["shards"]
    for shard in shards:
        data_path = root / "activations" / shard["data_file"]
        index_path = root / "activations" / shard["index_file"]
        with data_path.open("rb") as data_fp, index_path.open("r", encoding="utf-8") as index_fp:
            for line in index_fp:
                row_bytes = data_fp.read(row_size_bytes)
                if len(row_bytes) != row_size_bytes:
                    raise RuntimeError(
                        f"Activation data truncated for layer {transformer_layer} in {data_path}"
                    )
                vector = list(row_unpack.unpack(row_bytes))
                pending_vectors.append(vector)
                pending_meta.append(json.loads(line))
                if len(pending_vectors) >= batch_size:
                    yield pending_vectors, pending_meta
                    pending_vectors, pending_meta = [], []

    if pending_vectors:
        yield pending_vectors, pending_meta

