from __future__ import annotations

import json
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True)
class ShardDescriptor:
    shard_id: int
    rows: int
    data_file: str
    index_file: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "rows": self.rows,
            "data_file": self.data_file,
            "index_file": self.index_file,
        }


class _LayerWriter:
    def __init__(self, root: Path, layer: int, d_model: int, max_rows_per_shard: int) -> None:
        self._root = root
        self._layer = layer
        self._d_model = d_model
        self._max_rows_per_shard = max_rows_per_shard
        self._layer_dir = self._root / f"layer_{layer:02d}"
        self._layer_dir.mkdir(parents=True, exist_ok=True)

        self._current_shard_id = -1
        self._rows_in_shard = 0
        self._total_rows = 0
        self._data_fp: Any | None = None
        self._index_fp: TextIO | None = None
        self._shards: list[ShardDescriptor] = []
        self._open_next_shard()

    def _close_current_shard(self) -> None:
        if self._data_fp is None or self._index_fp is None or self._current_shard_id < 0:
            return
        self._data_fp.close()
        self._index_fp.close()
        self._shards.append(
            ShardDescriptor(
                shard_id=self._current_shard_id,
                rows=self._rows_in_shard,
                data_file=f"layer_{self._layer:02d}/shard_{self._current_shard_id:05d}.f32",
                index_file=f"layer_{self._layer:02d}/shard_{self._current_shard_id:05d}.jsonl",
            )
        )
        self._data_fp = None
        self._index_fp = None

    def _open_next_shard(self) -> None:
        self._close_current_shard()
        self._current_shard_id += 1
        self._rows_in_shard = 0
        data_path = self._layer_dir / f"shard_{self._current_shard_id:05d}.f32"
        index_path = self._layer_dir / f"shard_{self._current_shard_id:05d}.jsonl"
        self._data_fp = data_path.open("wb")
        self._index_fp = index_path.open("w", encoding="utf-8")

    def append(self, vector: list[float], metadata: dict[str, Any]) -> None:
        if len(vector) != self._d_model:
            raise ValueError(
                f"Layer {self._layer} expected vector length {self._d_model}, got {len(vector)}"
            )
        assert self._data_fp is not None and self._index_fp is not None
        array("f", vector).tofile(self._data_fp)
        record = dict(metadata)
        record["row_in_shard"] = self._rows_in_shard
        record["row_global"] = self._total_rows
        self._index_fp.write(json.dumps(record, sort_keys=True) + "\n")
        self._rows_in_shard += 1
        self._total_rows += 1

        if self._rows_in_shard >= self._max_rows_per_shard:
            self._open_next_shard()

    def finalize(self) -> dict[str, Any]:
        self._close_current_shard()
        return {
            "rows": self._total_rows,
            "shards": [descriptor.to_dict() for descriptor in self._shards if descriptor.rows > 0],
        }


class ActivationStore:
    def __init__(
        self,
        artifact_root: Path,
        selected_layers: list[int],
        d_model: int,
        max_rows_per_shard: int,
    ) -> None:
        self.artifact_root = artifact_root
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._d_model = d_model
        self._writers = {
            layer: _LayerWriter(
                root=self.artifact_root / "activations",
                layer=layer,
                d_model=d_model,
                max_rows_per_shard=max_rows_per_shard,
            )
            for layer in selected_layers
        }
        self._sequences_fp = (self.artifact_root / "sequences.jsonl").open("w", encoding="utf-8")

    def append_sequence(self, row: dict[str, Any]) -> None:
        self._sequences_fp.write(json.dumps(row, sort_keys=True) + "\n")

    def append_activation(self, layer: int, vector: list[float], row: dict[str, Any]) -> None:
        self._writers[layer].append(vector=vector, metadata=row)

    def finalize(self) -> dict[str, Any]:
        self._sequences_fp.close()
        layers_payload: dict[str, Any] = {}
        for layer, writer in self._writers.items():
            layers_payload[str(layer)] = writer.finalize()
        return layers_payload

