from __future__ import annotations

import json
import os
import re
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
    SHARD_RE = re.compile(r"^shard_(\d{5})\.(f32|jsonl)$")

    def __init__(
        self,
        root: Path,
        layer: int,
        d_model: int,
        max_rows_per_shard: int,
        resume: bool = False,
    ) -> None:
        self._root = root
        self._layer = layer
        self._d_model = d_model
        self._max_rows_per_shard = max_rows_per_shard
        self._layer_dir = self._root / f"layer_{layer:02d}"
        self._layer_dir.mkdir(parents=True, exist_ok=True)

        self._existing_shards: list[ShardDescriptor] = []
        self._current_shard_id = -1
        self._rows_in_shard = 0
        self._total_rows = 0
        self._data_fp: Any | None = None
        self._index_fp: TextIO | None = None
        self._shards: list[ShardDescriptor] = []
        if resume:
            self._existing_shards = self._scan_existing_shards()
            if self._existing_shards:
                self._current_shard_id = max(shard.shard_id for shard in self._existing_shards)
                self._total_rows = sum(shard.rows for shard in self._existing_shards)
        self._open_next_shard()

    @property
    def existing_rows(self) -> int:
        return sum(shard.rows for shard in self._existing_shards)

    def _count_valid_json_lines(self, path: Path) -> tuple[int, int]:
        valid_lines = 0
        valid_offset = 0
        with path.open("rb") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                next_offset = handle.tell()
                try:
                    json.loads(line.decode("utf-8"))
                except Exception:
                    break
                valid_lines += 1
                valid_offset = next_offset
        return valid_lines, valid_offset

    def _truncate_json_lines(self, path: Path, keep_lines: int) -> None:
        line_count = 0
        keep_offset = 0
        with path.open("rb") as handle:
            while line_count < keep_lines:
                line = handle.readline()
                if not line:
                    break
                keep_offset = handle.tell()
                line_count += 1
        with path.open("r+b") as handle:
            handle.truncate(keep_offset)

    def _scan_existing_shards(self) -> list[ShardDescriptor]:
        shard_ids: set[int] = set()
        for path in self._layer_dir.glob("shard_*.*"):
            match = self.SHARD_RE.match(path.name)
            if match is None:
                continue
            shard_ids.add(int(match.group(1)))

        descriptors: list[ShardDescriptor] = []
        row_size_bytes = self._d_model * 4
        for shard_id in sorted(shard_ids):
            data_path = self._layer_dir / f"shard_{shard_id:05d}.f32"
            index_path = self._layer_dir / f"shard_{shard_id:05d}.jsonl"
            if not data_path.exists() or not index_path.exists():
                continue

            data_rows = data_path.stat().st_size // row_size_bytes
            json_rows, json_valid_offset = self._count_valid_json_lines(index_path)
            valid_rows = int(min(data_rows, json_rows))
            if valid_rows <= 0:
                continue

            expected_data_size = valid_rows * row_size_bytes
            if data_path.stat().st_size != expected_data_size:
                with data_path.open("r+b") as data_fp:
                    data_fp.truncate(expected_data_size)

            if json_rows != valid_rows:
                self._truncate_json_lines(index_path, keep_lines=valid_rows)
            elif json_valid_offset != index_path.stat().st_size:
                with index_path.open("r+b") as index_fp:
                    index_fp.truncate(json_valid_offset)

            descriptors.append(
                ShardDescriptor(
                    shard_id=shard_id,
                    rows=valid_rows,
                    data_file=f"layer_{self._layer:02d}/shard_{shard_id:05d}.f32",
                    index_file=f"layer_{self._layer:02d}/shard_{shard_id:05d}.jsonl",
                )
            )
        return descriptors

    def _close_current_shard(self) -> None:
        if self._data_fp is None or self._index_fp is None or self._current_shard_id < 0:
            return
        self._data_fp.flush()
        os.fsync(self._data_fp.fileno())
        self._index_fp.flush()
        os.fsync(self._index_fp.fileno())
        self._data_fp.close()
        self._index_fp.close()
        if self._rows_in_shard > 0:
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

    def flush(self) -> None:
        if self._data_fp is not None:
            self._data_fp.flush()
            os.fsync(self._data_fp.fileno())
        if self._index_fp is not None:
            self._index_fp.flush()
            os.fsync(self._index_fp.fileno())

    def finalize(self) -> dict[str, Any]:
        self._close_current_shard()
        return {
            "rows": self._total_rows,
            "shards": [
                descriptor.to_dict()
                for descriptor in (self._existing_shards + self._shards)
                if descriptor.rows > 0
            ],
        }


class ActivationStore:
    def __init__(
        self,
        artifact_root: Path,
        selected_layers: list[int],
        d_model: int,
        max_rows_per_shard: int,
        resume: bool = False,
    ) -> None:
        self.artifact_root = artifact_root
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._d_model = d_model
        self._sequences_path = self.artifact_root / "sequences.jsonl"
        self.completed_sequence_ids: set[str] = set()
        self.existing_sequences_kept = 0
        self.existing_tokens = 0
        if resume and self._sequences_path.exists():
            self.existing_sequences_kept, self.existing_tokens, self.completed_sequence_ids = (
                self._load_existing_sequences()
            )

        self._writers = {
            layer: _LayerWriter(
                root=self.artifact_root / "activations",
                layer=layer,
                d_model=d_model,
                max_rows_per_shard=max_rows_per_shard,
                resume=resume,
            )
            for layer in selected_layers
        }
        self.existing_rows_written = sum(writer.existing_rows for writer in self._writers.values())
        mode = "a" if resume and self._sequences_path.exists() else "w"
        self._sequences_fp = self._sequences_path.open(mode, encoding="utf-8")

    def _load_existing_sequences(self) -> tuple[int, int, set[str]]:
        valid_count = 0
        total_tokens = 0
        ids: set[str] = set()
        valid_offset = 0
        with self._sequences_path.open("rb") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                next_offset = handle.tell()
                try:
                    row = json.loads(line.decode("utf-8"))
                except Exception:
                    break
                valid_count += 1
                total_tokens += int(row.get("token_count", 0))
                seq_id = row.get("sequence_id")
                if seq_id is not None:
                    ids.add(str(seq_id))
                valid_offset = next_offset
        if self._sequences_path.stat().st_size != valid_offset:
            with self._sequences_path.open("r+b") as handle:
                handle.truncate(valid_offset)
        return valid_count, total_tokens, ids

    def append_sequence(self, row: dict[str, Any]) -> None:
        self._sequences_fp.write(json.dumps(row, sort_keys=True) + "\n")
        seq_id = row.get("sequence_id")
        if seq_id is not None:
            self.completed_sequence_ids.add(str(seq_id))

    def append_activation(self, layer: int, vector: list[float], row: dict[str, Any]) -> None:
        self._writers[layer].append(vector=vector, metadata=row)

    def flush(self) -> None:
        self._sequences_fp.flush()
        os.fsync(self._sequences_fp.fileno())
        for writer in self._writers.values():
            writer.flush()

    def finalize(self) -> dict[str, Any]:
        self._sequences_fp.close()
        layers_payload: dict[str, Any] = {}
        for layer, writer in self._writers.items():
            layers_payload[str(layer)] = writer.finalize()
        return layers_payload
