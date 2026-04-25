from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .config import InputConfig
from .schemas import SequenceRecord


def _iter_jsonl(cfg: InputConfig) -> Iterator[SequenceRecord]:
    path = Path(cfg.path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc

            sequence = payload.get(cfg.sequence_key)
            sequence_id = payload.get(cfg.id_key)
            if sequence is None or sequence_id is None:
                raise ValueError(
                    f"Line {line_no}: missing required keys "
                    f"{cfg.id_key!r} and/or {cfg.sequence_key!r}"
                )

            metadata = {}
            if cfg.metadata_keys:
                for key in cfg.metadata_keys:
                    if key in payload:
                        metadata[key] = payload[key]
            else:
                for key, value in payload.items():
                    if key not in {cfg.id_key, cfg.sequence_key}:
                        metadata[key] = value

            yield SequenceRecord(
                sequence_id=str(sequence_id),
                sequence=str(sequence),
                metadata=metadata,
            )


def _iter_fasta(cfg: InputConfig) -> Iterator[SequenceRecord]:
    path = Path(cfg.path)
    current_id = None
    seq_parts: list[str] = []

    def flush() -> SequenceRecord | None:
        nonlocal current_id, seq_parts
        if current_id is None:
            return None
        record = SequenceRecord(sequence_id=current_id, sequence="".join(seq_parts), metadata={})
        current_id = None
        seq_parts = []
        return record

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                prior = flush()
                if prior is not None:
                    yield prior
                current_id = stripped[1:].split()[0]
            else:
                seq_parts.append(stripped)
        final = flush()
        if final is not None:
            yield final


def iter_sequence_records(cfg: InputConfig) -> Iterator[SequenceRecord]:
    if cfg.format == "jsonl":
        yield from _iter_jsonl(cfg)
        return
    if cfg.format == "fasta":
        yield from _iter_fasta(cfg)
        return
    raise ValueError(f"Unsupported input format: {cfg.format}")

