from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


SCHEMA_VERSION = "1.0.0"


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    sequence: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelDescription:
    model_id: str
    tokenizer_id: str
    revision: str | None
    num_transformer_layers: int
    d_model: int


@dataclass(frozen=True)
class ExtractionStats:
    total_sequences_seen: int = 0
    total_sequences_kept: int = 0
    total_sequences_skipped: int = 0
    total_tokens: int = 0
    total_rows_written: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class RunManifest:
    schema_version: str
    run_id: str
    created_at: str
    input_path: str
    input_format: str
    model: dict[str, Any]
    layer_selection: dict[str, Any]
    preprocess: dict[str, Any]
    runtime: dict[str, Any]
    stats: dict[str, Any]
    layers: dict[str, list[dict[str, Any]]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

