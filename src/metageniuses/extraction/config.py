from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InputConfig:
    path: str
    format: str = "jsonl"
    sequence_key: str = "sequence"
    id_key: str = "sequence_id"
    metadata_keys: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.format not in {"jsonl", "fasta"}:
            raise ValueError(f"Unsupported input format: {self.format}")
        if not self.path:
            raise ValueError("Input path is required.")


@dataclass(frozen=True)
class PreprocessConfig:
    uppercase: bool = True
    allowed_chars: str = "ACGTUN"
    replace_invalid_with: str = "N"
    max_invalid_fraction: float = 0.05
    min_length: int = 1
    max_length: int = 512
    strip_whitespace: bool = True

    def validate(self) -> None:
        if self.min_length < 1:
            raise ValueError("min_length must be >= 1")
        if self.max_length < self.min_length:
            raise ValueError("max_length must be >= min_length")
        if self.replace_invalid_with not in self.allowed_chars:
            raise ValueError("replace_invalid_with must be in allowed_chars")
        if not (0.0 <= self.max_invalid_fraction <= 1.0):
            raise ValueError("max_invalid_fraction must be in [0, 1]")


@dataclass(frozen=True)
class LayerSelectionConfig:
    layers: list[int] | None = None
    last_n_layers: int | None = None

    def validate(self) -> None:
        if self.layers and self.last_n_layers:
            raise ValueError("Set either layers or last_n_layers, not both.")
        if self.layers is None and self.last_n_layers is None:
            raise ValueError("Set layers or last_n_layers.")
        if self.layers is not None:
            if not self.layers:
                raise ValueError("layers cannot be empty.")
            for layer in self.layers:
                if layer < 1:
                    raise ValueError("layers must use transformer-layer indices starting at 1.")
        if self.last_n_layers is not None and self.last_n_layers < 1:
            raise ValueError("last_n_layers must be >= 1")

    def resolve(self, num_transformer_layers: int) -> list[int]:
        self.validate()
        if self.layers is not None:
            resolved = sorted(set(self.layers))
        else:
            assert self.last_n_layers is not None
            start = max(1, num_transformer_layers - self.last_n_layers + 1)
            resolved = list(range(start, num_transformer_layers + 1))

        for layer in resolved:
            if layer > num_transformer_layers:
                raise ValueError(
                    f"Requested layer {layer}, but model only has {num_transformer_layers} layers."
                )
        return resolved


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    tokenizer_id: str | None = None
    revision: str | None = None
    local_files_only: bool = False
    trust_remote_code: bool = False
    device: str = "auto"
    dtype: str = "auto"

    def validate(self) -> None:
        if not self.model_id:
            raise ValueError("model_id is required.")


@dataclass(frozen=True)
class RuntimeConfig:
    output_root: str = "results/extraction"
    run_id: str | None = None
    batch_size: int = 4
    max_rows_per_shard: int = 100000
    max_reads: int | None = None
    resume: bool = False

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_rows_per_shard < 1:
            raise ValueError("max_rows_per_shard must be >= 1")
        if self.max_reads is not None and self.max_reads < 1:
            raise ValueError("max_reads must be >= 1 when set")


@dataclass(frozen=True)
class ExtractionConfig:
    input: InputConfig
    preprocess: PreprocessConfig
    model: ModelConfig
    layer_selection: LayerSelectionConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExtractionConfig":
        return cls(
            input=InputConfig(**payload["input"]),
            preprocess=PreprocessConfig(**payload.get("preprocess", {})),
            model=ModelConfig(**payload["model"]),
            layer_selection=LayerSelectionConfig(**payload["layer_selection"]),
            runtime=RuntimeConfig(**payload.get("runtime", {})),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ExtractionConfig":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    def validate(self) -> None:
        self.input.validate()
        self.preprocess.validate()
        self.model.validate()
        self.layer_selection.validate()
        self.runtime.validate()
