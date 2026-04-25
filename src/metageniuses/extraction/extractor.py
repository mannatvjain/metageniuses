from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from .config import ExtractionConfig
from .input_io import iter_sequence_records
from .model_adapter import ModelAdapter, TransformersModelAdapter
from .preprocess import preprocess_record
from .schemas import ExtractionStats, ModelDescription, RunManifest, SCHEMA_VERSION, utc_now_iso
from .storage import ActivationStore


def _batch_iter(values: Iterable, batch_size: int):
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class ResidualExtractionPipeline:
    """Extract token-level hidden states for selected transformer layers."""

    def run(
        self,
        cfg: ExtractionConfig,
        adapter: ModelAdapter | None = None,
    ) -> Path:
        cfg.validate()
        model_adapter = adapter or TransformersModelAdapter(cfg.model)

        desc = model_adapter.describe()
        selected_layers = cfg.layer_selection.resolve(desc.num_transformer_layers)

        run_id = cfg.runtime.run_id or self._make_run_id()
        artifact_root = Path(cfg.runtime.output_root) / run_id
        store = ActivationStore(
            artifact_root=artifact_root,
            selected_layers=selected_layers,
            d_model=desc.d_model,
            max_rows_per_shard=cfg.runtime.max_rows_per_shard,
        )

        stats = ExtractionStats()
        records = iter_sequence_records(cfg.input)
        for batch_records in _batch_iter(records, cfg.runtime.batch_size):
            processed_records = []
            preprocess_metadata = []

            for record in batch_records:
                if cfg.runtime.max_reads is not None and stats.total_sequences_seen >= cfg.runtime.max_reads:
                    break

                stats = ExtractionStats(
                    total_sequences_seen=stats.total_sequences_seen + 1,
                    total_sequences_kept=stats.total_sequences_kept,
                    total_sequences_skipped=stats.total_sequences_skipped,
                    total_tokens=stats.total_tokens,
                    total_rows_written=stats.total_rows_written,
                )
                prep = preprocess_record(record, cfg.preprocess)
                if prep.record is None:
                    stats = ExtractionStats(
                        total_sequences_seen=stats.total_sequences_seen,
                        total_sequences_kept=stats.total_sequences_kept,
                        total_sequences_skipped=stats.total_sequences_skipped + 1,
                        total_tokens=stats.total_tokens,
                        total_rows_written=stats.total_rows_written,
                    )
                    continue

                processed_records.append(prep.record)
                preprocess_metadata.append(
                    {
                        "invalid_char_count": prep.invalid_char_count,
                        "invalid_fraction": prep.invalid_fraction,
                    }
                )

            if not processed_records:
                if cfg.runtime.max_reads is not None and stats.total_sequences_seen >= cfg.runtime.max_reads:
                    break
                continue

            sequences = [record.sequence for record in processed_records]
            batch = model_adapter.extract_batch(
                sequences=sequences,
                transformer_layers=selected_layers,
                max_length=cfg.preprocess.max_length,
            )

            for seq_idx, record in enumerate(processed_records):
                token_ids = batch.token_ids[seq_idx]
                store.append_sequence(
                    {
                        "sequence_id": record.sequence_id,
                        "sequence": record.sequence,
                        "sequence_length": len(record.sequence),
                        "token_count": len(token_ids),
                        "metadata": record.metadata,
                        "preprocess": preprocess_metadata[seq_idx],
                    }
                )
                stats = ExtractionStats(
                    total_sequences_seen=stats.total_sequences_seen,
                    total_sequences_kept=stats.total_sequences_kept + 1,
                    total_sequences_skipped=stats.total_sequences_skipped,
                    total_tokens=stats.total_tokens + len(token_ids),
                    total_rows_written=stats.total_rows_written,
                )
                for token_idx, token_id in enumerate(token_ids):
                    for layer in selected_layers:
                        vector = batch.hidden_states_by_layer[layer][seq_idx][token_idx]
                        store.append_activation(
                            layer=layer,
                            vector=vector,
                            row={
                                "sequence_id": record.sequence_id,
                                "token_index": token_idx,
                                "token_id": token_id,
                                "layer": layer,
                            },
                        )
                        stats = ExtractionStats(
                            total_sequences_seen=stats.total_sequences_seen,
                            total_sequences_kept=stats.total_sequences_kept,
                            total_sequences_skipped=stats.total_sequences_skipped,
                            total_tokens=stats.total_tokens,
                            total_rows_written=stats.total_rows_written + 1,
                        )

            if cfg.runtime.max_reads is not None and stats.total_sequences_seen >= cfg.runtime.max_reads:
                break

        layers_payload = store.finalize()
        self._write_manifest(
            cfg=cfg,
            desc=desc,
            run_id=run_id,
            selected_layers=selected_layers,
            stats=stats,
            layers_payload=layers_payload,
            artifact_root=artifact_root,
        )
        return artifact_root

    def _make_run_id(self) -> str:
        return f"extract_{utc_now_iso().replace(':', '-').replace('+00:00', 'Z')}_{uuid4().hex[:8]}"

    def _write_manifest(
        self,
        cfg: ExtractionConfig,
        desc: ModelDescription,
        run_id: str,
        selected_layers: list[int],
        stats: ExtractionStats,
        layers_payload: dict[str, list[dict]],
        artifact_root: Path,
    ) -> None:
        manifest = RunManifest(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            created_at=utc_now_iso(),
            input_path=cfg.input.path,
            input_format=cfg.input.format,
            model={
                "model_id": desc.model_id,
                "tokenizer_id": desc.tokenizer_id,
                "revision": desc.revision,
                "num_transformer_layers": desc.num_transformer_layers,
                "d_model": desc.d_model,
            },
            layer_selection={
                "selected_transformer_layers": selected_layers,
            },
            preprocess=asdict(cfg.preprocess),
            runtime=asdict(cfg.runtime),
            stats=stats.to_dict(),
            layers=layers_payload,
        )
        path = artifact_root / "manifest.json"
        path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))

