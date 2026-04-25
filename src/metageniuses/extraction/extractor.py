from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .config import ExtractionConfig
from .input_io import iter_sequence_records
from .model_adapter import ModelAdapter, TransformersModelAdapter
from .preprocess import preprocess_record
from .schemas import ExtractionStats, ModelDescription, RunManifest, SCHEMA_VERSION, utc_now_iso
from .storage import ActivationStore


class ResidualExtractionPipeline:
    """Extract token-level hidden states for selected transformer layers."""

    def run(
        self,
        cfg: ExtractionConfig,
        adapter: ModelAdapter | None = None,
    ) -> Path:
        cfg.validate()
        self._validate_output_root(cfg)

        run_id = cfg.runtime.run_id or self._make_run_id()
        artifact_root = Path(cfg.runtime.output_root) / run_id
        self._validate_run_directory(artifact_root=artifact_root, resume=cfg.runtime.resume)
        prior_progress = self._read_progress(artifact_root=artifact_root) if cfg.runtime.resume else {}

        model_adapter = adapter or TransformersModelAdapter(cfg.model)
        desc = model_adapter.describe()
        selected_layers = cfg.layer_selection.resolve(desc.num_transformer_layers)

        store = ActivationStore(
            artifact_root=artifact_root,
            selected_layers=selected_layers,
            d_model=desc.d_model,
            max_rows_per_shard=cfg.runtime.max_rows_per_shard,
            resume=cfg.runtime.resume,
        )

        stats = ExtractionStats(
            total_sequences_seen=max(
                int(prior_progress.get("total_sequences_seen", 0)),
                store.existing_sequences_kept,
            ),
            total_sequences_kept=store.existing_sequences_kept,
            total_sequences_skipped=int(prior_progress.get("total_sequences_skipped", 0)),
            total_tokens=store.existing_tokens,
            total_rows_written=store.existing_rows_written,
        )

        records = iter_sequence_records(cfg.input)
        skip_count = stats.total_sequences_seen
        pending_records = []

        for record in records:
            if skip_count > 0:
                skip_count -= 1
                continue
            pending_records.append(record)
            if len(pending_records) < cfg.runtime.batch_size:
                continue

            stats = self._process_batch(
                batch_records=pending_records,
                cfg=cfg,
                stats=stats,
                store=store,
                model_adapter=model_adapter,
                selected_layers=selected_layers,
                artifact_root=artifact_root,
            )
            pending_records = []
            if cfg.runtime.max_reads is not None and stats.total_sequences_seen >= cfg.runtime.max_reads:
                break

        if pending_records and (
            cfg.runtime.max_reads is None or stats.total_sequences_seen < cfg.runtime.max_reads
        ):
            stats = self._process_batch(
                batch_records=pending_records,
                cfg=cfg,
                stats=stats,
                store=store,
                model_adapter=model_adapter,
                selected_layers=selected_layers,
                artifact_root=artifact_root,
            )

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
        self._write_progress(artifact_root, stats)
        return artifact_root

    def _process_batch(
        self,
        batch_records: list,
        cfg: ExtractionConfig,
        stats: ExtractionStats,
        store: ActivationStore,
        model_adapter: ModelAdapter,
        selected_layers: list[int],
        artifact_root: Path,
    ) -> ExtractionStats:
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

            # If a sequence was fully committed before a disconnect, skip safely.
            if record.sequence_id in store.completed_sequence_ids:
                stats = ExtractionStats(
                    total_sequences_seen=stats.total_sequences_seen,
                    total_sequences_kept=stats.total_sequences_kept,
                    total_sequences_skipped=stats.total_sequences_skipped + 1,
                    total_tokens=stats.total_tokens,
                    total_rows_written=stats.total_rows_written,
                )
                self._write_progress(artifact_root, stats)
                continue

            prep = preprocess_record(record, cfg.preprocess)
            if prep.record is None:
                stats = ExtractionStats(
                    total_sequences_seen=stats.total_sequences_seen,
                    total_sequences_kept=stats.total_sequences_kept,
                    total_sequences_skipped=stats.total_sequences_skipped + 1,
                    total_tokens=stats.total_tokens,
                    total_rows_written=stats.total_rows_written,
                )
                self._write_progress(artifact_root, stats)
                continue

            processed_records.append(prep.record)
            preprocess_metadata.append(
                {
                    "invalid_char_count": prep.invalid_char_count,
                    "invalid_fraction": prep.invalid_fraction,
                }
            )

        if not processed_records:
            return stats

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
            store.flush()
            self._write_progress(artifact_root, stats)

        return stats

    def _validate_run_directory(self, artifact_root: Path, resume: bool) -> None:
        if resume:
            return
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError(
                f"Run directory already exists and is not empty: {artifact_root}. "
                "Use a new run_id or set runtime.resume=true."
            )

    def _read_progress(self, artifact_root: Path) -> dict:
        progress_path = artifact_root / "_progress.json"
        if not progress_path.exists():
            return {}
        return json.loads(progress_path.read_text())

    def _write_progress(self, artifact_root: Path, stats: ExtractionStats) -> None:
        artifact_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": utc_now_iso(),
            "total_sequences_seen": stats.total_sequences_seen,
            "total_sequences_kept": stats.total_sequences_kept,
            "total_sequences_skipped": stats.total_sequences_skipped,
            "total_tokens": stats.total_tokens,
            "total_rows_written": stats.total_rows_written,
        }
        tmp_path = artifact_root / "_progress.json.tmp"
        final_path = artifact_root / "_progress.json"
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(final_path)

    def _validate_output_root(self, cfg: ExtractionConfig) -> None:
        output_root = Path(cfg.runtime.output_root).resolve()
        blocked_roots = [
            Path("data/raw_sources").resolve(),
            Path("data/curated_sequences").resolve(),
        ]
        for blocked in blocked_roots:
            try:
                output_root.relative_to(blocked)
            except ValueError:
                continue
            raise ValueError(
                "runtime.output_root must not be inside input data directories. "
                f"Got {output_root}, which is under blocked path {blocked}."
            )

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
