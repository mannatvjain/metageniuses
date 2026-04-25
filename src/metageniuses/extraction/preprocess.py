from __future__ import annotations

from dataclasses import dataclass

from .config import PreprocessConfig
from .schemas import SequenceRecord


@dataclass(frozen=True)
class PreprocessResult:
    record: SequenceRecord | None
    invalid_char_count: int
    invalid_fraction: float
    reason: str | None = None


def clean_sequence(raw_sequence: str, cfg: PreprocessConfig) -> tuple[str, int]:
    sequence = raw_sequence
    if cfg.strip_whitespace:
        sequence = "".join(sequence.split())
    if cfg.uppercase:
        sequence = sequence.upper()

    allowed = set(cfg.allowed_chars)
    out = []
    invalid = 0
    for ch in sequence:
        if ch in allowed:
            out.append(ch)
        else:
            out.append(cfg.replace_invalid_with)
            invalid += 1
    return "".join(out), invalid


def preprocess_record(record: SequenceRecord, cfg: PreprocessConfig) -> PreprocessResult:
    cleaned, invalid_count = clean_sequence(record.sequence, cfg)
    if not cleaned:
        return PreprocessResult(
            record=None,
            invalid_char_count=invalid_count,
            invalid_fraction=1.0,
            reason="empty_after_cleaning",
        )

    invalid_fraction = invalid_count / max(1, len(cleaned))
    if invalid_fraction > cfg.max_invalid_fraction:
        return PreprocessResult(
            record=None,
            invalid_char_count=invalid_count,
            invalid_fraction=invalid_fraction,
            reason="too_many_invalid_characters",
        )

    if len(cleaned) < cfg.min_length:
        return PreprocessResult(
            record=None,
            invalid_char_count=invalid_count,
            invalid_fraction=invalid_fraction,
            reason="too_short",
        )

    if len(cleaned) > cfg.max_length:
        cleaned = cleaned[: cfg.max_length]

    cleaned_record = SequenceRecord(
        sequence_id=record.sequence_id,
        sequence=cleaned,
        metadata=record.metadata,
    )
    return PreprocessResult(
        record=cleaned_record,
        invalid_char_count=invalid_count,
        invalid_fraction=invalid_fraction,
        reason=None,
    )

