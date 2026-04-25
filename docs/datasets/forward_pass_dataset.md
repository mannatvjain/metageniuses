# Forward Pass Dataset Organization

## Raw Sources

Uploaded CSV sources are stored in:

- `data/raw_sources/`

Files:

1. `human_microbiome_disease.csv`
2. `human_microbiome_multi-label.csv`
3. `human_microbiome_sex.csv`
4. `human_microbiome_source.csv`
5. `human_virus_infecting.csv`
6. `human_virus_reference.csv`

The uploaded doc is stored in:

- `docs/datasets/data_sources.docx`

## Curated Inputs

Curated extraction inputs are stored in:

- `data/curated_sequences/forward_pass_all_rows.jsonl`
- `data/curated_sequences/forward_pass_unique_sequences.jsonl`
- `data/curated_sequences/forward_pass_summary.json`

`forward_pass_unique_sequences.jsonl` is the default extraction input in `configs/extraction/default.json`.

## Curation Rules

1. All sequences are uppercased.
2. Keep records with sequence length `50..5000` bp.
3. Merge duplicate sequences across source files by exact sequence hash.
4. Preserve merged metadata as `label_*` fields and `source_datasets`.
5. Keep `source_row_ids` truncated to first 10 IDs per merged sequence.

## Notes

`human_virus_reference.csv` includes very long outlier sequences; records over 5000 bp are excluded from the curated forward-pass input.

