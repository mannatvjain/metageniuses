# metageniuses

Residual extraction scaffold for MetaGene-style hidden-state capture is in:

- `src/metageniuses/extraction/`
- `configs/extraction/`
- `docs/architecture/residual_extraction.md`

Forward-pass dataset organization:

- `data/raw_sources/` for uploaded source CSVs
- `data/curated_sequences/forward_pass_unique_sequences.jsonl` as default curated input
- `docs/datasets/forward_pass_dataset.md` for curation details

Run local fake-model extraction:

```bash
PYTHONPATH=src python3 -m metageniuses.extraction.cli \
  --config configs/extraction/tiny-test.json \
  --adapter fake
```
