# metageniuses

Hackathon project for interpretable metagenomic modeling:

1. Curated sequence inputs
2. MetaGene-style forward pass
3. Hidden-state extraction for SAE training
4. Downstream analysis and visualization (later phases)

Current repo focus is step 2-3: robust extraction + storage contracts.

## Current Project Structure

```text
metageniuses/
├── src/metageniuses/extraction/      # extraction pipeline, adapters, storage, contracts, CLI
├── configs/extraction/               # run configs (test, local smoke, cloud production)
├── data/raw_sources/                 # uploaded source CSV datasets
├── data/curated_sequences/           # forward-pass-ready JSONL inputs + summary
├── data/test-activations/            # local/test extraction outputs
├── results/                          # cloud/production extraction outputs
├── docs/architecture/                # extraction architecture docs
├── docs/datasets/                    # dataset curation docs
├── tests/extraction/                 # extraction unit/contract tests
└── tests/fixtures/                   # tiny local test data
```

## Key Files

- `configs/extraction/default.json`: baseline config (writes to `results/extraction/`)
- `configs/extraction/tiny-test.json`: local no-download test config
- `configs/extraction/metagene-smoke-local.json`: real-weights local smoke run (writes to `data/test-activations/`)
- `configs/extraction/metagene-cloud-prod.json`: cloud production run preference (`4 layers`, `20,000 sequences`, fresh run directory under `results/cloud-extraction/` each launch)
- `data/curated_sequences/forward_pass_unique_sequences.jsonl`: deduplicated default forward-pass input
- `data/curated_sequences/forward_pass_summary.json`: row counts, dedupe counts, filtering stats
- `docs/architecture/residual_extraction.md`: extraction component contract and design
- `docs/architecture/runpod_setup.md`: RunPod deployment/run instructions
- `docs/datasets/forward_pass_dataset.md`: how raw uploads were organized and curated

## Quickstart

Run local fake-model extraction test:

```bash
PYTHONPATH=src python3 -m metageniuses.extraction.cli \
  --config configs/extraction/tiny-test.json \
  --adapter fake
```

Run unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/extraction -p 'test_*.py'
```

## Data Organization Rules

1. Put uploaded source files in `data/raw_sources/`.
2. Keep extraction-ready inputs in `data/curated_sequences/`.
3. Keep local/test run artifacts in `data/test-activations/` (ignored by git).
4. Keep cloud/production run outputs in `results/` (ignored by git).
5. Do not write extraction outputs into `data/curated_sequences/` or `data/raw_sources/`.

## Current Status

Implemented:

1. Configurable `model_id` selection before forward pass
2. Configurable layer selection before forward pass
3. Token-level hidden-state extraction and sharded storage
4. Resume support for interrupted runs (`--resume` with same `run_id`)
5. Manifest + loader contract for later SAE module consumption
6. No-download fake adapter path for reliable local testing

Not implemented yet:

1. SAE training
2. Biological interpretation
3. Website/visualization app
