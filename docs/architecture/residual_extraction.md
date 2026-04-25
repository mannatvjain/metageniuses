# Residual Extraction Component (MetaGene-1)

## Scope

This module does one job:

1. Load curated metagenomic reads.
2. Run a configured model forward pass.
3. Capture configured transformer hidden-state layers.
4. Store token-level activations + metadata for later SAE training and visualization.

Out of scope:

1. SAE training.
2. Biological interpretation.
3. Website/UI.

## Design Decisions

1. `model_id` is configurable in config.
2. Layer count is fixed before extraction starts via `layer_selection`.
3. Layer indices use transformer-layer numbering starting at `1`.
4. Captured rows are token-level hidden states for selected layers.

## Artifact Contract

Output root:

- `manifest.json`
- `sequences.jsonl`
- `activations/layer_XX/shard_*.f32`
- `activations/layer_XX/shard_*.jsonl`

`manifest.json` declares:

1. Schema version.
2. Model/tokenizer/revision and dimensions.
3. Selected layers.
4. Preprocess/runtime configs.
5. Row counts and shard file map.

Per-layer shard contract:

1. Binary file (`.f32`) stores row-major `float32` vectors of width `d_model`.
2. Parallel JSONL index stores one metadata row per activation vector.
3. Row order in index matches row order in binary.

## SAE Interface

`metageniuses.extraction.contracts.iter_layer_batches(...)` yields:

1. `vectors`: `list[list[float]]` with shape `[batch_size, d_model]`.
2. `metadata`: row-level metadata aligned to vectors.

This makes SAE training independent of model loading.

