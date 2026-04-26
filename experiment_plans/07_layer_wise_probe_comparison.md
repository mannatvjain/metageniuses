# Experiment 7: Layer-Wise Probe Comparison (SAE Features vs Raw MetaGene-1 Activations)

## Purpose

Determine (a) at which layer in MetaGene-1 pathogen information emerges, and (b) whether the SAE preserves, loses, or enhances that signal compared to the raw residual stream.

This directly replicates the methodology from InterProt Section 4.3 and Table 3 (Adams et al., ICML 2025 — see `INTERPROT_REFERENCE.md` → "Linear Probing (Downstream Tasks)"). InterProt found that SAE probes perform competitively with raw ESM-2 probes across all layers, and for secondary structure prediction, SAE *consistently outperformed* raw ESM. We test whether this holds for MetaGene-1 on pathogen detection — a sequence-level binary classification task analogous to InterProt's subcellular localization probe (also sequence-level, also sklearn logistic regression).

## Why this matters

Reviewers will ask: "Why use an SAE at all if raw activations work just as well?" This experiment provides the answer. If SAE probes match or exceed raw-activation probes, it validates that the SAE's sparse decomposition preserves biological signal while gaining interpretability. If SAE probes are worse, it quantifies the cost of interpretability. Either way, it's a necessary methodological claim.

Additionally, the layer-wise curve tells us where MetaGene-1 builds its pathogen representation. The MetaGene-1 paper (Liu et al., Table 2 — see `METAGENE_REFERENCE.md`) reports 92.96 MCC for pathogen detection using a LoRA adapter on the full model, but doesn't say which layers carry the signal. InterProt found family-specific features peak in early-to-mid layers (see `INTERPROT_REFERENCE.md` → "Key Hyperparameter Effects" → "Layer choice"). We test whether pathogen features follow the same pattern or concentrate in later layers.

## Prior results this builds on

- **Exp 2 (linear probe, layer 32 SAE)**: 94.6% accuracy, 0.892 MCC, 0.987 AUROC on pathogen detection using mean-pooled SAE features. Code at `experiments/linear_probe_pathogen.py`. This is our layer-32 SAE data point.
- **Exp 3 (SAE health check)**: 803 dead latents (2.4%), 31,965 alive. ~892 active features per sequence after mean-pooling. Confirms the SAE is functioning properly.
- **MetaGene-1 paper benchmark**: 92.96 MCC with LoRA fine-tuning (Table 2). Our probe should be compared to this — a linear probe on frozen representations is a weaker baseline than LoRA, so we expect lower MCC but the comparison is informative.

## Prerequisites — what Peyton needs to produce

This experiment requires data that Peyton is currently generating on RunPod. Specifically:

### 1. Raw residual stream activations at layers 8, 16, 24

The extraction pipeline (`src/metageniuses/extraction/`) saves per-token residual stream vectors. The extraction config should be:

```json
{
  "layer_selection": {
    "layers": [8, 16, 24],
    "last_n_layers": null
  }
}
```

The rest of the config matches `configs/extraction/metagene-cloud-prod.json` — same input file (`data/human_virus_class1.jsonl`), same preprocessing, same 20k sequences.

After extraction, the output directory will contain `manifest.json` and `activations/layer_XX/shard_XXXXX.{f32,jsonl}` files. The per-token metadata in the JSONL index includes `sequence_id`, `token_index`, `token_id`, and `layer` (see `src/metageniuses/extraction/extractor.py:198-203`).

### 2. Mean-pooled SAE features at layers 8, 16, 24

After training SAEs at each layer (see below), run `src/metageniuses/sae/encode_features.py` for each layer to produce `features.npy` (20k × 32768) and `sequence_ids.json`. This is the same encoding pipeline that produced `data/sae_model/features.npy` for layer 32.

### 3. SAE training at layers 8, 16, 24

Train one SAE per layer using `src/metageniuses/sae/train.py`:

```bash
python -m metageniuses.sae.train --artifact_root <extraction_output> --layer 8  --output_dir data/sae_layer8
python -m metageniuses.sae.train --artifact_root <extraction_output> --layer 16 --output_dir data/sae_layer16
python -m metageniuses.sae.train --artifact_root <extraction_output> --layer 24 --output_dir data/sae_layer24
```

All SAE hyperparameters stay the same as the layer-32 SAE (see `src/metageniuses/sae/config.py`): d_model=4096, expansion_factor=8 (d_sae=32768), k=64, lr=2e-4, batch_size=4096, n_epochs=10, aux_loss_coeff=1/32, dead_steps_threshold=200, normalize_activations=True.

**Important**: Use the same hyperparameters across all layers so the comparison is fair. The only variable should be the layer.

### 4. Mean-pooled raw activations at layers 8, 16, 24

For the raw-activation probes, we also need mean-pooled raw residual stream vectors (not SAE-encoded) at each layer. This requires a small script that iterates `iter_layer_batches()` and mean-pools per sequence — identical to what `encode_features.py` does but *without* the SAE encode step. The output should be `raw_features_layerXX.npy` (20k × 4096).

## Expected file layout after Peyton's work

```
data/sae_layer8/
  sae_final.pt              # trained SAE weights
  activation_scale.json     # normalization factor
  features.npy              # (20000, 32768) mean-pooled SAE features
  sequence_ids.json         # row-to-sequence_id mapping

data/sae_layer16/
  sae_final.pt
  activation_scale.json
  features.npy
  sequence_ids.json

data/sae_layer24/
  sae_final.pt
  activation_scale.json
  features.npy
  sequence_ids.json

data/sae_model/              # layer 32 — already exists
  sae_final.pt
  features.npy
  sequence_ids.json

data/raw_features/
  raw_features_layer8.npy    # (20000, 4096) mean-pooled raw activations
  raw_features_layer16.npy
  raw_features_layer24.npy
  raw_features_layer32.npy
  sequence_ids.json          # shared — same ordering as SAE features
```

## Implementation

Single script: `experiments/layer_wise_probes.py`

Output directory: `results/layer_wise_probes/`

### Part A: Load and align data (~10 sec)

For each layer L in [8, 16, 24, 32]:
1. Load SAE features: `data/sae_layer{L}/features.npy` (for layer 32: `data/sae_model/features.npy`)
2. Load raw features: `data/raw_features/raw_features_layer{L}.npy`
3. Load sequence IDs from the corresponding `sequence_ids.json`
4. Load labels from `data/human_virus_class1_labeled.jsonl` — field `source` (string "0" or "1"), cast to int
5. Align features to labels by sequence_id (same method as `experiments/linear_probe_pathogen.py:28-41`)

Verify: all layers should have 20,000 sequences. SAE features are 32,768-dim. Raw features are 4,096-dim. Labels are balanced 10k/10k.

### Part B: Train probes (~5-10 min total)

For each layer L in [8, 16, 24, 32], for each feature type in [SAE, raw]:

```python
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

# Use the SAME train/test split across all layers and feature types
# so results are directly comparable
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# Match the probe setup from Exp 2 (experiments/linear_probe_pathogen.py:47-52)
# and InterProt Section 4.3 (sklearn logistic regression for sequence-level tasks)
clf = LogisticRegressionCV(
    Cs=10,                    # 10 regularization values
    cv=StratifiedKFold(5),    # 5-fold stratified CV
    scoring='roc_auc',
    solver='saga',            # handles large feature counts efficiently
    max_iter=5000,            # saga may need more iterations on 32k features
    penalty='l2',
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)
```

**Solver note**: For SAE features (32,768 dims), `saga` solver is recommended — `lbfgs` (the default in Exp 2) may be slow on 32k features. For raw features (4,096 dims), either works. Use `saga` for both for consistency.

**Important**: Use the exact same `random_state=42` and `test_size=0.2` across ALL probes. The comparison is only valid if train/test splits are identical.

Compute metrics for each probe:

```python
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

results = {
    "layer": L,
    "feature_type": feature_type,  # "sae" or "raw"
    "accuracy": accuracy_score(y_test, y_pred),
    "mcc": matthews_corrcoef(y_test, y_pred),
    "auroc": roc_auc_score(y_test, y_prob),
    "auprc": average_precision_score(y_test, y_prob),
    "best_C": clf.C_[0],
    "n_features": X_train.shape[1],
}
```

This produces 8 probes total: 4 layers × 2 feature types.

### Part C: Figures

**C1. Main comparison line plot** (`probe_comparison.png`)

The most important figure. Two lines on one plot:
- x-axis: Layer (8, 16, 24, 32) — evenly spaced
- y-axis: MCC (Matthews Correlation Coefficient)
- Blue line: SAE features (32,768-dim)
- Red line: Raw activations (4,096-dim)
- Horizontal dashed gray line at MCC=0.892 labeled "Exp 2 (layer 32 SAE)" for reference
- Horizontal dashed green line at MCC=0.9296 labeled "MetaGene-1 + LoRA (paper)" for reference
- Markers at each data point with MCC value annotated
- Size: (10, 6), dpi 150
- Title: "Pathogen Detection Accuracy by Layer: SAE Features vs Raw Activations"

**C2. Full metrics table** (`probe_metrics.csv`)

| layer | feature_type | n_features | accuracy | mcc | auroc | auprc | best_C |
|-------|-------------|-----------|----------|-----|-------|-------|--------|
| 8 | raw | 4096 | ... | ... | ... | ... | ... |
| 8 | sae | 32768 | ... | ... | ... | ... | ... |
| 16 | raw | 4096 | ... | ... | ... | ... | ... |
| 16 | sae | 32768 | ... | ... | ... | ... | ... |
| 24 | raw | 4096 | ... | ... | ... | ... | ... |
| 24 | sae | 32768 | ... | ... | ... | ... | ... |
| 32 | raw | 4096 | ... | ... | ... | ... | ... |
| 32 | sae | 32768 | ... | ... | ... | ... | ... |

**C3. ROC curves by layer** (`roc_by_layer.png`)

4 subplots (one per layer). Each subplot has two ROC curves (SAE vs raw) with AUROC in legend. Size: (14, 10), dpi 150.

**C4. Delta plot** (`sae_vs_raw_delta.png`)

Bar chart: x-axis = layer, y-axis = MCC(SAE) - MCC(raw). Positive = SAE better, negative = raw better. Color: green for positive, red for negative. Size: (8, 5), dpi 150.

### Part D: Summary (`summary.json`)

```json
{
    "experiment": "layer_wise_probe_comparison",
    "n_layers": 4,
    "layers": [8, 16, 24, 32],
    "train_size": 16000,
    "test_size": 4000,
    "random_state": 42,
    "probe_type": "LogisticRegressionCV_saga_l2",
    "results": [ ... ],
    "best_sae_layer": { "layer": ..., "mcc": ... },
    "best_raw_layer": { "layer": ..., "mcc": ... },
    "sae_vs_raw_avg_delta_mcc": ...,
    "conclusion": "..."
}
```

## Dependencies

```
pip install numpy scipy scikit-learn matplotlib
```

No GPU needed — this runs on the pre-computed features.

## Runtime estimate

| Step | Time | Compute |
|------|------|---------|
| A: Load data | ~10 sec | Local (numpy) |
| B: Train 8 probes | ~5-10 min | Local (sklearn, saga solver) |
| C: Figures | ~30 sec | Local (matplotlib) |
| **Total** | **~10 min** | **$0** |

## What success looks like

1. **SAE probes match or exceed raw probes** at most layers → validates the SAE preserves biological signal. This replicates InterProt's finding (Section 4.3: "SAE probes perform competitively with ESM probes").
2. **Pathogen information peaks at middle layers** → consistent with InterProt's finding that family-specific features peak at early-to-mid layers (`INTERPROT_REFERENCE.md` → "Layer choice").
3. **Layer 32 is not the best** → suggests earlier layers may have sharper organism-specific features, motivating Experiments 8 and 10 at those layers.

## What failure looks like

- **SAE probes significantly worse than raw** → the SAE is lossy for pathogen detection. May need more latents (higher expansion factor), different k, or better training. Still publishable as "the cost of interpretability."
- **Flat curve across layers** → pathogen information is present from early layers and doesn't change. Less interesting scientifically but still validates multi-layer extraction.

## What NOT to edit

- `src/metageniuses/sae/analyze.py` — Peyton's pipeline
- `tests/sae/test_analyze.py` — Peyton's tests
- `pyproject.toml` — shared config

## Instructions for Codex

This experiment depends on files Peyton generates on RunPod. Before writing the script:

1. Read `experiments/linear_probe_pathogen.py` — this is your template for data loading, probe training, and metric computation. Match its structure.
2. Read `src/metageniuses/sae/encode_features.py` — understand how features.npy is produced (mean-pooled SAE activations per sequence).
3. Read `src/metageniuses/extraction/contracts.py` — understand `iter_layer_batches()` which yields `(vectors, metadata)` per token. The metadata dict has keys: `sequence_id`, `token_index`, `token_id`, `layer`.
4. The script needs to handle the case where some layer directories may not exist yet. Check for file existence and print a clear error if missing.
5. Write a helper function `mean_pool_raw_activations(artifact_root, layer) -> (np.ndarray, list[str])` that uses `iter_layer_batches()` to produce mean-pooled raw activation vectors — same logic as `encode_features.py:46-69` but without the SAE encode step. This is needed if `raw_features_layerXX.npy` doesn't exist.
6. All file paths should be configurable via argparse with sensible defaults matching the layout above.
7. Print progress as probes train — each probe takes 30-90 seconds on 32k features.
