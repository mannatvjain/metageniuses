# Experiment 2: Linear Probe — Pathogen Detection

Prove that SAE features linearly separate pathogen from non-pathogen sequences. Identify which latents carry the most predictive weight.

## Motivation

MetaGene-1 achieves 92.96 MCC on pathogen detection using LoRA fine-tuning — but that's a black box. If a linear probe on SAE features achieves comparable accuracy, it means the SAE has decomposed MetaGene-1's pathogen knowledge into interpretable directions. The probe coefficients directly tell us *which* latents encode pathogenicity.

This mirrors InterProt Section 5 (interpretable probing): they trained linear probes on SAE features for secondary structure, subcellular localization, and thermostability, then inspected the top-weighted latents.

## Data

- `data/sae_model/features.npy` — (20,000 x 32,768) sequence-level SAE activations
- `data/sae_model/sequence_ids.json` — maps row index to sequence_id
- `data/human_virus_class1_labeled.jsonl` — `source` field: 0 = non-pathogen, 1 = pathogen (10k each)

## Pipeline

### Step 1: Load and align

Join `features.npy` rows to `source` labels via `sequence_ids.json`. Verify all 20k sequences have labels and class balance is 10k/10k.

### Step 2: Train linear probe

- Model: `sklearn.linear_model.LogisticRegressionCV` (L2-regularized, CV over regularization strength)
- Split: 80/20 stratified train/test (random_state=42)
- Metrics: accuracy, MCC (to match MetaGene-1 paper), AUROC
- Baseline comparison: random = 50% accuracy / 0.0 MCC

### Step 3: Inspect top predictive latents

- Extract probe coefficient vector: `clf.coef_[0]` — shape (32,768,)
- Rank latents by absolute coefficient magnitude
- Top 10 most positive = pathogen-associated
- Top 10 most negative = non-pathogen-associated
- For each: compute activation frequency and mean activation in each class

### Step 4: Enrichment cross-check

Compare the probe's top-20 latents with the enrichment analysis from Experiment 1. Do they agree? If the most predictive latents are also the most pathogen-enriched, that's strong validation.

### Step 5: Output

Save to `results/linear_probe_pathogen/`:

- `summary.json` — accuracy, MCC, AUROC, best regularization C
- `top_latents.json` — top 20 latents with coefficients and per-class stats
- `coefficient_distribution.png` — histogram of all 32,768 probe weights, with top latents marked
- `roc_curve.png` — ROC curve with AUROC annotation
- `top_latent_activations.png` — grouped bar chart: mean activation of top 5 pathogen and top 5 non-pathogen latents, by class

## Cost

$0. sklearn on a 20k x 32k matrix, runs in under a minute.

## Code

Already written at `experiments/linear_probe_pathogen.py`. Just needs `pip install scikit-learn matplotlib` and then `python experiments/linear_probe_pathogen.py`.

## What success looks like

- **High accuracy (>80%, MCC >0.6)**: SAE features encode pathogenicity. The gap to MetaGene-1's 92.96 MCC measures information lost during SAE compression.
- **Low accuracy (~50%)**: Pathogen information isn't linearly accessible from sequence-level SAE features at layer 32. Still informative — tells us about what this layer represents.
