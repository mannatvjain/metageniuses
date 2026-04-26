# Linear Probe: Pathogen Detection from SAE Features

Can SAE features trained on MetaGene-1 predict whether a metagenomic sequence is pathogenic? Which specific latents are most predictive?

## Motivation

MetaGene-1 achieves SOTA on pathogen detection (92.96 MCC avg, Table 2 in paper) using LoRA fine-tuning on the full model. But that's a black box — you get a score, not a reason.

If a linear probe on SAE features achieves comparable accuracy, it means the SAE has decomposed MetaGene-1's pathogen knowledge into interpretable directions. The probe coefficients directly tell us *which* latents encode pathogenicity, and we can inspect those latents to ask: what biological concept is this feature capturing?

This mirrors InterProt Section 5 (interpretable probing), adapted for our domain.

## Data

- **Features**: `data/sae_model/features.npy` — (20,000 x 32,768), sequence-level SAE activations from class 1 human virus sequences
- **Labels**: `data/human_virus_class1_labeled.jsonl` — `source` field: `0` = non-pathogen, `1` = pathogen (10k each, balanced)
- **Sequence IDs**: `data/sae_model/sequence_ids.json` — maps row index in features.npy to sequence_id in the labeled JSONL

## Pipeline

### Step 1: Load and align features with labels

Join `features.npy` rows to `source` labels via `sequence_ids.json`. Verify all 20k sequences have labels.

### Step 2: Train linear probe

- Model: sklearn `LogisticRegressionCV` (L2-regularized, cross-validated over regularization strength)
- Split: 80/20 stratified train/test
- Metric: accuracy, MCC (to match MetaGene-1 paper), AUROC
- Baseline: logistic regression on raw MetaGene-1 embeddings (mean-pooled last hidden state) if available; otherwise compare to random (50% accuracy)

### Step 3: Inspect top predictive latents

- Extract probe coefficients (1 x 32,768 weight vector)
- Rank latents by absolute coefficient magnitude
- Report top 20 most predictive latents (10 most positive = pathogen-associated, 10 most negative = non-pathogen-associated)
- For each: report coefficient sign/magnitude, number of sequences it activates on, mean activation in pathogen vs non-pathogen class

### Step 4: Enrichment validation

Cross-check: do the top probe latents also show high enrichment scores (frequency_in_pathogen / frequency_in_all)?

This validates that the probe isn't relying on noisy or low-signal features — the most predictive latents should also be the most class-specific.

### Step 5: Report

- Table: top 20 latents with (latent_id, coefficient, activation_rate_pathogen, activation_rate_nonpathogen, enrichment_score)
- Figures: (1) coefficient distribution histogram, (2) ROC curve, (3) bar chart of top latent activations by class
- Headline number: probe accuracy + MCC

## Cost Estimate

| Component | Cost |
|-----------|------|
| Everything | $0 (local compute, sklearn on 20k x 32k matrix) |

## What This Proves

**If accuracy is high**: MetaGene-1's SAE has learned interpretable features that encode pathogenicity. The top latents are candidate "pathogen detector" features — specific directions in activation space that fire on dangerous sequences. This is the core biosecurity result: we can point to *which* internal features drive a pathogen flag, not just that the model flagged it.

**If accuracy is low**: The SAE at layer 32 hasn't cleanly separated pathogen signal into individual latents — the information may be distributed, in a different layer, or lost during SAE compression. Still informative: tells us something about where MetaGene-1 represents pathogenicity.

## Comparison to MetaGene-1 paper

The MetaGene-1 paper's pathogen detection benchmark (Table 2) uses LoRA fine-tuning on the full model. Our probe is intentionally weaker (linear, on compressed SAE features). The gap between our probe accuracy and their 92.96 MCC measures how much pathogen information survives SAE compression — smaller gap = better SAE.

## Prerequisites

- `features.npy`, `sequence_ids.json`, `human_virus_class1_labeled.jsonl` (all present in repo)
- Python with numpy, sklearn, matplotlib
