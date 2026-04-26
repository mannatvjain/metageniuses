# Experiment 6: Cross-Delivery Generalization

Train probe on class 1 sequences, test on class 2 sequences (different sequencing delivery). Do pathogen-specific SAE features generalize?

## Motivation

The MetaGene-1 paper's Pathogen Detection benchmark (Table 2) specifically tests generalization across sequencing deliveries — different collection locations, pipelines, and dates. If our pathogen-specific SAE features only work on class 1 but fail on class 2, they might be detecting sequencing artifacts (batch effects, adapter contamination) rather than real biology.

This is the validation step that turns "we found pathogen features" into "we found *real* pathogen features."

## Data

- `data/sae_model/features.npy` — (20,000 x 32,768) class 1 SAE features (HAVE)
- `data/human_virus_class1_labeled.jsonl` — class 1 labels (HAVE)
- `data/human_virus_class2_labeled.jsonl` — class 2 labels (HAVE)
- `data/human_virus_class2.jsonl` — class 2 sequences (HAVE)
- **Class 2 SAE features** — `features_class2.npy` — (NEED TO GENERATE)

## What's needed to generate class 2 features

1. MetaGene-1 extraction outputs for class 2 (residual stream activations at layer 32)
   - Either already on RunPod from prior extraction runs, or need to run extraction with class 2 config
2. SAE encoder forward pass on those activations
   - Load `sae_final.pt`, run encoder on class 2 layer-32 activations
   - Aggregate to sequence level (same method used for class 1 — likely max-pool)
   - Save as `features_class2.npy`

This requires GPU access (RunPod) and the SAE encoder code from Peyton's `modeling/sae` branch.

## Pipeline (once class 2 features exist)

### Step 1: Train probe on class 1

Use the same logistic regression from Experiment 2 (train on 100% of class 1, since class 2 is the held-out test).

### Step 2: Evaluate on class 2

Run the trained probe on `features_class2.npy` with class 2 labels. Report accuracy, MCC, AUROC.

### Step 3: Feature stability analysis

For each pathogen-enriched latent from Experiment 1:
- Compute enrichment score on class 2 data
- Compare to class 1 enrichment
- Scatter plot: class 1 enrichment vs class 2 enrichment for all significant latents

If the same latents are enriched in both deliveries, the features are real biology.

### Step 4: Organism consistency

If Experiment 1 labeled latents with organisms (via BLAST):
- BLAST the top-activating class 2 sequences for those same latents
- Do they hit the same organisms?

## Output

Save to `results/cross_delivery/`:

- `summary.json` — class 2 accuracy, MCC, AUROC
- `comparison_table.md` — class 1 vs class 2 performance side by side
- `enrichment_scatter.png` — class 1 vs class 2 enrichment for significant latents
- `feature_stability.csv` — per-latent enrichment in both classes

## Cost

GPU time for extraction + encoding. Local analysis is $0 once features exist.

## Status

**BLOCKED** — need class 2 SAE features. Peyton: can you run this on RunPod?

## What success looks like

- **High class 2 accuracy (within 5-10% of class 1)**: features generalize across deliveries. The SAE learned real biology.
- **Same latents enriched**: the Influenza A detector from class 1 also detects Influenza A in class 2. Strong validation.
- **Low class 2 accuracy**: features are delivery-specific. Concerning but informative — may indicate batch effects in the SAE training data.
