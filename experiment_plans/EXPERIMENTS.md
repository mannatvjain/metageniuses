# Experiments

Master list of experiments for the hackathon submission. All runnable on local compute unless noted.

## Data we have

- `data/sae_model/features.npy` — (20,000 x 32,768) sequence-level SAE activations, class 1 human virus sequences
- `data/sae_model/sae_final.pt` — trained SAE weights (layer 32, d_model=4096, expansion=8x, k=64)
- `data/sae_model/sequence_ids.json` — maps row index to sequence_id
- `data/human_virus_class1_labeled.jsonl` — 20k sequences with `source` label (0=non-pathogen, 1=pathogen, 10k each) and nucleotide sequences
- `data/human_virus_class2_labeled.jsonl` — held-out test set (20k, labeled, no SAE features yet)
- 803 dead latents, 31,965 alive latents

---

## Experiment 1: Organism-Specific Pathogen Detectors

**The main result.** Find SAE latents that fire specifically on pathogen sequences, then use BLAST to identify *which organism* each latent is detecting.

### Motivation

MetaGene-1 can detect pathogens (92.96 MCC). But *why*? We show it has internally learned organism-level detectors — individual SAE features that fire on Influenza A, or SARS-CoV-2, or norovirus — without ever being trained with organism labels.

This is the analog of InterProt finding protein-family-specific features (Section 4.2, Figure 3a) and Bridget's SURF finding PBD-family-specific features. But for pathogen species in metagenomic data.

### Pipeline

**Step 1: Enrichment scan.** For each of 32,768 latents, test whether it fires disproportionately on pathogen (source=1) vs non-pathogen (source=0) sequences.

Metrics (compute all three, they answer different questions):
- **Fisher's exact test** on 2x2 contingency table (active/inactive x pathogen/non-pathogen). FDR-correct across 32k tests. Standard bioinformatics enrichment — defensible in a paper.
- **Log-fold-change + Wilcoxon rank-sum** on continuous activation values. Produces a volcano plot (log2FC vs -log10 p-value). Standard differential expression style — every biologist recognizes this.
- **InterProt-style F1 sweep** (Appendix A.6). For each latent, sweep activation thresholds 0.1-0.9, compute F1 for binary pathogen classification. If max F1 > 0.7, call it "pathogen-specific." Direct comparability to InterProt's family-specificity results.

Output: ranked list of pathogen-specific latents with significance scores.

**Step 2: Sequence retrieval.** For each top pathogen-specific latent (~top 50), sort pathogen sequences by activation value for that latent. Pull top 10 highest-activating sequences (nucleotide strings from JSONL).

**Step 3: BLAST.** Submit top sequences to NCBI BLAST REST API (blastn, nt database). For each sequence, record top hit organism, percent identity, e-value, gene annotation if available.

**Step 4: Organism clustering.** Per latent: do the BLAST hits cluster by organism? If 8/10 top sequences for latent X hit Influenza A, that latent is an Influenza A detector. If hits are mixed, the latent detects something more generic (viral motif, GC pattern, etc.).

**Step 5: Report.**

Table (the main figure of the paper):

| Latent | Enrichment | p-value (FDR) | Top BLAST hits | Proposed label |
|--------|-----------|---------------|---------------|----------------|
| 7241 | 4.2x | 1.2e-15 | 9/10 Influenza A polymerase | Influenza A detector |
| 12803 | 3.8x | 3.4e-12 | 8/10 SARS-CoV-2 spike | Coronavirus spike detector |
| ... | | | | |

Figures:
- Volcano plot (all 32k latents, pathogen-enriched in red, non-pathogen in blue)
- Bar chart: top 10 pathogen-specific latents with their organism labels
- Example sequences with BLAST annotations for cherry-picked features

### Cost

| Component | Cost |
|-----------|------|
| Enrichment scan | $0 (numpy/scipy) |
| Sequence retrieval | $0 |
| BLAST (~500 queries) | $0 (NCBI public API, rate-limited) |
| **Total** | **$0** |

### Dependencies

All present in repo. No GPU needed.

---

## Experiment 2: Linear Probe — Pathogen Detection

**The setup result.** Prove that SAE features linearly separate pathogen from non-pathogen before diving into which individual features matter.

### Pipeline

1. Logistic regression (L2, CV over regularization) on features.npy → predict source label
2. 80/20 stratified train/test split
3. Report accuracy, MCC, AUROC
4. Extract probe coefficient vector (1 x 32,768) — rank latents by weight magnitude
5. Compare top probe latents to top enrichment latents from Experiment 1

### What you'd show

- Headline number: accuracy + MCC (compare to MetaGene-1's 92.96 MCC with LoRA)
- ROC curve
- Coefficient distribution histogram (most weights near zero, few large — those are the interesting ones)
- Top 10 pathogen-predictive and top 10 non-pathogen-predictive latents with activation stats
- Overlap analysis: do probe top-20 and enrichment top-20 agree?

### Why this matters separately from Experiment 1

Enrichment finds features that are individually class-specific. The probe finds features useful *in combination*. A feature could have low enrichment but high probe weight (subtle signal, useful in combination) or high enrichment but low probe weight (redundant with another feature). Showing both strengthens the story.

### Cost

$0. Code already written at `experiments/linear_probe_pathogen.py`.

---

## Experiment 3: SAE Health Check (Dead/Alive Census)

**Sanity check.** Descriptive statistics on the SAE — how many features are alive, how active are they, is the sparsity what we expect?

### What to compute

- Dead vs alive count (already know: 803 dead, 31,965 alive)
- Distribution of "number of sequences activating each latent"
- Distribution of max activation values
- Mean number of active features per sequence (should be ~k=64)
- Comparison to InterProt's stats

### What you'd show

- Histogram: sequences-per-latent distribution (expect long tail)
- Histogram: max activation distribution
- One-line stats table: dead count, alive count, mean active features per sequence
- Training curves (already have `sae_training_curves.png`)

### Cost

$0. Fastest experiment — 5 minutes.

---

## Experiment 4: Sequence UMAP

**Visual intuition.** Project the 20,000 sequences from 32,768-dim SAE space to 2D. Do pathogen and non-pathogen sequences naturally separate?

### Pipeline

1. PCA to 50 dims (for speed)
2. UMAP to 2D
3. Color by pathogen label

### What you'd show

- UMAP scatter: 20k points, red (pathogen) vs blue (non-pathogen)
- If they separate: "SAE features naturally cluster pathogenic sequences" — visual confirmation of probe result
- If they don't: still interesting — what structure *does* emerge? Sub-clusters within pathogen class could correspond to different organisms (ties into Experiment 1)

### Cost

$0. umap-learn + matplotlib.

---

## Experiment 5: Feature Clustering

**Hypothesis generation.** Cluster the 32k latents by co-activation pattern. Do natural "modules" of features emerge?

### Pipeline

1. Transpose features.npy to (32,768 x 20,000)
2. PCA to 50 dims, UMAP to 2D
3. Cluster with HDBSCAN
4. For each cluster: compute mean pathogen enrichment

### What you'd show

- UMAP of latents (32k points), colored by cluster
- Same UMAP colored by pathogen enrichment score — do pathogen-specific latents cluster together?
- Cluster summary table: cluster ID, size, mean enrichment, proposed interpretation

### Why this matters

If pathogen-specific latents cluster together, it suggests the SAE has learned an organized "pathogen module" — a group of co-activating features that collectively represent pathogenicity. If they're scattered, pathogen detection is distributed across many independent features.

### Cost

$0. Lower priority than experiments 1-4.

---

## Experiment 6: Cross-Delivery Generalization

**Validation.** Train probe on class 1, test on class 2 (different sequencing delivery). Do pathogen features generalize?

### What we need

**Class 2 SAE features** — run 20k class 2 sequences through MetaGene-1 extraction + SAE encoder to produce `features_class2.npy`. Requires GPU (RunPod) + SAE encoder code from Peyton's branch.

### What you'd show

- Probe accuracy/MCC on class 2 vs class 1 test split
- Do the same pathogen-enriched latents from Experiment 1 show enrichment on class 2?
- If yes: features are real biology, not sequencing artifacts

### Status

**Blocked** on class 2 feature encoding. Peyton — can you run this?

---

---

## Experiment 7: Layer-Wise Probe Comparison (SAE vs Raw MetaGene-1)

**Methodological validation.** Train linear probes on SAE features vs raw residual stream activations at layers 8, 16, 24, 32. Does the SAE preserve pathogen signal? Where does pathogen information emerge?

Replicates InterProt Section 4.3 / Table 3. See `experiment_plans/07_layer_wise_probe_comparison.md`.

### Dependencies

- **Requires multi-layer data from Peyton**: extraction at layers 8, 16, 24 + SAE training + encoding at each layer + mean-pooled raw activations
- Exp 2 results (layer 32 baseline)

### Cost

$0 once features exist. ~10 min local compute.

---

## Experiment 8: Activation Pattern Classification (Multi-Layer)

**Free table for the paper.** Classify every latent at every layer as dead/point/short motif/medium motif/long motif/periodic/whole-sequence. Compare distribution across layers.

Direct replication of InterProt Table 2 / Section 4.1. See `experiment_plans/08_activation_pattern_classification.md`.

### Dependencies

- **Requires per-token SAE activations from Peyton** (sparse NPZ format at each layer)
- No labels needed — purely structural analysis

### Cost

$0. ~25-45 min local compute (4 layers × ~10 min each).

---

## Experiment 10: Token-Level Pathogen Localization

**The hero figure.** For individual pathogen sequences, identify exactly which tokens drive the pathogen classification. BLAST the high-signal subsequences to identify the specific gene.

Nucleotide analog of InterProt's core visualization (Figure 1). See `experiment_plans/10_token_level_pathogen_localization.md`.

### Dependencies

- **Requires per-token SAE activations from Peyton** (same data as Exp 8)
- Exp 2 probe coefficients (layer 32)
- Exp 7 probe coefficients (layers 8, 16, 24) for multi-layer version
- Exp 1 BLAST results for organism cross-referencing

### Cost

$0 for scoring. ~15-30 min for BLAST on hotspot subsequences.

---

## Priority order

| Priority | Experiment | Time | Status |
|----------|-----------|------|--------|
| 1 | Organism-Specific Pathogen Detectors | ~1hr | Script written, needs run |
| 2 | Linear Probe | ~10 min | **Done** (94.6% acc, 0.892 MCC) |
| 3 | SAE Health Check | ~5 min | **Done** (803 dead, 31965 alive) |
| 4 | Sequence UMAP | ~10 min | **Done** (clear separation, 49 clusters) |
| 5 | Feature Clustering | ~15 min | **Done** (12 clusters, pathogen gradient) |
| 6 | Cross-Delivery Generalization | ~30 min | Script written, needs run |
| 7 | Layer-Wise Probe Comparison | ~10 min | Blocked on multi-layer data (Peyton) |
| 8 | Activation Pattern Classification | ~30 min | Blocked on per-token activations (Peyton) |
| 10 | Token-Level Pathogen Localization | ~30 min | Blocked on per-token activations (Peyton) |

Experiments 1-5 use existing layer-32 mean-pooled data. Experiments 7, 8, 10 require the multi-layer per-token data Peyton is generating on RunPod.
