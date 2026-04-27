# Future: Token-Level SAE Activations

## Status: Requires GPU re-run

## What we have now

`data/sae_model/features.npy` is **mean-pooled** over tokens. For each sequence, the SAE encoder ran on every token, producing a 32,768-dim sparse vector per token. These were averaged into one vector per sequence (see `src/metageniuses/sae/encode_features.py`, line 80).

This is fine for sequence-level tasks (linear probes, specificity ranking, classification). It's what InterProt used for their protein-level probes.

## What we're missing

Token-level activations — the full (n_tokens × 32,768) sparse matrix per sequence. We need these for:

1. **Activation pattern classification** (InterProt Table 2) — categorize latents as point/motif/periodic/whole by analyzing *where* in the sequence they fire. Impossible with mean-pooled data.

2. **BLAST-grounded feature labeling** — extract the nucleotide subsequences where a feature fires most strongly, submit to NCBI BLAST, get back "93% match to Influenza A polymerase." Requires knowing *which tokens* activated the feature.

3. **Sequence-level activation heatmaps** — for the visualizer (`viz/`), color each token by how strongly a given feature fires. The core visual of the InterProt tool.

4. **Motif discovery** — if a latent fires on a specific 15-nucleotide motif across many sequences, that's a candidate biological signal. Mean-pooling erases this.

## How to get them

Re-run `encode_features.py` (or a modified version) that saves per-token activations instead of mean-pooling. The modification is small — instead of accumulating into `seq_feature_sum`, save the sparse vectors directly.

Storage consideration: 20k sequences × ~150 tokens avg × 64 nonzero features per token (TopK k=64) = ~192M entries. Storing as sparse matrices (COO or CSR) would be manageable (~1-2 GB).

Needs GPU access (RunPod) since it requires the SAE checkpoint + extraction outputs.

## Priority

High — this unlocks the three most visually compelling experiments for the hackathon paper (activation patterns, BLAST labeling, heatmap visualizer). The linear probe works without it, but the "here's what the pathogen-detector feature is actually looking at in the sequence" story requires token-level data.
