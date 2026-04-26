# Future: Multi-Class Virus Species Classifier

## Status: Requires BLAST labels + GPU (target: ICML, not hackathon)

## Goal

Train a classifier that predicts which specific virus species a sequence comes from, using SAE features. Goes beyond binary pathogen/non-pathogen detection to "this is Norovirus GII" or "this is Human astrovirus 1."

## Prerequisites

1. **Per-sequence species labels** — BLAST all ~10k pathogen sequences (source=1) against NCBI to get organism IDs. Partial BLAST data already exists (`results/organism_detectors/blast_results_partial.json`) covering 40 unique organisms across 14 latents, but need per-sequence labels for all pathogen sequences.

2. **SAE features** — already have `features.npy` (20k x 32768) for class 1. Could also use per-token activations (once saved) for richer signal.

## Approach

- Use existing `features.npy` as input (or token-level features if available)
- Multi-class logistic regression or random forest on species labels from BLAST
- Template: `experiments/linear_probe_pathogen.py` — swap binary `source` label for multi-class organism label
- Evaluate per-species precision/recall, confusion matrix across virus families
- Connect back to SAE latents: "latent X is a Norovirus detector" validated by both BLAST and classifier coefficients

## Why this matters

Stronger claim than BLAST-on-top-sequences alone: the SAE features can *predict* virus species, meaning the latents have learned taxonomically meaningful representations. This is the "interpretable surveillance" story — not just "pathogen detected" but "which pathogen."

## Storage of BLAST labels

BLAST all 10k pathogen sequences via NCBI API (~5 seq/min batched, few hours). Save as `data/blast_labels/sequence_species.json` mapping sequence_id to top BLAST hit organism.
