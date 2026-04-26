# Feature Labeling MVP

Label the top 100 most specific SAE latents using BLAST + LLM synthesis.

## Motivation

We have ~10-15k SAE latents. Manual inspection doesn't scale, and we lack deep bio expertise. But we don't need to label all of them — 100 well-grounded features with evidence is enough for the hackathon submission.

## Pipeline

### Step 1: Rank features by specificity (free, local)

For each latent, compute how unevenly it activates across known labels (virus class, source type, etc.).

```python
specificity_score = max_class_count / total_activating_sequences
```

Sort descending. Take top 100.

### Step 2: Extract activating subsequences

For each of the 100 features, extract the nucleotide regions where the feature fires on its top ~10 activating sequences.

### Step 3: BLAST the subsequences (free, NCBI API)

Submit activating subsequences to NCBI BLAST REST API. Get back hits like "93% match to E. coli 16S ribosomal RNA."

~100 features x ~10 subsequences = ~1000 BLAST queries. Free, just rate-limited.

### Step 4: LLM labels (cheap)

Per-feature dossier sent to an LLM (Haiku or Sonnet):
- Specificity stats ("85% of activating sequences are class-1 virus")
- Activation pattern ("short motif, median 14nt")
- BLAST hits for activating regions ("top hits: Influenza A polymerase")

100 calls = ~$1-2 total.

### Step 5: Report

Table of 100 features: (feature_id, label, confidence, specificity_score, BLAST evidence). Cherry-pick 5-10 best for writeup figures.

## Cost Estimate

| Component              | Cost                |
|------------------------|---------------------|
| Specificity ranking    | $0 (local compute)  |
| BLAST queries          | $0 (NCBI public API) |
| LLM labeling (100x)   | ~$1-2               |
| **Total**              | **~$2**             |

## What This Proves

"We found SAE latents that specifically activate on viral sequences, and BLAST confirms the activating regions correspond to known pathogen genes."

We don't need 15k labels. We need 10 compelling examples with evidence.

## Risks (from inversion analysis)

- **Hallucinated labels**: Mitigated by BLAST grounding — LLM synthesizes real database hits, not guessing.
- **Generic labels**: If BLAST returns nothing useful for a feature, label it "unknown/ungrounded" rather than forcing a name.
- **Scope creep**: This is capped at 100 features. Do not expand until these 100 are done and validated.

## Validation

- Spot-check ~20 features manually: do the BLAST hits make sense for the specificity stats?
- Check agreement between statistical specificity and LLM label.
- Flag any feature where LLM label contradicts BLAST evidence.

## Prerequisites

- Trained SAE weights (Bridget/Peyton's work)
- Extraction outputs (activation shards from RunPod)
- Test sequences with known labels (class 1 + class 2 JSONL files)
