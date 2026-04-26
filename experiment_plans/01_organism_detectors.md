# Experiment 1: Organism-Specific Pathogen Detectors

Find SAE latents that fire specifically on pathogen sequences, then use BLAST to identify *which organism* each latent is detecting.

## Motivation

MetaGene-1 can detect pathogens (92.96 MCC). But *why*? We show it has internally learned organism-level detectors — individual SAE features that fire on Influenza A, or SARS-CoV-2, or norovirus — without ever being trained with organism labels.

This is the analog of InterProt finding protein-family-specific features (Section 4.2, Figure 3a) and SURF finding PBD-family-specific features. But for pathogen species in metagenomic data.

## Data

- `data/sae_model/features.npy` — (20,000 x 32,768) sequence-level SAE activations
- `data/sae_model/sequence_ids.json` — maps row index to sequence_id
- `data/human_virus_class1_labeled.jsonl` — sequences with `source` (0=non-pathogen, 1=pathogen, 10k each), includes nucleotide `sequence` field
- `data/human_virus_class1.jsonl` — same sequences without labels (use labeled version)

## Pipeline

### Step 1: Enrichment scan

For each of 32,768 latents, test whether it fires disproportionately on pathogen vs non-pathogen sequences. Compute all three metrics:

**1a. Fisher's exact test.** Build a 2x2 contingency table per latent:

```
                Pathogen    Non-pathogen
Active            a              b
Inactive          c              d
```

Use `scipy.stats.fisher_exact`. Collect odds ratios and p-values. FDR-correct across 32,768 tests using Benjamini-Hochberg (`statsmodels.stats.multitest.multipletests` with `method='fdr_bh'`).

A latent is "pathogen-enriched" if FDR-adjusted p < 0.01 and odds ratio > 1. "Non-pathogen-enriched" if FDR-adjusted p < 0.01 and odds ratio < 1.

**1b. Log-fold-change + Wilcoxon rank-sum.** For each latent:
- `log2FC = log2(mean_activation_pathogen / mean_activation_nonpathogen)` (add small epsilon to avoid division by zero)
- Wilcoxon rank-sum test (`scipy.stats.mannwhitneyu`) on activation values between classes
- FDR-correct p-values

This uses continuous activation values, not just binary active/inactive.

**1c. InterProt-style F1 sweep.** For each latent, treat its activation as a classifier score for pathogen detection:
- Sweep thresholds from 0.1 to 0.9 of normalized max activation
- At each threshold, compute F1 for binary pathogen classification
- Record max F1 across thresholds
- If max F1 > 0.7, call it "pathogen-specific" (matching InterProt's criterion)

### Step 2: Sequence retrieval

Take the top ~50 most pathogen-enriched latents (by Fisher's odds ratio, FDR-significant).

For each latent:
- Sort all 10,000 pathogen sequences by activation value for that latent (descending)
- Pull the top 10 highest-activating sequences
- Extract their nucleotide strings from the labeled JSONL

### Step 3: BLAST

For each of the ~500 sequences (50 latents x 10 sequences):
- Submit to NCBI BLAST REST API (`blastn` against `nt` database)
- Use the REST API: PUT to `https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi` with `CMD=Put`, poll with `CMD=Get`
- Record: top hit organism, percent identity, e-value, gene annotation, accession

Rate limiting: NCBI allows ~3 requests/second. 500 queries will take ~10-15 minutes with polling.

### Step 4: Organism clustering

Per latent: tally the organisms from the top 10 BLAST hits.
- If >= 7/10 hits map to the same organism or genus → label the latent as a detector for that organism
- If hits are mixed → label as "generic viral motif" or "unresolved"
- Record the dominant organism, hit consistency (e.g., "9/10 Influenza A"), and representative gene annotations

### Step 5: Output

Save to `results/organism_detectors/`:

- `enrichment_results.csv` — all 32,768 latents with Fisher OR, p-value, FDR, log2FC, Wilcoxon p, F1 score
- `pathogen_specific_latents.csv` — filtered to significant pathogen-enriched latents
- `blast_results.json` — per-latent BLAST hits
- `organism_labels.csv` — final table: latent_id, enrichment, p-value, dominant_organism, hit_consistency, proposed_label

Figures:
- `volcano_plot.png` — x = log2FC, y = -log10(FDR p-value), colored by significance
- `top_organism_detectors.png` — bar chart of top 10-15 labeled features
- `enrichment_histogram.png` — distribution of odds ratios across all latents

## Cost

| Component | Cost |
|-----------|------|
| Enrichment scan | $0 (scipy/numpy, ~2 min) |
| Sequence retrieval | $0 |
| BLAST (~500 queries) | $0 (NCBI public API, ~15 min) |
| **Total** | **$0** |

## Dependencies

All data present in repo. Python packages: numpy, scipy, statsmodels, matplotlib, requests (for BLAST API).

## What success looks like

A table of SAE latents with organism labels backed by BLAST evidence: "Latent 7241 is an Influenza A polymerase detector (9/10 top sequences BLAST to Influenza A, mean identity 96%)."
