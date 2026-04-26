# Activation Pattern Classification

Categorize every SAE latent by how it fires across sequences. No labels or bio expertise needed — purely structural analysis.

## Motivation

The distribution of activation patterns is itself a result. It tells us what *kind* of representations MetaGene-1 learned: does it mostly recognize local motifs (gene fragments, binding sites)? Global properties (organism identity, GC content)? Periodic structure (codon usage)? This is a free table/figure for the writeup.

## Categories

Adapted from InterProt's taxonomy, with thresholds recalibrated for nucleotide sequences (their thresholds were tuned for protein residue-level features).

| Category | Criteria | Biological intuition |
|----------|----------|---------------------|
| Dead | Never activates on any test sequence | Learned nothing useful |
| Not Enough Data | <5 sequences activate it | Can't characterize reliably |
| Point | Median activation region length = 1 token | Single-position signals: start codons, splice sites |
| Short Motif (1-30nt) | Contiguous region, median 1-30 tokens, <80% coverage | Promoters, primer sites, restriction sites, short conserved regions |
| Medium Motif (30-100nt) | Contiguous region, median 30-100 tokens, <80% coverage | Gene fragments, regulatory regions |
| Long Motif (100-500nt) | Contiguous region, median 100-500 tokens, <80% coverage | Larger gene regions, operons |
| Periodic | >50% same inter-activation distance, >10 activation regions, short contigs | Codon structure (period 3), tandem repeats |
| Whole | >80% mean activation coverage | Global properties: GC content, organism identity, codon usage bias |
| Other | Doesn't fit above | Uncategorized |

Note: thresholds above are starting points. May need adjustment after looking at actual activation distributions — if everything clusters at one boundary, shift it.

## Pipeline

### Step 1: Run all test sequences through the SAE

For each sequence, record per-latent activations at each token position.

### Step 2: Per-latent statistics

For each latent:
- How many sequences activate it (activation count)
- Median length of contiguous activation regions
- Mean fraction of sequence covered
- Inter-activation distances (for periodicity check)

### Step 3: Classify

Apply the category rules from the table above. Output a label per latent.

### Step 4: Report

- Distribution histogram: how many latents in each category
- Breakdown by layer (if we trained SAEs on multiple layers)
- Compare to InterProt's distribution on ESM-2 — are metagenomic features distributed differently than protein features?

## Cost Estimate

| Component | Cost |
|-----------|------|
| Everything | $0 (local compute on existing activations) |

## Output

`results/activation_pattern_classification.csv`: one row per latent with columns (latent_id, category, num_activating_sequences, median_region_length, mean_coverage, periodicity_score).

## Dependencies

- Trained SAE weights
- Test sequences (class 2 JSONL — held-out set)
- Extraction outputs (activation shards)
