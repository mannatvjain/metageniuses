# Experiment 8: Activation Pattern Classification (Multi-Layer)

## Purpose

Categorize every SAE latent at every layer by *how* it fires across tokens within a sequence: point, short motif, medium motif, long motif, periodic, whole-sequence, or dead. Then compare the distribution of pattern types across layers 8, 16, 24, and 32.

This is a direct replication of InterProt's activation pattern taxonomy (Adams et al., ICML 2025 — Table 2, Section 4.1, Appendix A.3). InterProt classified latents for ESM-2 (a protein language model) into categories based on the spatial pattern of their activations along the protein sequence. They found that earlier layers have more long contiguous activation features (motifs/domains), while later layers have shorter, more specific activations. We test whether MetaGene-1 — an autoregressive model on nucleotide sequences, not a masked model on amino acids — follows the same pattern.

**This is a free table and figure for the paper.** It requires no labels, no BLAST, no external data — just per-token SAE activations and simple statistics.

## Background: InterProt's taxonomy

From `INTERPROT_REFERENCE.md` → "Feature Classification Schemes" → "By Activation Pattern":

| Category | InterProt Criteria (protein, ESM-2) | Our Adapted Criteria (nucleotide, MetaGene-1) |
|----------|-------------------------------------|----------------------------------------------|
| Dead | Never activates on any test sequence | Same |
| Not Enough Data | <5 sequences activate it | Same |
| Periodic | >50% same inter-activation distance, >10 activation regions, short contigs | Same logic but thresholds may need tuning (see below) |
| Point | Median length of highest activating region = 1 token | Same |
| Short Motif | Contiguous region, median 1-20 residues, <80% coverage | Median 1-30 tokens, <80% coverage |
| Medium Motif | Contiguous region, median 20-50 residues, <80% coverage | Median 30-100 tokens, <80% coverage |
| Long Motif | Contiguous region, median 50-300 residues, <80% coverage | Median 100-300 tokens, <80% coverage |
| Whole | >80% mean activation coverage | Same |
| Other | Doesn't fit above | Same |

**Why different thresholds**: MetaGene-1 uses BPE tokenization (vocab=1024) on nucleotides, so one token can span multiple base pairs (see `METAGENE_REFERENCE.md` → "Tokenizer" — tokens range from 2-letter pairs like "AA" to 50+ nt strings). InterProt's ESM-2 uses single amino acid tokens. Our "short motif" at 30 tokens might span ~60-150 nucleotides depending on token lengths. The thresholds in `experiment_plans/activation_pattern_classification.md` (the earlier, pre-token-data spec) suggested 1-30 / 30-100 / 100-500 for short/medium/long — we use those with a 300-token cap on long motif since MetaGene-1 sequences are max 512 tokens.

**Key InterProt finding to test against** (from `INTERPROT_REFERENCE.md` → "Key Hyperparameter Effects" → "Layer choice"):
- "Long contiguous activation features (motif/domain) more common in earlier layers"
- "Later layers have shorter, more specific activations (specializing for final logit computation)"
- "Family-specific features peak in early-to-mid layers, then decline"

If MetaGene-1 follows this same pattern, we'd expect layer 8 to be motif-heavy and layer 32 to be whole-sequence-heavy.

## Prerequisites — what Peyton needs to produce

This experiment requires **per-token SAE activations** (not mean-pooled). See `future_experiments/token_level_activations.md` for full context on why mean-pooled data is insufficient.

### Required data format

For each layer L in [8, 16, 24, 32], per-token sparse SAE activations stored in one of these formats:

**Option A (preferred): Sparse NPZ per layer**

```
data/sae_layer{L}/token_activations/
  activations.npz          # scipy.sparse CSR or COO matrix, shape (total_tokens, 32768)
  token_metadata.jsonl     # one line per token row, with fields:
                           #   sequence_id, token_index, token_id
```

To produce this, modify `src/metageniuses/sae/encode_features.py` as follows. The current code (lines 54-69) runs `sae.encode(x)` → `topk` → accumulates into `seq_feature_sum`. Instead:

```python
import scipy.sparse as sp

# Instead of seq_feature_sum accumulation, collect sparse rows:
all_rows = []
all_cols = []
all_vals = []
all_meta = []
global_row = 0

for batch_vecs, batch_meta in iter_layer_batches(artifact_root, layer, batch_size=batch_size):
    x = torch.tensor(batch_vecs, dtype=torch.float32, device=device)
    x = x / activation_scale

    with torch.no_grad():
        z = sae.encode(x)
        topk = torch.topk(z, sae.k, dim=-1)
        # topk.indices: (batch, k) — which latents are active
        # topk.values:  (batch, k) — activation values

    indices = topk.indices.cpu().numpy()  # (batch, k)
    values = topk.values.cpu().numpy()    # (batch, k)

    for i in range(len(batch_meta)):
        for j in range(sae.k):
            all_rows.append(global_row)
            all_cols.append(int(indices[i, j]))
            all_vals.append(float(values[i, j]))
        all_meta.append({
            "sequence_id": batch_meta[i]["sequence_id"],
            "token_index": batch_meta[i]["token_index"],
            "token_id": batch_meta[i]["token_id"],
        })
        global_row += 1

# Save as sparse COO matrix
sparse_mat = sp.coo_matrix(
    (all_vals, (all_rows, all_cols)),
    shape=(global_row, sae.d_sae),
    dtype=np.float32,
)
sp.save_npz(output_dir / "token_activations.npz", sparse_mat.tocsr())
```

**Storage estimate**: 20k sequences × ~150 tokens × 64 active features = ~192M nonzero entries. As CSR with float32 values and int32 indices: ~192M × (4+4) bytes ≈ 1.5 GB per layer, ~6 GB total.

**Option B (fallback): The existing extraction format + on-the-fly encoding**

If sparse NPZ is not pre-computed, the script can compute per-token activations on-the-fly by loading the raw extraction activations via `iter_layer_batches()` and running the SAE encoder. This requires the SAE checkpoint and is slower but avoids needing Peyton to produce a new file format. The script should support both options.

### Also needed: mean-pooled features (for cross-referencing with Exp 1 enrichment)

The mean-pooled `features.npy` at each layer (same as Exp 7 prerequisite) allows cross-referencing: "is this whole-sequence feature also pathogen-enriched?"

## Expected file layout

```
data/sae_layer8/
  sae_final.pt
  features.npy                          # (20000, 32768) mean-pooled
  sequence_ids.json
  token_activations.npz                 # sparse (total_tokens_layer8, 32768)
  token_metadata.jsonl                  # sequence_id, token_index, token_id per row

data/sae_layer16/
  (same structure)

data/sae_layer24/
  (same structure)

data/sae_model/                         # layer 32
  sae_final.pt
  features.npy
  sequence_ids.json
  token_activations.npz                 # NEW — Peyton needs to produce this
  token_metadata.jsonl                  # NEW
```

## Implementation

Single script: `experiments/activation_patterns.py`

Output directory: `results/activation_patterns/`

### Part A: Per-latent activation statistics (~5-10 min per layer)

For each layer L, for each latent i (0 to 32,767):

1. **Load the sparse activation matrix** for layer L. Each row is one token, each column is one latent.

2. **Group tokens by sequence**. Use `token_metadata.jsonl` to map row indices back to `(sequence_id, token_index)` pairs. Build a dict: `sequence_tokens[sequence_id] = [(token_index, activation_value), ...]` for latent i, sorted by token_index.

3. **For each sequence that activates latent i** (activation > 0 at any token):

   a. **Find contiguous activation regions**. A "region" is a maximal run of consecutive token positions where the latent fires. Gap tolerance: allow 1-token gaps (a single inactive token between two active tokens counts as part of the same region). This accounts for BPE tokenization artifacts where a latent might skip a short bridging token.

   ```python
   def find_regions(token_positions: list[int], gap_tolerance: int = 1) -> list[tuple[int, int]]:
       """Return list of (start, end) token position pairs for contiguous regions."""
       if not token_positions:
           return []
       regions = []
       start = token_positions[0]
       prev = token_positions[0]
       for pos in token_positions[1:]:
           if pos - prev > gap_tolerance + 1:
               regions.append((start, prev))
               start = pos
           prev = pos
       regions.append((start, prev))
       return regions
   ```

   b. **Compute region lengths** (in tokens): `end - start + 1` for each region.

   c. **Compute coverage**: (number of active tokens) / (total tokens in this sequence).

   d. **Record the length of the highest-activating region** (the region containing the token with the max activation value for this latent in this sequence).

4. **Aggregate across sequences** to get per-latent statistics:

   - `n_activating_sequences`: how many sequences have at least one active token for this latent
   - `median_highest_region_length`: median (across sequences) of the length of the highest-activating region
   - `mean_coverage`: mean (across sequences) of the coverage fraction
   - `mean_n_regions`: mean number of distinct activation regions per sequence
   - `inter_region_distances`: for sequences with ≥3 regions, compute pairwise distances between region starts. If >50% of distances are equal (within ±1 token), flag as periodic.
   - `periodicity_score`: fraction of sequences where the latent shows periodic behavior
   - `max_activation_value`: global max activation across all tokens and sequences

### Part B: Classification (~1 sec)

Apply the category rules to each latent:

```python
def classify_latent(stats: dict) -> str:
    if stats["n_activating_sequences"] == 0:
        return "dead"
    if stats["n_activating_sequences"] < 5:
        return "not_enough_data"
    if stats["periodicity_score"] > 0.5 and stats["mean_n_regions"] > 10:
        return "periodic"

    med_len = stats["median_highest_region_length"]
    cov = stats["mean_coverage"]

    if cov > 0.8:
        return "whole"
    if med_len == 1:
        return "point"
    if med_len <= 30:
        return "short_motif"
    if med_len <= 100:
        return "medium_motif"
    if med_len <= 300:
        return "long_motif"
    return "other"
```

**Threshold tuning note**: These thresholds are starting points adapted from InterProt (see table above). After the first run, check the distribution — if >80% of latents fall into one category, the thresholds need adjustment. Print a warning if any category has <1% of latents or >60% of latents.

### Part C: Cross-reference with pathogen enrichment (optional but high-value)

If Experiment 1 results exist (`results/organism_detectors/enrichment_results.csv`), load the enrichment data and tag each latent with its pathogen enrichment status. This enables the analysis: "are pathogen-specific features more likely to be whole-sequence or motif-type?"

```python
# Load Exp 1 enrichment results
enrichment = pd.read_csv("results/organism_detectors/enrichment_results.csv")
# Merge with activation pattern results
merged = patterns_df.merge(enrichment[["latent_id", "fisher_or", "is_pathogen_enriched", "best_f1"]], on="latent_id", how="left")
```

### Part D: Figures

**D1. Pattern distribution by layer** (`pattern_distribution.png`) — **THE MAIN FIGURE**

Stacked bar chart or grouped bar chart:
- x-axis: Layer (8, 16, 24, 32)
- y-axis: Percentage of latents in each category
- Colors: one per category (dead=gray, point=red, short_motif=orange, medium_motif=yellow, long_motif=green, periodic=purple, whole=blue, other=white, not_enough_data=light gray)
- This is the direct analog of InterProt Table 2
- Size: (10, 6), dpi 150

**D2. Category count table** (`pattern_counts.csv`)

| layer | dead | not_enough_data | point | short_motif | medium_motif | long_motif | periodic | whole | other | total |
|-------|------|-----------------|-------|-------------|-------------|-----------|----------|-------|-------|-------|
| 8 | ... | ... | ... | ... | ... | ... | ... | ... | ... | 32768 |
| 16 | ... | ... | ... | ... | ... | ... | ... | ... | ... | 32768 |
| 24 | ... | ... | ... | ... | ... | ... | ... | ... | ... | 32768 |
| 32 | ... | ... | ... | ... | ... | ... | ... | ... | ... | 32768 |

**D3. Pattern vs enrichment** (`pattern_vs_enrichment.png`) — if Exp 1 data available

For layer 32 (where we have enrichment data), grouped bar chart:
- x-axis: Pattern category
- Two bars per category: pathogen-enriched count vs not-enriched count
- Shows whether organism detectors are concentrated in a specific pattern type
- Size: (10, 6), dpi 150

**D4. Activation region length distributions** (`region_length_histograms.png`)

4 subplots (one per layer). Each is a histogram of `median_highest_region_length` across all non-dead latents. Log x-axis since lengths span 1 to 300+. Vertical dashed lines at category boundaries (1, 30, 100, 300). Size: (14, 10), dpi 150.

**D5. Coverage vs region length scatter** (`coverage_vs_length.png`)

4 subplots (one per layer). Each is a scatter of mean_coverage (y) vs median_highest_region_length (x) for all non-dead latents. Color by pattern category. This shows the 2D decision boundary of the classification rules. Size: (14, 10), dpi 150.

### Part E: Summary

**`summary.json`**:

```json
{
    "experiment": "activation_pattern_classification",
    "layers": [8, 16, 24, 32],
    "latents_per_layer": 32768,
    "thresholds": {
        "short_motif_max": 30,
        "medium_motif_max": 100,
        "long_motif_max": 300,
        "whole_coverage_min": 0.8,
        "periodic_score_min": 0.5,
        "periodic_min_regions": 10,
        "min_sequences": 5,
        "gap_tolerance": 1
    },
    "results_by_layer": {
        "8": { "dead": ..., "point": ..., ... },
        "16": { ... },
        "24": { ... },
        "32": { ... }
    },
    "interprot_comparison_notes": "..."
}
```

**`latent_patterns.csv`** — one row per (layer, latent) with all computed statistics:

```
layer, latent_id, category, n_activating_sequences, median_highest_region_length,
mean_coverage, mean_n_regions, periodicity_score, max_activation_value,
is_pathogen_enriched (if available), fisher_or (if available)
```

## Dependencies

```
pip install numpy scipy matplotlib pandas
```

Optional: `scikit-learn` if adding the enrichment cross-reference.

## Runtime estimate

| Step | Time | Compute |
|------|------|---------|
| A: Per-latent stats (per layer) | ~5-10 min | Local (sparse matrix ops) |
| A: All 4 layers | ~20-40 min | Local, can parallelize across layers |
| B: Classification | ~1 sec | Local |
| C: Enrichment cross-reference | ~10 sec | Local |
| D: Figures | ~1 min | Local (matplotlib) |
| **Total** | **~25-45 min** | **$0** |

**Performance note**: The bottleneck is iterating over 32,768 latents and computing region statistics per sequence. For efficiency:
- Load the sparse matrix once per layer into memory (CSR format for fast row slicing)
- For each latent, use column slicing (`mat[:, latent_id]`) to get all tokens that fire for that latent
- Group by sequence using the metadata index
- Consider processing in chunks of 1000 latents with progress bars

## What success looks like

1. **Clear layer-dependent pattern shift**: early layers are motif-heavy, late layers are whole-sequence-heavy. Matches InterProt's finding.
2. **Pathogen-enriched features cluster in specific pattern types**: e.g., organism detectors at layer 32 are mostly "whole" (the model's global assessment of the sequence), while layer 8 pathogen features are "short motif" (conserved viral gene fragments).
3. **Periodic features exist**: codon usage bias or tandem repeat patterns. These would be unique to nucleotide sequences — InterProt wouldn't see these in protein sequences.

## What failure looks like

- **Distribution is uniform across layers**: MetaGene-1 doesn't develop hierarchical representations. Still publishable as a negative result — "unlike ESM-2, MetaGene-1's autoregressive architecture produces uniform activation patterns across depth."
- **Almost everything is "whole"**: mean-pooling artifacts in the SAE training, or the sequences are too short (~150 tokens avg) for motif-type patterns to emerge. If this happens, try lowering the "whole" threshold from 80% to 70% coverage and see if motif patterns emerge.

## What NOT to edit

- `src/metageniuses/sae/analyze.py` — Peyton's pipeline
- `tests/sae/test_analyze.py` — Peyton's tests
- `pyproject.toml` — shared config

## Instructions for Codex

Before writing this script:

1. Read `src/metageniuses/sae/encode_features.py` — understand the SAE encoding pipeline. Lines 54-61 show how TopK is applied at inference (per-token, not batch).
2. Read `src/metageniuses/extraction/contracts.py` — `iter_layer_batches()` yields `(vectors: list[list[float]], metadata: list[dict])`. Each metadata dict contains `sequence_id` (str), `token_index` (int), `token_id` (int), `layer` (int). See `src/metageniuses/extraction/extractor.py:198-203` for where this metadata is written.
3. Read `experiment_plans/activation_pattern_classification.md` — the original spec (pre-token-data). This new plan supersedes it with multi-layer support and concrete implementation details.
4. The script MUST support two modes:
   - **Mode 1 (fast)**: Load pre-computed `token_activations.npz` + `token_metadata.jsonl`
   - **Mode 2 (fallback)**: Load raw activations via `iter_layer_batches()` + SAE checkpoint, compute per-token activations on the fly. This is for when the sparse NPZ hasn't been pre-computed. Use `--sae_checkpoint` and `--artifact_root` flags for this mode.
5. Use `scipy.sparse` for all sparse matrix operations. CSR format for efficient row/column slicing. Do NOT densify the full matrix — it would be 32768 × total_tokens × 4 bytes ≈ 360 GB.
6. Print progress every 1000 latents: `"Layer 8: classified 1000/32768 latents (14 dead, 203 point, 487 short_motif, ...)"`.
7. The `--layers` flag should accept a comma-separated list (default: "8,16,24,32"). Process layers sequentially to manage memory.
8. Write `latent_patterns.csv` incrementally (append after each layer) so partial results are available if the script is interrupted.
