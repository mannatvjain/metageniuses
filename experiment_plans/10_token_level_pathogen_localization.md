# Experiment 10: Token-Level Pathogen Localization

## Purpose

For individual pathogen sequences, identify exactly *which tokens* (nucleotide subsequences) are responsible for the model's pathogen classification. Then BLAST those high-signal subsequences to identify the specific gene or genomic region the model is responding to.

This produces the hero figure of the paper: a heatmap of a single metagenomic read, colored by how strongly each token contributes to pathogen detection, with BLAST annotations pointing to "this 20-nucleotide stretch matches Norovirus RNA-dependent RNA polymerase with 97% identity."

## Why this matters

The project's central claim (see `CONTEXT.md`) is that SAEs make MetaGene-1 interpretable for pandemic surveillance. Experiments 1-5 demonstrate this at the *feature level* (which latents detect pathogens) and the *sequence level* (which sequences are classified as pathogenic). But the most compelling evidence is at the *token level*: showing practitioners exactly which part of a metagenomic read triggered the pathogen alert, grounded in a BLAST hit to a known dangerous gene.

This is the nucleotide analog of InterProt's core visualization — overlaying SAE activations on protein structures to show which residues a feature responds to (Adams et al., Section 4.1, Figure 1 — see `INTERPROT_REFERENCE.md` → "Interpretation Method" step 2). We can't overlay on 3D structure (no structure prediction for raw metagenomic reads), but we can overlay on the linear nucleotide sequence and ground with BLAST, which is arguably more actionable for biosecurity practitioners.

## Prior results this builds on

- **Exp 2 (linear probe)**: Trained a logistic regression on layer-32 SAE features → 94.6% accuracy, 0.892 MCC. The probe's coefficient vector (1 × 32,768) tells us which latents are most important for pathogen classification. Code at `experiments/linear_probe_pathogen.py`.
- **Exp 1 (organism detectors)**: Identified pathogen-enriched latents via Fisher's exact test and BLAST. Partial results in `results/organism_detectors/blast_results_partial.json` show 40 unique organisms across 14 latents (Norovirus GI/GII, Human astrovirus 1, Sapovirus, Adenovirus 41, etc.).
- **Exp 7 (layer-wise probes)**: If completed, provides probe coefficients at layers 8, 16, 24, 32 — enabling multi-layer localization.
- **MetaGene-1 BPE tokenizer**: 1024-token vocabulary on nucleotides. Tokens range from 2-letter pairs ("AA") to 50+ nt strings (see `METAGENE_REFERENCE.md` → "Tokenizer"). This means each token position in the heatmap can represent a variable number of nucleotides — the visualization must account for this.

## The core idea: per-token pathogen score

We combine two things:
1. **Per-token SAE activations**: at each token position t in a sequence, we have a sparse 32,768-dim vector `z_t` of SAE feature activations (k=64 nonzero entries per token).
2. **Probe coefficients**: the linear probe from Exp 2 learned a coefficient vector `w` (32,768-dim) where `w_j > 0` means latent j pushes toward "pathogen" and `w_j < 0` pushes toward "non-pathogen."

The **per-token pathogen score** is simply the dot product:

```
pathogen_score(t) = z_t · w = Σ_j z_t[j] * w[j]
```

This decomposes the sequence-level probe prediction into per-token contributions. Tokens with high positive scores are "why the model thinks this is a pathogen." Tokens with high negative scores are "evidence against pathogen." Tokens near zero are uninformative.

**Why this works**: The sequence-level probe prediction is `sigmoid(mean(z_t · w) + bias)` — the mean of per-token scores plus a bias term. So the per-token scores literally sum to the sequence-level prediction (up to the bias and sigmoid).

**Multi-layer extension**: If Exp 7 probe coefficients are available at layers 8, 16, 24, 32, compute per-token scores at each layer. This shows how the pathogen signal builds up through the network — at layer 8, a few tokens light up on a conserved motif; by layer 32, the signal has spread across the sequence.

## Prerequisites — what Peyton needs to produce

### 1. Per-token SAE activations (REQUIRED)

Same format as Experiment 8's prerequisite. For each layer, sparse matrix of shape (total_tokens, 32768) plus token metadata:

```
data/sae_layer{L}/
  token_activations.npz      # scipy sparse CSR, (total_tokens, 32768)
  token_metadata.jsonl        # one line per row: {sequence_id, token_index, token_id}
```

At minimum, layer 32 is required (we have the probe coefficients for layer 32 from Exp 2). Layers 8, 16, 24 are needed for the multi-layer extension.

### 2. Probe coefficients from Exp 2

Already available — retrain or load from `results/linear_probe_pathogen/`. The coefficient vector is `clf.coef_[0]` (shape: 32768). If Exp 7 is done, also load probe coefficients at layers 8, 16, 24.

### 3. Token-to-nucleotide mapping

To map token positions back to nucleotide positions in the original sequence, we need the MetaGene-1 tokenizer. The tokenizer is at `vendor/metagene-pretrain/train/minbpe/mgfm-1024/`. Each token decodes to a variable-length nucleotide string.

**However**, if the tokenizer is not easily loadable, we can use a simpler approach: the extraction metadata stores `token_id` per token. We can build a lookup table from token_id to nucleotide string from the tokenizer vocabulary, then reconstruct the nucleotide-level mapping.

**Simplest fallback**: skip nucleotide-level resolution entirely and work at the token level. The heatmap shows activation per *token*, and we annotate with the decoded nucleotide string for each token. This is sufficient for the paper.

## Implementation

Single script: `experiments/token_pathogen_localization.py`

Output directory: `results/token_localization/`

### Part A: Compute per-token pathogen scores (~2-5 min per layer)

```python
import numpy as np
import scipy.sparse as sp
import json

def compute_token_pathogen_scores(
    token_activations_path: str,
    token_metadata_path: str,
    probe_coef: np.ndarray,        # shape (32768,)
    probe_intercept: float,
) -> dict[str, list[dict]]:
    """
    Returns: {sequence_id: [{token_index, token_id, pathogen_score, top_latents}, ...]}
    """
    # Load sparse activations
    activations = sp.load_npz(token_activations_path)  # (total_tokens, 32768) CSR

    # Load metadata
    meta = []
    with open(token_metadata_path) as f:
        for line in f:
            meta.append(json.loads(line))

    # Compute pathogen scores via sparse matrix-vector multiply
    # This is efficient: only touches the k=64 nonzero entries per row
    scores = activations.dot(probe_coef)  # (total_tokens,) dense vector

    # Group by sequence
    seq_tokens = {}
    for row_idx, m in enumerate(meta):
        sid = m["sequence_id"]
        if sid not in seq_tokens:
            seq_tokens[sid] = []

        # Also record which latents contributed most to this token's score
        # (for annotation purposes)
        token_row = activations[row_idx]  # sparse (1, 32768)
        if token_row.nnz > 0:
            latent_ids = token_row.indices
            latent_vals = token_row.data
            contributions = latent_vals * probe_coef[latent_ids]
            # Top 5 contributing latents for this token
            top5_idx = np.argsort(np.abs(contributions))[-5:][::-1]
            top_latents = [
                {"latent_id": int(latent_ids[j]), "activation": float(latent_vals[j]),
                 "contribution": float(contributions[j])}
                for j in top5_idx
            ]
        else:
            top_latents = []

        seq_tokens[sid].append({
            "token_index": m["token_index"],
            "token_id": m["token_id"],
            "pathogen_score": float(scores[row_idx]),
            "top_latents": top_latents,
        })

    # Sort each sequence's tokens by token_index
    for sid in seq_tokens:
        seq_tokens[sid].sort(key=lambda x: x["token_index"])

    return seq_tokens
```

**Performance**: The sparse matrix-vector multiply `activations.dot(probe_coef)` is O(nnz) where nnz = total_tokens × 64. For 3M tokens, this is ~192M multiplications — under a second. The per-token top-latent extraction is slower (O(total_tokens × 64 × log(5))) but still <1 min.

### Part B: Select showcase sequences (~10 sec)

Select sequences that best demonstrate the localization. Criteria:

1. **High-confidence pathogens**: Select the top 50 pathogen sequences by sequence-level probe probability (highest `sigmoid(mean(pathogen_score) + intercept)`). These are the sequences the model is most sure about.

2. **Diverse organisms**: If Exp 1 BLAST results exist, preferentially select sequences that BLAST to different organisms (at least one Norovirus, one Astrovirus, one Adenovirus, etc.) to show the model detects different pathogens via different token-level patterns.

3. **High-contrast sequences**: Among the top 50, prefer sequences with high variance in per-token pathogen score — this means some tokens are very informative and others aren't, making the heatmap visually compelling. Avoid sequences where all tokens have similar scores (those are "whole-sequence" detections, less interesting for localization).

4. **Also select 10 non-pathogen sequences** with the highest-magnitude negative scores — these show what the "non-pathogen" signal looks like at the token level.

### Part C: BLAST high-signal subsequences (~15-30 min, NCBI API)

For each showcase pathogen sequence:

1. Find the contiguous region of tokens with the highest cumulative pathogen score. This is the "hotspot" — the nucleotide region most responsible for pathogen classification.

   ```python
   def find_hotspot(token_scores: list[float], min_length: int = 3) -> tuple[int, int]:
       """Find the contiguous window of tokens with maximum cumulative score.
       Uses Kadane's algorithm variant with minimum window length."""
       best_sum = float("-inf")
       best_start = best_end = 0
       for start in range(len(token_scores)):
           cumsum = 0
           for end in range(start, len(token_scores)):
               cumsum += token_scores[end]
               if end - start + 1 >= min_length and cumsum > best_sum:
                   best_sum = cumsum
                   best_start, best_end = start, end
       return best_start, best_end
   ```

2. Extract the nucleotide subsequence corresponding to the hotspot tokens. If the tokenizer is available, decode `token_id`s to nucleotides. Otherwise, look up the original sequence from `data/human_virus_class1_labeled.jsonl` and use token position offsets.

3. BLAST the hotspot subsequence against NCBI nt database. Use the same BLAST API mechanics as Experiment 1 (see `experiment_plans/01_organism_detectors.md` → Part C for full API details: POST to `blast.ncbi.nlm.nih.gov`, poll for results, parse JSON2 format).

4. Record: organism, gene, percent identity, e-value. This grounds the localization: "the model's pathogen signal concentrates on tokens 12-25, which BLAST to Norovirus GII RNA-dependent RNA polymerase at 97% identity."

**Rate limiting**: Same as Exp 1 — max ~3 requests/second, `TOOL=metageniuses`, `EMAIL=mannat.v.jain@columbia.edu`, exponential backoff on errors. ~50-60 BLAST queries total (manageable).

**Checkpoint**: Save results to `blast_hotspots_partial.json` after each sequence. Resume by checking existing results.

### Part D: Figures

**D1. Single-sequence heatmap** (`heatmap_example_{sequence_id}.png`) — **THE HERO FIGURE**

For a single well-chosen pathogen sequence:
- Horizontal bar: each cell is one token, colored by pathogen score (diverging colormap: blue = non-pathogen, white = neutral, red = pathogen)
- Below the bar: nucleotide string, with each token's decoded nucleotides grouped and labeled
- Above the bar: for the hotspot region, annotate with BLAST result: "Norovirus GII RdRp (97% identity, e=2.1e-45)"
- Below: latent annotations for the top-3 tokens: "Token 15: L7241 (+0.23), L12803 (+0.18)" showing which SAE features are driving the score
- Size: (16, 4), dpi 200

Generate this for 5-10 cherry-picked sequences that best illustrate the story (different organisms, clear hotspots).

**D2. Multi-layer localization** (`multilayer_heatmap_{sequence_id}.png`)

For a single pathogen sequence, 4 stacked heatmaps (one per layer):
- Same token positions across all 4 rows
- Layer 8 at top, layer 32 at bottom
- Shows how the pathogen signal builds: at layer 8, a narrow region lights up (motif-level); by layer 32, the signal has spread
- Annotate layer 8 hotspot with BLAST hit (the motif), annotate layer 32 with the sequence-level organism label
- Size: (16, 8), dpi 200

This requires Exp 7 probe coefficients at all 4 layers.

**D3. Hotspot length distribution** (`hotspot_lengths.png`)

Histogram of hotspot lengths (in tokens) across the top 50 pathogen sequences. Shows whether the pathogen signal is typically concentrated (short hotspot = specific gene region) or diffuse (long hotspot = whole-sequence property). Size: (8, 5), dpi 150.

**D4. Pathogen score profiles, overlaid** (`score_profiles.png`)

Line plot for 10 pathogen sequences overlaid:
- x-axis: token position (0 to seq_length)
- y-axis: pathogen score
- Each sequence is a semi-transparent line
- Shows common patterns: do pathogen signals tend to concentrate at the start, end, or middle of reads?
- Size: (12, 6), dpi 150

**D5. Token contribution breakdown** (`token_contributions_{sequence_id}.png`)

For a single cherry-picked sequence, stacked bar chart:
- x-axis: token position
- y-axis: pathogen score, decomposed into contributions from individual latents
- Each stack segment is one latent's contribution (latent_activation × probe_coefficient)
- Color by latent (top 5 contributing latents get distinct colors, rest are gray)
- Shows that specific tokens are driven by specific latents: "token 15's pathogen score comes primarily from latent 7241 (Norovirus detector from Exp 1)"
- Size: (16, 6), dpi 150

### Part E: Summary and outputs

**`hotspot_blast_results.json`**:
```json
{
    "human_virus_class1_4523": {
        "sequence_level_probe_prob": 0.997,
        "hotspot_start_token": 12,
        "hotspot_end_token": 25,
        "hotspot_nucleotides": "ATGCGTACCGATCCC...",
        "hotspot_length_tokens": 14,
        "hotspot_cumulative_score": 3.45,
        "blast_hit": {
            "organism": "Norovirus GII",
            "gene": "RNA-dependent RNA polymerase",
            "accession": "MK789234.1",
            "percent_identity": 97.2,
            "e_value": 2.1e-45,
            "query_coverage": 100
        },
        "top_contributing_latents": [
            {"latent_id": 7241, "total_contribution": 1.23, "organism_label": "Norovirus GII"},
            {"latent_id": 12803, "total_contribution": 0.87, "organism_label": "..."}
        ]
    },
    ...
}
```

**`token_scores_all.npz`**: Sparse matrix (n_sequences × max_seq_length) of per-token pathogen scores. Useful for downstream analyses.

**`summary.json`**:
```json
{
    "experiment": "token_pathogen_localization",
    "layers_analyzed": [32],
    "n_showcase_pathogens": 50,
    "n_showcase_nonpathogens": 10,
    "n_blast_queries": ...,
    "mean_hotspot_length_tokens": ...,
    "median_hotspot_length_tokens": ...,
    "hotspots_with_blast_hits": ...,
    "unique_organisms_in_hotspots": ...,
    "conclusion": "..."
}
```

## Dependencies

```
pip install numpy scipy matplotlib requests
```

Optional: `pandas` for tabular outputs.

## Runtime estimate

| Step | Time | Compute |
|------|------|---------|
| A: Per-token scores (per layer) | ~2-5 min | Local (sparse matmul) |
| B: Select showcase sequences | ~10 sec | Local |
| C: BLAST hotspots | ~15-30 min | NCBI API |
| D: Figures | ~2 min | Local (matplotlib) |
| **Total** | **~20-40 min** | **$0** |

Multi-layer version (4 layers): ~30-50 min total.

## What success looks like

1. **Clear hotspots in pathogen sequences**: The pathogen score is not uniform across the sequence — it concentrates on a specific region of 5-20 tokens. This means the model is responding to a specific genomic feature, not just overall sequence composition.
2. **Hotspots BLAST to known pathogen genes**: "tokens 12-25 match Norovirus RdRp at 97% identity." This is the strongest possible evidence that the SAE has learned biologically meaningful features.
3. **Different organisms have different hotspot patterns**: Norovirus sequences light up at the polymerase, Adenovirus sequences light up at the capsid gene, etc. The model has learned organism-specific detection strategies.
4. **Multi-layer story is coherent**: Layer 8 shows a narrow motif-level signal, layer 32 shows a broad sequence-level signal. The mechanistic story: "MetaGene-1 first recognizes a conserved viral motif, then propagates that recognition across the sequence to make a global pathogen call."

## What failure looks like

- **Uniform scores across tokens**: The pathogen signal is a whole-sequence property (GC content, k-mer distribution), not localizable to specific regions. The heatmap is all one color. Still informative — it means pathogen detection is holistic, not gene-specific. Show the heatmap anyway as evidence.
- **Hotspots don't BLAST to anything**: The model is responding to regions that don't match GenBank. Could be novel viral sequences, non-coding regions, or sequencing artifacts. Label as "uncharacterized signal" — still publishable.
- **Hotspots are at sequence boundaries**: The model might be using padding/separator artifacts from context stuffing (see `METAGENE_REFERENCE.md` → "Context Stuffing"). If this happens, mask the first and last 2 tokens and re-analyze.

## Relationship to other experiments

- **Exp 1 (organism detectors)**: Exp 1 says "latent 7241 is a Norovirus detector." Exp 10 says "latent 7241 fires on tokens 12-25 of this specific sequence, and those tokens are the Norovirus RdRp." Exp 10 provides the *spatial evidence* for Exp 1's *statistical claim*.
- **Exp 7 (layer-wise probes)**: Provides the probe coefficients at multiple layers needed for the multi-layer localization figure.
- **Exp 8 (activation patterns)**: Exp 8 classifies latents as "motif" vs "whole." Exp 10 shows what a motif-type latent actually looks like on a real sequence: it fires on 5-20 tokens in a specific region, not the whole read.

## What NOT to edit

- `src/metageniuses/sae/analyze.py` — Peyton's pipeline
- `tests/sae/test_analyze.py` — Peyton's tests
- `pyproject.toml` — shared config

## Instructions for Codex

Before writing the script:

1. **Read `experiments/linear_probe_pathogen.py`** — this is where the probe coefficients come from. After fitting `LogisticRegressionCV`, the coefficients are `clf.coef_[0]` (shape 32768) and the intercept is `clf.intercept_[0]`. You will need to either retrain the probe in this script (copy the same code) or load a saved model. Since sklearn models can be saved with `joblib.dump(clf, path)`, consider adding a save step to the existing probe script or retraining here.

2. **Read `src/metageniuses/sae/encode_features.py`** — lines 54-61 show how per-token TopK is applied at inference. The same logic produced the token activations that Peyton will save.

3. **Read `src/metageniuses/extraction/contracts.py`** — `iter_layer_batches()` yields `(vectors, metadata)`. Metadata has `sequence_id`, `token_index`, `token_id`, `layer`. Token order within a sequence is determined by `token_index`.

4. **Read `experiment_plans/01_organism_detectors.md` Part C** — BLAST API mechanics are documented there in full (submit, poll, parse, rate-limit, checkpoint/resume). Re-use the same BLAST utility code — consider factoring it into a shared module `experiments/blast_utils.py` if it doesn't already exist.

5. **Read `data/human_virus_class1_labeled.jsonl`** — the `sequence` field has the nucleotide string for each sequence. You'll need this to extract the hotspot nucleotides for BLAST.

6. **The tokenizer**: If the MetaGene-1 tokenizer at `vendor/metagene-pretrain/train/minbpe/mgfm-1024/` can be loaded, use it to map token_id → nucleotide string. If not, the extraction metadata already stores token_id per token, and you can reconstruct the nucleotide mapping by aligning the tokenized sequence against the original nucleotide string from the labeled JSONL. As a last resort, skip nucleotide-level resolution and work at the token level — the heatmap is still informative.

7. **The script should have clear `--mode` flags**:
   - `--mode scores` — compute per-token scores only (fast, no network)
   - `--mode blast` — BLAST the hotspots (requires network, slow)
   - `--mode figures` — generate figures from saved scores + BLAST results
   - `--mode all` — run everything end-to-end

8. **Memory**: The sparse activation matrix for one layer is ~1.5 GB. The probe coefficient vector is 128 KB. The per-token score vector (dense, float32) is ~12 MB for 3M tokens. This all fits in memory easily.

9. **Matplotlib heatmap rendering**: For the hero figure (D1), use `matplotlib.patches.Rectangle` or `imshow` on a (1 × n_tokens) array to create the heatmap strip. Use `matplotlib.colors.TwoSlopeNorm` with `vcenter=0` for the diverging colormap so zero is exactly white. Use `RdBu_r` colormap (red=pathogen, blue=non-pathogen).
