# Experiment 3: SAE Health Check

Descriptive statistics on the trained SAE. How many features are alive, how active are they, is sparsity behaving as expected?

## Motivation

Before interpreting features, verify the SAE is well-trained. InterProt reported dead feature percentages, activation distributions, and the effect of hyperparameters (Figure 8). We need the equivalent for our MetaGene-1 SAE as a sanity check and as a free figure for the paper.

## Data

- `data/sae_model/features.npy` — (20,000 x 32,768) sequence-level SAE activations
- `data/sae_model/sae_config.json` — hyperparameters (d_model=4096, expansion=8, k=64, layer 32)
- `data/sae_model/sae_training_curves.png` — training loss + dead feature % over time

## What to compute

### Per-latent statistics (32,768 latents)
- **Dead vs alive**: does the latent ever activate (max > 0) on any sequence?
- **Activation count**: how many of the 20,000 sequences activate each latent?
- **Max activation**: highest activation value observed for each latent
- **Mean activation** (over sequences where it's active): average strength when it fires
- **Activation fraction**: what fraction of sequences activate it?

### Per-sequence statistics (20,000 sequences)
- **Active features per sequence**: how many of the 32,768 latents are nonzero? Should be approximately k=64 (TopK).
- **Total activation mass per sequence**: sum of all feature activations

### Global statistics
- Dead count, alive count, dead percentage
- Mean/median/std of activation counts across latents
- Sparsity: fraction of nonzero entries in the full matrix (already know: 2.72%)

## Output

Save to `results/sae_health_check/`:

- `stats.json` — summary statistics (dead count, alive count, sparsity, mean active features per sequence, etc.)
- `latent_stats.csv` — per-latent: latent_id, is_dead, activation_count, max_activation, mean_activation
- `sequences_per_latent.png` — histogram: how many sequences activate each latent (x: num sequences, y: num latents). Log scale on y-axis. Expect long tail.
- `max_activation_distribution.png` — histogram of max activation values across latents
- `active_features_per_sequence.png` — histogram of number of active features per sequence (expect peak near k=64)
- `summary_table.png` — formatted summary table suitable for paper supplementary

## Comparison to InterProt

| Metric | InterProt (ESM-2) | Ours (MetaGene-1) |
|--------|------------------|-------------------|
| d_model | 1280 | 4096 |
| Expansion | varies (2x-4x) | 8x |
| k | 64 | 64 |
| Total latents | 4096-8192 | 32,768 |
| Dead % | varies by layer | ? (compute) |

Note: we already know 803/32,768 = 2.4% dead — very healthy. InterProt's dead feature % varied by layer and spiked then recovered during training (similar to our training curves).

## Cost

$0. Pure numpy, runs in under a minute.
