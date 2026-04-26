# Experiment 5: Feature Clustering

Cluster the 32,768 SAE latents by their co-activation patterns. Do natural "modules" of features emerge? Do pathogen-specific features cluster together?

## Motivation

InterProt found that features organize by type: motif features, domain features, periodic features, etc. (Figure 3c). Bridget's SURF found that PBD families form a network where some families share many features (hubs like 14-3-3) while others are isolated (peripherals like PDZ).

We ask: do our 32k latents self-organize into coherent groups? If pathogen-specific features cluster together, it suggests the SAE has learned an organized "pathogen detection module" — a group of co-activating features that collectively represent pathogenicity. If they're scattered, pathogen detection is distributed.

## Data

- `data/sae_model/features.npy` — (20,000 x 32,768)
- Enrichment results from Experiment 1 (optional — for coloring)

## Pipeline

### Step 1: Transpose and reduce

Transpose `features.npy` to (32,768 x 20,000) — each latent is now a point in 20k-dimensional "sequence space."

PCA to 50 dimensions for computational tractability.

### Step 2: UMAP

Run UMAP on the (32,768 x 50) matrix → 2D embedding. Each point is one latent.

Parameters: `n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42`.

Use cosine distance since we care about activation *patterns* not magnitudes.

### Step 3: Cluster

Run `hdbscan.HDBSCAN(min_cluster_size=20)` on the 2D UMAP coordinates.

### Step 4: Annotate clusters

For each cluster:
- Number of latents in the cluster
- Mean pathogen enrichment score (from Experiment 1, if available; otherwise compute Fisher OR here)
- Mean activation count (how broadly do these latents fire?)
- List of latent IDs

Identify:
- "Pathogen module": cluster(s) with high mean pathogen enrichment
- "Non-pathogen module": cluster(s) with low enrichment
- "Broad features": cluster(s) where latents fire on most sequences regardless of class
- "Rare features": cluster(s) where latents fire on very few sequences

### Step 5: Output

Save to `results/feature_clustering/`:

- `latent_umap.png` — 32k points colored by cluster ID
- `latent_umap_enrichment.png` — same UMAP colored by pathogen enrichment (continuous colormap, red=pathogen, blue=non-pathogen)
- `latent_umap_activation_count.png` — same UMAP colored by number of activating sequences
- `cluster_summary.csv` — cluster_id, size, mean_enrichment, mean_activation_count
- `cluster_assignments.csv` — latent_id, cluster_id, enrichment, activation_count

## Dependencies

`pip install umap-learn hdbscan scikit-learn matplotlib`

## Cost

$0. UMAP on 32k x 50 matrix takes ~2-3 minutes.

## What success looks like

- **Organized clusters with pathogen signal**: a distinct cluster of pathogen-enriched latents, separate from non-pathogen or generic latents. "The SAE has learned a coherent pathogen detection module."
- **No pathogen clustering but other structure**: latents cluster by activation breadth (rare vs broad) or activation magnitude. Still interesting — tells us how the SAE organizes its dictionary.
- **No structure at all**: latents are uniformly distributed. Less interesting but still reportable.

## Priority

Lower than experiments 1-4. Run if time permits. The main value is a figure showing organized feature structure.
