# Experiment 4: Sequence UMAP

Project the 20,000 sequences from 32,768-dim SAE feature space to 2D. Visualize whether pathogen and non-pathogen sequences naturally separate.

## Motivation

Before looking at individual features, see the big picture: does the SAE's representation of sequences encode pathogenicity at a global level? If pathogen and non-pathogen sequences cluster separately in SAE space, the linear probe will work and the enrichment analysis will find strong features. If they don't separate, pathogenicity might be encoded in a way that's not captured by sequence-level SAE features.

This also reveals structure *beyond* pathogenicity — sub-clusters within the pathogen class could correspond to different organisms, connecting to Experiment 1.

## Data

- `data/sae_model/features.npy` — (20,000 x 32,768)
- `data/sae_model/sequence_ids.json` — row-to-sequence-id mapping
- `data/human_virus_class1_labeled.jsonl` — `source` labels (0/1)

## Pipeline

### Step 1: Load and align labels

Same as other experiments — join features to source labels via sequence_ids.

### Step 2: PCA

Reduce from 32,768 to 50 dimensions using `sklearn.decomposition.PCA`. This is for speed — UMAP on 32k dims with 20k points is slow. Also report variance explained by top 50 PCs (if most variance is in first few PCs, the SAE features are low-rank).

### Step 3: UMAP

Run `umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)` on the 50-dim PCA output.

### Step 4: Plot

- Scatter plot: 20k points, colored by pathogen label (red=pathogen, blue=non-pathogen)
- Use alpha=0.3 or so for overplotting
- Add legend and title

### Step 5: Optional — sub-cluster analysis

If clear sub-clusters are visible within the pathogen class:
- Run HDBSCAN on the 2D UMAP embedding
- For each sub-cluster: pull representative sequences, note their sequence IDs
- These sub-clusters may correspond to different organism types (connect to BLAST in Experiment 1)

## Output

Save to `results/sequence_umap/`:

- `umap_pathogen.png` — main scatter plot, colored by pathogen label
- `pca_variance.png` — scree plot of PCA explained variance
- `umap_coords.csv` — UMAP x,y coordinates + sequence_id + label (for downstream use)
- `subclusters.csv` — if HDBSCAN run: cluster assignments per sequence

## Dependencies

`pip install umap-learn scikit-learn matplotlib`

## Cost

$0. PCA + UMAP on 20k x 32k takes ~1-2 minutes.

## What success looks like

- **Clean separation**: two blobs, one red, one blue. "SAE features naturally encode pathogenicity."
- **Partial separation with sub-clusters**: pathogen blob has internal structure — sub-clusters that might correspond to different viruses. Connects to Experiment 1.
- **No separation**: pathogen and non-pathogen are mixed. Means sequence-level SAE features don't globally separate the classes, but individual features might still be enriched (Experiment 1 could still succeed).
