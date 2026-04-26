"""
Experiment 5: Feature Clustering
Cluster 32,768 SAE latents by co-activation patterns across 20,000 sequences.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import fisher_exact
from sklearn.decomposition import PCA

import umap
import hdbscan
from _shared import REPO_ROOT, resolve_sae_dir, write_json

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = REPO_ROOT / "data"
SAE_DIR = resolve_sae_dir()
LABELED_PATH = DATA_DIR / "human_virus_class1_labeled.jsonl"

OUT_DIR = REPO_ROOT / "results" / "feature_clustering"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: Load features ───────────────────────────────────────────────────
print("Loading features.npy ...")
features = np.load(SAE_DIR / "features.npy")  # (20000, 32768)
print(f"  Shape: {features.shape}, dtype: {features.dtype}")

# Load sequence IDs
with open(SAE_DIR / "sequence_ids.json") as f:
    sequence_ids = json.load(f)
print(f"  Sequence IDs: {len(sequence_ids)}")

# ── Step 2: Build label mapping ─────────────────────────────────────────────
print("Loading labels from human_virus_class1_labeled.jsonl ...")
seq_to_label = {}
with open(LABELED_PATH) as f:
    for line in f:
        obj = json.loads(line)
        seq_to_label[obj["sequence_id"]] = int(obj["source"])  # 0=non-pathogen, 1=pathogen

# Build per-sequence label array aligned with features matrix
labels = np.array([seq_to_label.get(sid, -1) for sid in sequence_ids], dtype=np.int8)
n_labeled = np.sum(labels >= 0)
n_pathogen = np.sum(labels == 1)
n_nonpathogen = np.sum(labels == 0)
print(f"  Labeled: {n_labeled}/{len(sequence_ids)} (pathogen={n_pathogen}, non-pathogen={n_nonpathogen})")

# ── Step 3: Transpose to latent-space ────────────────────────────────────────
print("Transposing to (latents x sequences) ...")
latent_matrix = features.T  # (32768, 20000)
n_latents = latent_matrix.shape[0]
print(f"  Latent matrix shape: {latent_matrix.shape}")

# ── Step 4: Filter dead latents ─────────────────────────────────────────────
print("Filtering dead latents (max activation = 0) ...")
max_act = latent_matrix.max(axis=1)
alive_mask = max_act > 0
alive_indices = np.where(alive_mask)[0]
n_dead = n_latents - len(alive_indices)
print(f"  Dead latents: {n_dead}, alive: {len(alive_indices)}")

latent_alive = latent_matrix[alive_mask]  # (n_alive, 20000)

# ── Step 5: PCA to 50 dims ──────────────────────────────────────────────────
print("Running PCA to 50 dims ...")
pca = PCA(n_components=50, random_state=42)
latent_pca = pca.fit_transform(latent_alive)
print(f"  PCA output: {latent_pca.shape}")
print(f"  Explained variance (top 10): {pca.explained_variance_ratio_[:10].round(4)}")
print(f"  Total explained variance (50 PCs): {pca.explained_variance_ratio_.sum():.4f}")

# ── Step 6: UMAP to 2D ──────────────────────────────────────────────────────
print("Running UMAP to 2D (this may take a few minutes) ...")
reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
    n_components=2,
)
latent_umap = reducer.fit_transform(latent_pca)
print(f"  UMAP output: {latent_umap.shape}")

# ── Step 7: HDBSCAN clustering ──────────────────────────────────────────────
print("Running HDBSCAN (min_cluster_size=20) ...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
cluster_labels = clusterer.fit_predict(latent_umap)
n_clusters = len(set(cluster_labels) - {-1})
n_noise = np.sum(cluster_labels == -1)
print(f"  Clusters found: {n_clusters}")
print(f"  Noise points: {n_noise}/{len(cluster_labels)}")

if n_clusters > 0:
    cluster_sizes = np.bincount(cluster_labels[cluster_labels >= 0])
    print(f"  Largest cluster size: {cluster_sizes.max()}")
    print(f"  Smallest cluster size: {cluster_sizes.min()}")
    print(f"  Median cluster size: {np.median(cluster_sizes):.0f}")

# ── Step 8: Pathogen enrichment (Fisher's exact test) ────────────────────────
print("Computing per-latent pathogen enrichment (Fisher's exact test) ...")
# Binary activation matrix for alive latents
active_matrix = (latent_alive > 0)  # (n_alive, 20000)

# Only use labeled sequences
labeled_mask = labels >= 0
is_pathogen = labels[labeled_mask] == 1  # boolean, length = n_labeled

log2_or = np.zeros(len(alive_indices), dtype=np.float64)
for i in range(len(alive_indices)):
    active_on_seq = active_matrix[i, labeled_mask]
    # 2x2 table: [[active&pathogen, active&nonpathogen], [inactive&pathogen, inactive&nonpathogen]]
    a = np.sum(active_on_seq & is_pathogen)
    b = np.sum(active_on_seq & ~is_pathogen)
    c = np.sum(~active_on_seq & is_pathogen)
    d = np.sum(~active_on_seq & ~is_pathogen)
    table = [[a, b], [c, d]]
    odds_ratio, _ = fisher_exact(table)
    # Handle inf/0/nan odds ratios
    if odds_ratio is None or np.isnan(odds_ratio):
        log2_or[i] = 0.0    # undefined — treat as neutral
    elif odds_ratio == 0:
        log2_or[i] = -10.0  # floor
    elif np.isinf(odds_ratio):
        log2_or[i] = 10.0   # ceiling
    else:
        log2_or[i] = np.log2(odds_ratio)

    if (i + 1) % 5000 == 0:
        print(f"    {i + 1}/{len(alive_indices)} latents processed")

print(f"  Done. log2(OR) range: [{log2_or.min():.2f}, {log2_or.max():.2f}]")
n_enriched = np.sum(log2_or > 1)
n_depleted = np.sum(log2_or < -1)
print(f"  Pathogen-enriched (log2 OR > 1): {n_enriched}")
print(f"  Pathogen-depleted (log2 OR < -1): {n_depleted}")

# Activation counts per latent
activation_counts = np.sum(active_matrix, axis=1)  # how many sequences activate each latent

# ── Step 9: Generate figures ─────────────────────────────────────────────────
print("Generating figures ...")

# Figure 1: UMAP colored by HDBSCAN cluster
fig, ax = plt.subplots(figsize=(10, 8))
noise_mask = cluster_labels == -1
ax.scatter(
    latent_umap[noise_mask, 0], latent_umap[noise_mask, 1],
    c="lightgray", s=1, alpha=0.3, label="noise", rasterized=True,
)
if n_clusters > 0:
    sc = ax.scatter(
        latent_umap[~noise_mask, 0], latent_umap[~noise_mask, 1],
        c=cluster_labels[~noise_mask], cmap="tab20", s=2, alpha=0.5, rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label="Cluster ID", shrink=0.7)
ax.set_title(f"SAE Latent UMAP — {n_clusters} HDBSCAN Clusters")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
fig.savefig(OUT_DIR / "latent_umap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved latent_umap.png")

# Figure 2: UMAP colored by log2(odds ratio) enrichment
fig, ax = plt.subplots(figsize=(10, 8))
clipped_or = np.clip(log2_or, -5, 5)
sc = ax.scatter(
    latent_umap[:, 0], latent_umap[:, 1],
    c=clipped_or, cmap="RdBu_r", s=2, alpha=0.5, vmin=-5, vmax=5, rasterized=True,
)
plt.colorbar(sc, ax=ax, label="log2(Odds Ratio)", shrink=0.7)
ax.set_title("SAE Latent UMAP — Pathogen Enrichment")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
fig.savefig(OUT_DIR / "latent_umap_enrichment.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved latent_umap_enrichment.png")

# Figure 3: UMAP colored by activation count
fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(
    latent_umap[:, 0], latent_umap[:, 1],
    c=activation_counts, cmap="viridis", s=2, alpha=0.5, rasterized=True,
)
plt.colorbar(sc, ax=ax, label="# Sequences Activating", shrink=0.7)
ax.set_title("SAE Latent UMAP — Activation Count")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
fig.savefig(OUT_DIR / "latent_umap_activation_count.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved latent_umap_activation_count.png")

# ── Step 10: Save CSVs ──────────────────────────────────────────────────────
print("Saving CSV outputs ...")

# Per-latent data
import csv

with open(OUT_DIR / "cluster_assignments.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["latent_id", "cluster_id", "log2_odds_ratio", "activation_count", "umap_x", "umap_y"])
    for i, latent_idx in enumerate(alive_indices):
        writer.writerow([
            int(latent_idx),
            int(cluster_labels[i]),
            f"{log2_or[i]:.4f}",
            int(activation_counts[i]),
            f"{latent_umap[i, 0]:.4f}",
            f"{latent_umap[i, 1]:.4f}",
        ])

print(f"  Saved cluster_assignments.csv ({len(alive_indices)} rows)")

# Cluster summary
with open(OUT_DIR / "cluster_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["cluster_id", "size", "mean_enrichment", "mean_activation_count"])

    unique_clusters = sorted(set(cluster_labels))
    for cid in unique_clusters:
        mask = cluster_labels == cid
        size = int(np.sum(mask))
        mean_enrich = float(np.mean(log2_or[mask]))
        mean_act = float(np.mean(activation_counts[mask]))
        writer.writerow([
            int(cid),
            size,
            f"{mean_enrich:.4f}",
            f"{mean_act:.1f}",
        ])

print(f"  Saved cluster_summary.csv ({len(unique_clusters)} rows, including noise=-1)")

max_points = 8000
stride = max(1, len(alive_indices) // max_points)
sampled_idx = np.arange(0, len(alive_indices), stride)

def _cluster_label(mean_or):
    if mean_or > 2.5:
        return "Pathogen module"
    if mean_or < 0.4:
        return "Non-pathogen module"
    return "Mixed module"

cluster_summary_payload = []
for cid in unique_clusters:
    mask = cluster_labels == cid
    size = int(np.sum(mask))
    mean_log2 = float(np.mean(log2_or[mask]))
    mean_or = float(2 ** mean_log2)
    mean_act = float(np.mean(activation_counts[mask]))
    cluster_summary_payload.append(
        {
            "cluster_id": int(cid),
            "size": size,
            "mean_enrichment": mean_or,
            "mean_activation_count": mean_act,
            "label": "Noise" if cid == -1 else _cluster_label(mean_or),
        }
    )

points = [
    {
        "x": float(latent_umap[i, 0]),
        "y": float(latent_umap[i, 1]),
        "cluster_id": int(cluster_labels[i]),
        "latent_id": int(alive_indices[i]),
        "enrichment": float(2 ** log2_or[i]),
        "activation_count": int(activation_counts[i]),
    }
    for i in sampled_idx
]

write_json(
    OUT_DIR / "api_results.json",
    {
        "points": points,
        "cluster_summary": cluster_summary_payload,
        "summary": {
            "n_latents": int(n_latents),
            "n_clusters": int(n_clusters),
            "noise_count": int(n_noise),
        },
    },
)

print("\n=== DONE ===")
print(f"Output directory: {OUT_DIR}")
