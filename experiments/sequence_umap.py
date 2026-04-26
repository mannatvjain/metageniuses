"""
Experiment 4: Sequence UMAP
Project 20,000 metagenomic sequences from 32,768-dim SAE feature space
to 2D using UMAP. Color by pathogen label.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import umap
import hdbscan
from _shared import REPO_ROOT, resolve_sae_dir, write_json

# ── Paths ──────────────────────────────────────────────────────────────
SAE_DIR = resolve_sae_dir()
FEATURES_PATH = SAE_DIR / "features.npy"
SEQ_IDS_PATH = SAE_DIR / "sequence_ids.json"
LABELED_JSONL = REPO_ROOT / "data" / "human_virus_class1_labeled.jsonl"

OUT_DIR = REPO_ROOT / "results" / "sequence_umap"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_labels(jsonl_path: Path) -> dict[str, str]:
    """Return {sequence_id: source} from labeled JSONL."""
    labels = {}
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            labels[obj["sequence_id"]] = obj["source"]
    return labels


def main():
    # 1. Load data
    print("Loading features …")
    features = np.load(FEATURES_PATH)
    print(f"  features shape: {features.shape}")

    with open(SEQ_IDS_PATH) as f:
        seq_ids = json.load(f)
    print(f"  sequence IDs: {len(seq_ids)}")

    labels_map = load_labels(LABELED_JSONL)
    print(f"  labeled sequences: {len(labels_map)}")

    # Align rows to labels
    sources = []
    for sid in seq_ids:
        src = labels_map.get(sid)
        if src is None:
            raise ValueError(f"sequence_id {sid!r} not found in labeled JSONL")
        sources.append(int(src))
    sources = np.array(sources)
    n_pathogen = (sources == 1).sum()
    n_nonpathogen = (sources == 0).sum()
    print(f"  pathogen: {n_pathogen}, non-pathogen: {n_nonpathogen}")

    # 2. PCA to 50 dims
    print("\nRunning PCA (50 components) …")
    pca = PCA(n_components=50, random_state=42)
    features_pca = pca.fit_transform(features)
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"  Variance explained by 50 components: {cumulative_var[-1]:.4f}")
    print(f"  Top-10 components explain: {cumulative_var[9]:.4f}")

    # PCA scree plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, 51), pca.explained_variance_ratio_, alpha=0.6, label="Individual")
    ax.plot(range(1, 51), cumulative_var, "r-o", markersize=3, label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Scree Plot (SAE Features)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_variance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT_DIR / 'pca_variance.png'}")

    # 3. UMAP to 2D
    print("\nRunning UMAP (n_neighbors=15, min_dist=0.1) …")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
    )
    embedding = reducer.fit_transform(features_pca)
    print(f"  Embedding shape: {embedding.shape}")

    # 4. Scatter plot colored by pathogen label
    fig, ax = plt.subplots(figsize=(10, 8))
    mask_np = sources == 0
    mask_p = sources == 1
    ax.scatter(
        embedding[mask_np, 0], embedding[mask_np, 1],
        c="blue", alpha=0.3, s=4, label=f"Non-pathogen (n={n_nonpathogen})",
    )
    ax.scatter(
        embedding[mask_p, 0], embedding[mask_p, 1],
        c="red", alpha=0.3, s=4, label=f"Pathogen (n={n_pathogen})",
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("UMAP of SAE Features — Pathogen vs Non-Pathogen")
    ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "umap_pathogen.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT_DIR / 'umap_pathogen.png'}")

    # 5. Save UMAP coordinates + labels to CSV
    df = pd.DataFrame({
        "sequence_id": seq_ids,
        "umap_x": embedding[:, 0],
        "umap_y": embedding[:, 1],
        "source": sources,
    })
    df.to_csv(OUT_DIR / "umap_coords.csv", index=False)
    print(f"  Saved {OUT_DIR / 'umap_coords.csv'}")

    # 6. HDBSCAN sub-clustering
    print("\nRunning HDBSCAN on 2D embedding …")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
    cluster_labels = clusterer.fit_predict(embedding)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise}")

    df_clusters = pd.DataFrame({
        "sequence_id": seq_ids,
        "cluster_id": cluster_labels,
    })
    df_clusters.to_csv(OUT_DIR / "subclusters.csv", index=False)
    print(f"  Saved {OUT_DIR / 'subclusters.csv'}")

    # Print cluster composition
    for cid in sorted(set(cluster_labels)):
        mask = cluster_labels == cid
        n_total = mask.sum()
        n_path = (sources[mask] == 1).sum()
        label = "noise" if cid == -1 else f"cluster {cid}"
        print(f"    {label}: {n_total} sequences ({n_path} pathogen, {n_total - n_path} non-pathogen)")

    # 7. API payload for frontend
    max_points = 8000
    stride = max(1, len(seq_ids) // max_points)
    sampled_idx = np.arange(0, len(seq_ids), stride)
    points = [
        {
            "x": float(embedding[i, 0]),
            "y": float(embedding[i, 1]),
            "label": int(sources[i]),
            "sequence_id": seq_ids[i],
        }
        for i in sampled_idx
    ]
    pca_variance = [
        {
            "component": int(i + 1),
            "explained_variance": float(pca.explained_variance_ratio_[i]),
        }
        for i in range(len(pca.explained_variance_ratio_))
    ]
    write_json(
        OUT_DIR / "api_results.json",
        {
            "points": points,
            "pca_variance": pca_variance,
            "summary": {
                "n_sequences": int(len(seq_ids)),
                "n_pathogen": int(n_pathogen),
                "n_nonpathogen": int(n_nonpathogen),
                "pca_dims": 50,
                "variance_explained_50": float(cumulative_var[-1]),
            },
        },
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
