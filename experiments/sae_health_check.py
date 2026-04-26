"""
Experiment 3: SAE Health Check

Computes descriptive statistics on trained SAE activations — dead/alive
features, activation distributions, sparsity patterns. Produces figures
for the paper and a JSON summary.

Data: data/sae_model/features.npy  (20000 x 32768, float32)
Config: data/sae_model/sae_config.json
Output: results/sae_health_check/
"""

import json
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _shared import REPO_ROOT, resolve_sae_dir, write_json

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = resolve_sae_dir()

FEATURES_PATH = DATA_DIR / "features.npy"
CONFIG_PATH = DATA_DIR / "sae_config.json"
OUTPUT_DIR = REPO_ROOT / "results" / "sae_health_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(f"Loading features from {FEATURES_PATH} ...")
features = np.load(FEATURES_PATH)  # (20000, 32768)
n_sequences, n_latents = features.shape
print(f"  Shape: {features.shape}  dtype: {features.dtype}")
print(f"  {n_sequences} sequences, {n_latents} latents")

with open(CONFIG_PATH) as f:
    config = json.load(f)
k = config.get("k", 64)
d_model = config.get("d_model", 4096)
expansion = config.get("expansion_factor", 8)
print(f"  TopK k = {k}")

# ---------------------------------------------------------------------------
# 1. Per-latent statistics (axis=0 → across sequences)
# ---------------------------------------------------------------------------
print("\nComputing per-latent statistics ...")

is_active = features > 0  # bool mask (20000, 32768)

# Max activation per latent
max_activation = features.max(axis=0)  # (32768,)

# Is dead?
is_dead = max_activation == 0  # (32768,)

# Activation count per latent
activation_count = is_active.sum(axis=0)  # (32768,)

# Activation fraction
activation_fraction = activation_count / n_sequences  # (32768,)

# Mean activation when active (avoid div-by-zero for dead latents)
with np.errstate(invalid="ignore"):
    sum_when_active = (features * is_active).sum(axis=0)  # (32768,)
    mean_activation_when_active = np.where(
        activation_count > 0,
        sum_when_active / activation_count,
        0.0,
    )

# ---------------------------------------------------------------------------
# 2. Per-sequence statistics (axis=1 → across latents)
# ---------------------------------------------------------------------------
print("Computing per-sequence statistics ...")

active_features = is_active.sum(axis=1)  # (20000,)
total_activation_mass = features.sum(axis=1)  # (20000,)

# ---------------------------------------------------------------------------
# 3. Global statistics
# ---------------------------------------------------------------------------
print("Computing global statistics ...")

dead_count = int(is_dead.sum())
alive_count = n_latents - dead_count
dead_percentage = 100.0 * dead_count / n_latents

total_nonzero = int(is_active.sum())
total_entries = n_sequences * n_latents
overall_sparsity = total_nonzero / total_entries

global_stats = {
    "n_sequences": n_sequences,
    "n_latents": n_latents,
    "k": k,
    "dead_count": dead_count,
    "alive_count": alive_count,
    "dead_percentage": round(dead_percentage, 4),
    "activation_count_mean": round(float(activation_count.mean()), 4),
    "activation_count_median": round(float(np.median(activation_count)), 4),
    "activation_count_std": round(float(activation_count.std()), 4),
    "active_features_per_seq_mean": round(float(active_features.mean()), 4),
    "active_features_per_seq_median": round(float(np.median(active_features)), 4),
    "active_features_per_seq_std": round(float(active_features.std()), 4),
    "overall_sparsity_fraction": round(overall_sparsity, 8),
    "total_nonzero_entries": total_nonzero,
    "total_entries": total_entries,
}

# ---------------------------------------------------------------------------
# 4. Print summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SAE HEALTH CHECK SUMMARY")
print("=" * 60)
print(f"Sequences:            {n_sequences:>10,}")
print(f"Latents:              {n_latents:>10,}")
print(f"TopK k:               {k:>10}")
print("-" * 60)
print(f"Dead latents:         {dead_count:>10,}  ({dead_percentage:.2f}%)")
print(f"Alive latents:        {alive_count:>10,}  ({100 - dead_percentage:.2f}%)")
print("-" * 60)
print(f"Activation count (per latent):")
print(f"  Mean:               {activation_count.mean():>10.2f}")
print(f"  Median:             {np.median(activation_count):>10.2f}")
print(f"  Std:                {activation_count.std():>10.2f}")
print(f"Active features (per sequence):")
print(f"  Mean:               {active_features.mean():>10.2f}")
print(f"  Median:             {np.median(active_features):>10.2f}")
print(f"  Std:                {active_features.std():>10.2f}")
print(f"Overall sparsity:     {overall_sparsity:.8f}  ({overall_sparsity*100:.6f}%)")
print(f"Total nonzero:        {total_nonzero:>14,} / {total_entries:>14,}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------
print("\nGenerating figures ...")

# 5a. Sequences per latent histogram
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(activation_count, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
ax.set_yscale("log")
mean_ac = activation_count.mean()
median_ac = np.median(activation_count)
ax.axvline(mean_ac, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean_ac:.1f}")
ax.axvline(median_ac, color="orange", linestyle="--", linewidth=1.5, label=f"Median = {median_ac:.1f}")
ax.set_xlabel("Number of sequences activating latent")
ax.set_ylabel("Count of latents (log scale)")
ax.set_title("Sequences per Latent")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "sequences_per_latent.png", dpi=DPI)
plt.close(fig)
print("  Saved sequences_per_latent.png")

# 5b. Max activation distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(max_activation[~is_dead], bins=100, color="steelblue", edgecolor="none", alpha=0.8)
ax.set_xlabel("Max activation value")
ax.set_ylabel("Count of latents")
ax.set_title("Max Activation Distribution (alive latents only)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "max_activation_distribution.png", dpi=DPI)
plt.close(fig)
print("  Saved max_activation_distribution.png")

# 5c. Active features per sequence
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(active_features, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
ax.axvline(k, color="red", linestyle="--", linewidth=1.5, label=f"k = {k}")
ax.set_xlabel("Number of active features")
ax.set_ylabel("Count of sequences")
ax.set_title("Active Features per Sequence")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "active_features_per_sequence.png", dpi=DPI)
plt.close(fig)
print("  Saved active_features_per_sequence.png")

# 5d. Activation fraction vs max activation (scatter)
fig, ax = plt.subplots(figsize=(8, 5))
alive_mask = ~is_dead
ax.scatter(
    activation_fraction[alive_mask],
    max_activation[alive_mask],
    s=1,
    alpha=0.3,
    color="steelblue",
    label=f"Alive ({alive_count:,})",
)
if dead_count > 0:
    ax.scatter(
        activation_fraction[is_dead],
        max_activation[is_dead],
        s=3,
        alpha=0.8,
        color="red",
        label=f"Dead ({dead_count:,})",
        zorder=5,
    )
ax.set_xlabel("Activation fraction (breadth)")
ax.set_ylabel("Max activation (strength)")
ax.set_title("Feature Breadth vs Strength")
ax.legend(markerscale=5)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "activation_fraction_vs_max.png", dpi=DPI)
plt.close(fig)
print("  Saved activation_fraction_vs_max.png")

# ---------------------------------------------------------------------------
# 6. Save outputs
# ---------------------------------------------------------------------------
print("\nSaving outputs ...")

# stats.json
stats_path = OUTPUT_DIR / "stats.json"
with open(stats_path, "w") as f:
    json.dump(global_stats, f, indent=2)
print(f"  Saved {stats_path}")

# latent_stats.csv
csv_path = OUTPUT_DIR / "latent_stats.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "latent_id",
        "is_dead",
        "activation_count",
        "max_activation",
        "mean_activation_when_active",
        "activation_fraction",
    ])
    for i in range(n_latents):
        writer.writerow([
            i,
            bool(is_dead[i]),
            int(activation_count[i]),
            float(max_activation[i]),
            float(mean_activation_when_active[i]),
            float(activation_fraction[i]),
        ])
print(f"  Saved {csv_path}")

# api_results.json for backend/frontend
seq_counts, seq_edges = np.histogram(activation_count, bins=20)
max_counts, max_edges = np.histogram(max_activation[~is_dead], bins=25)
active_counts, active_edges = np.histogram(active_features, bins=20)

payload = {
    "summary": {
        "total_latents": int(n_latents),
        "dead_count": dead_count,
        "alive_count": alive_count,
        "dead_pct": round(dead_percentage, 2),
        "sparsity_pct": round(overall_sparsity * 100, 4),
        "mean_active_per_seq": round(float(active_features.mean()), 2),
        "median_active_per_seq": int(np.median(active_features)),
        "mean_activation_count": round(float(activation_count.mean()), 2),
        "median_activation_count": round(float(np.median(activation_count)), 2),
    },
    "sequences_per_latent": [
        {
            "bin_start": float(seq_edges[i]),
            "bin_end": float(seq_edges[i + 1]),
            "count": int(seq_counts[i]),
        }
        for i in range(len(seq_counts))
    ],
    "max_activation_dist": [
        {
            "bin_start": float(max_edges[i]),
            "bin_end": float(max_edges[i + 1]),
            "count": int(max_counts[i]),
        }
        for i in range(len(max_counts))
    ],
    "active_features_per_seq": [
        {
            "bin_center": float((active_edges[i] + active_edges[i + 1]) / 2.0),
            "count": int(active_counts[i]),
        }
        for i in range(len(active_counts))
    ],
    "comparison": {
        "interprot": {
            "d_model": 1280,
            "expansion": "2-4x",
            "k": 64,
            "total_latents": "4096-8192",
            "dead_pct": "varies",
        },
        "ours": {
            "d_model": d_model,
            "expansion": f"{expansion}x",
            "k": k,
            "total_latents": int(n_latents),
            "dead_pct": round(dead_percentage, 2),
        },
    },
}
write_json(OUTPUT_DIR / "api_results.json", payload)

print("\nDone.")
