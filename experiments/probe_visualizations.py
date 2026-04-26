"""
Visualizations for linear probe results.
1. Top latent activation distributions (pathogen vs non-pathogen)
2. Cumulative coefficient importance (distributed vs sparse signal)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data/sae_model")
LABEL_FILE = Path("data/human_virus_class1_labeled.jsonl")
OUT_DIR = Path("results/linear_probe_pathogen")
RESULTS = json.loads((OUT_DIR / "top_latents.json").read_text())
SUMMARY = json.loads((OUT_DIR / "summary.json").read_text())

# Load data
features = np.load(DATA_DIR / "features.npy")
with open(DATA_DIR / "sequence_ids.json") as f:
    sequence_ids = json.load(f)

label_lookup = {}
with open(LABEL_FILE) as f:
    for line in f:
        row = json.loads(line)
        label_lookup[row["sequence_id"]] = int(row["source"])
labels = np.array([label_lookup[sid] for sid in sequence_ids])

pathogen_mask = labels == 1
nonpathogen_mask = labels == 0

# Recover coefficients from the saved top latents to get the latent IDs
top_pathogen = [r for r in RESULTS if r["coefficient"] > 0]
top_nonpathogen = [r for r in RESULTS if r["coefficient"] < 0]

# ---- VIZ 1: Activation distributions for top latents ----

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for i, lat in enumerate(top_pathogen[:5]):
    ax = axes[0, i]
    idx = lat["latent_id"]
    vals_p = features[pathogen_mask, idx]
    vals_np = features[nonpathogen_mask, idx]

    # Filter to nonzero for cleaner histograms
    bins = np.linspace(0, max(vals_p.max(), vals_np.max()) * 1.05, 50)

    ax.hist(vals_p, bins=bins, alpha=0.6, color="red", label="Pathogen", density=True)
    ax.hist(vals_np, bins=bins, alpha=0.6, color="steelblue", label="Non-path", density=True)
    ax.axvline(vals_p.mean(), color="red", ls="--", lw=1)
    ax.axvline(vals_np.mean(), color="steelblue", ls="--", lw=1)
    ax.set_title(f"L{idx}\ncoef={lat['coefficient']:.1f}", fontsize=10)
    if i == 0:
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)

for i, lat in enumerate(top_nonpathogen[:5]):
    ax = axes[1, i]
    idx = lat["latent_id"]
    vals_p = features[pathogen_mask, idx]
    vals_np = features[nonpathogen_mask, idx]

    bins = np.linspace(0, max(vals_p.max(), vals_np.max()) * 1.05, 50)

    ax.hist(vals_p, bins=bins, alpha=0.6, color="red", label="Pathogen", density=True)
    ax.hist(vals_np, bins=bins, alpha=0.6, color="steelblue", label="Non-path", density=True)
    ax.axvline(vals_p.mean(), color="red", ls="--", lw=1)
    ax.axvline(vals_np.mean(), color="steelblue", ls="--", lw=1)
    ax.set_title(f"L{idx}\ncoef={lat['coefficient']:.1f}", fontsize=10)
    if i == 0:
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)

axes[0, 0].annotate("Pathogen-associated", xy=(0, 0.5), xytext=(-0.35, 0.5),
                     xycoords="axes fraction", textcoords="axes fraction",
                     fontsize=12, fontweight="bold", rotation=90, va="center")
axes[1, 0].annotate("Non-pathogen-associated", xy=(0, 0.5), xytext=(-0.35, 0.5),
                     xycoords="axes fraction", textcoords="axes fraction",
                     fontsize=12, fontweight="bold", rotation=90, va="center")

fig.suptitle("Activation Distributions: Top 5 Latents per Class", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "activation_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved activation_distributions.png")


# ---- VIZ 2: Cumulative coefficient importance ----
# Re-train probe to get full coefficient vector (or load if available)
# Faster: just retrain quickly
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

print("Retraining probe for full coefficient vector...")
clf = LogisticRegressionCV(Cs=10, cv=5, max_iter=2000, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
coefs = clf.coef_[0]

abs_coefs = np.abs(coefs)
sorted_idx = np.argsort(abs_coefs)[::-1]
sorted_abs = abs_coefs[sorted_idx]

# Cumulative sum of |coefficients|, normalized
cumsum = np.cumsum(sorted_abs)
cumsum_norm = cumsum / cumsum[-1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Full view
ax1.plot(np.arange(len(cumsum_norm)), cumsum_norm, color="steelblue", lw=1.5)
ax1.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
ax1.axhline(0.9, color="gray", ls="--", lw=0.8, alpha=0.5)

# Find how many features for 50% and 90%
n_50 = np.searchsorted(cumsum_norm, 0.5) + 1
n_90 = np.searchsorted(cumsum_norm, 0.9) + 1
ax1.axvline(n_50, color="red", ls="--", lw=0.8)
ax1.axvline(n_90, color="red", ls="--", lw=0.8)
ax1.annotate(f"50%: {n_50:,} latents", xy=(n_50, 0.5), xytext=(n_50 + 1000, 0.4),
             fontsize=10, arrowprops=dict(arrowstyle="->", color="red"), color="red")
ax1.annotate(f"90%: {n_90:,} latents", xy=(n_90, 0.9), xytext=(n_90 + 1000, 0.8),
             fontsize=10, arrowprops=dict(arrowstyle="->", color="red"), color="red")

ax1.set_xlabel("Number of latents (ranked by |coefficient|)")
ax1.set_ylabel("Cumulative fraction of total |coefficient|")
ax1.set_title("How distributed is the pathogen signal?")

# Zoomed view: top 500
ax2.bar(np.arange(100), sorted_abs[:100], color="steelblue", alpha=0.8, width=1.0)
ax2.set_xlabel("Latent rank")
ax2.set_ylabel("|Coefficient|")
ax2.set_title("Top 100 latent coefficients")

fig.suptitle(
    f"Probe: {SUMMARY['accuracy']:.1%} acc, {SUMMARY['mcc']:.3f} MCC — "
    f"{n_50:,} latents for 50% signal, {n_90:,} for 90%",
    fontsize=12, y=1.02
)
fig.tight_layout()
fig.savefig(OUT_DIR / "cumulative_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved cumulative_importance.png")
print(f"  50% of signal in top {n_50:,} / 32,768 latents")
print(f"  90% of signal in top {n_90:,} / 32,768 latents")
