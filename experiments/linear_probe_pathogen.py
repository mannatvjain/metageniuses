"""
Linear probe: predict pathogen vs non-pathogen from SAE features.
See experiment_plans/linear_probe_pathogen.md for details.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    classification_report,
)

from _shared import REPO_ROOT, resolve_sae_dir, write_json

DATA_DIR = resolve_sae_dir()
LABEL_FILE = REPO_ROOT / "data" / "human_virus_class1_labeled.jsonl"
OUT_DIR = REPO_ROOT / "results" / "linear_probe_pathogen"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Step 1: Load and align ---
print("Loading features...")
features = np.load(DATA_DIR / "features.npy")
with open(DATA_DIR / "sequence_ids.json") as f:
    sequence_ids = json.load(f)

# Build label lookup
label_lookup = {}
with open(LABEL_FILE) as f:
    for line in f:
        row = json.loads(line)
        label_lookup[row["sequence_id"]] = int(row["source"])

# Align
labels = np.array([label_lookup[sid] for sid in sequence_ids])
print(f"Features: {features.shape}, Labels: {labels.shape}")
print(f"Class balance: pathogen={labels.sum()}, non-pathogen={(1-labels).sum()}")

# --- Step 2: Train linear probe ---
print("\nTraining logistic regression (CV over regularization)...")
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

clf = LogisticRegressionCV(
    Cs=10,
    cv=5,
    scoring="accuracy",
    max_iter=2000,
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
auroc = roc_auc_score(y_test, y_prob)

print(f"\n{'='*50}")
print(f"RESULTS")
print(f"{'='*50}")
print(f"Accuracy:  {acc:.4f}")
print(f"MCC:       {mcc:.4f}")
print(f"AUROC:     {auroc:.4f}")
print(f"Best C:    {clf.C_[0]:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['non-pathogen', 'pathogen'])}")

# --- Step 3: Inspect top predictive latents ---
coefs = clf.coef_[0]  # (32768,)
top_pos_idx = np.argsort(coefs)[-10:][::-1]  # top 10 pathogen-associated
top_neg_idx = np.argsort(coefs)[:10]          # top 10 non-pathogen-associated

def latent_stats(idx, features, labels):
    """Compute per-latent stats."""
    col = features[:, idx]
    active = col > 0
    freq_pathogen = active[labels == 1].mean()
    freq_nonpathogen = active[labels == 0].mean()
    mean_act_pathogen = col[labels == 1].mean()
    mean_act_nonpathogen = col[labels == 0].mean()
    enrichment = freq_pathogen / max(freq_nonpathogen, 1e-8) if freq_pathogen > 0 else 0
    return {
        "latent_id": int(idx),
        "coefficient": float(coefs[idx]),
        "freq_pathogen": float(freq_pathogen),
        "freq_nonpathogen": float(freq_nonpathogen),
        "mean_act_pathogen": float(mean_act_pathogen),
        "mean_act_nonpathogen": float(mean_act_nonpathogen),
        "enrichment": float(enrichment),
    }

top_latents = []
print(f"\n{'='*50}")
print("TOP 10 PATHOGEN-ASSOCIATED LATENTS")
print(f"{'='*50}")
print(f"{'Latent':>8} {'Coef':>10} {'Freq(P)':>8} {'Freq(NP)':>8} {'MeanAct(P)':>11} {'MeanAct(NP)':>11} {'Enrich':>8}")
for idx in top_pos_idx:
    s = latent_stats(idx, features, labels)
    top_latents.append(s)
    print(f"{s['latent_id']:>8} {s['coefficient']:>10.4f} {s['freq_pathogen']:>8.3f} {s['freq_nonpathogen']:>8.3f} {s['mean_act_pathogen']:>11.5f} {s['mean_act_nonpathogen']:>11.5f} {s['enrichment']:>8.2f}")

print(f"\n{'='*50}")
print("TOP 10 NON-PATHOGEN-ASSOCIATED LATENTS")
print(f"{'='*50}")
print(f"{'Latent':>8} {'Coef':>10} {'Freq(P)':>8} {'Freq(NP)':>8} {'MeanAct(P)':>11} {'MeanAct(NP)':>11} {'Enrich':>8}")
for idx in top_neg_idx:
    s = latent_stats(idx, features, labels)
    top_latents.append(s)
    print(f"{s['latent_id']:>8} {s['coefficient']:>10.4f} {s['freq_pathogen']:>8.3f} {s['freq_nonpathogen']:>8.3f} {s['mean_act_pathogen']:>11.5f} {s['mean_act_nonpathogen']:>11.5f} {s['enrichment']:>8.2f}")

# Save top latents
with open(OUT_DIR / "top_latents.json", "w") as f:
    json.dump(top_latents, f, indent=2)

# --- Step 4: Figures ---

# 4a: Coefficient distribution
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(coefs, bins=200, color="steelblue", edgecolor="none", alpha=0.8)
ax.axvline(0, color="black", lw=0.5)
for idx in top_pos_idx[:3]:
    ax.axvline(coefs[idx], color="red", lw=0.8, alpha=0.7, label=f"L{idx}" if idx == top_pos_idx[0] else None)
for idx in top_neg_idx[:3]:
    ax.axvline(coefs[idx], color="blue", lw=0.8, alpha=0.7, label=f"L{idx}" if idx == top_neg_idx[0] else None)
ax.set_xlabel("Probe coefficient")
ax.set_ylabel("Count")
ax.set_title("Linear Probe Coefficient Distribution (32,768 SAE latents)")
ax.legend(["top pathogen latents", "top non-pathogen latents"])
fig.tight_layout()
fig.savefig(OUT_DIR / "coefficient_distribution.png", dpi=150)
plt.close()

# 4b: ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"SAE probe (AUROC={auroc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC: Pathogen Detection from SAE Features")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "roc_curve.png", dpi=150)
plt.close()

# 4c: Top latent activations by class
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pathogen-associated
top5_pos = top_pos_idx[:5]
x = np.arange(len(top5_pos))
w = 0.35
bars_p = [features[labels == 1, i].mean() for i in top5_pos]
bars_np = [features[labels == 0, i].mean() for i in top5_pos]
axes[0].bar(x - w/2, bars_p, w, label="Pathogen", color="red", alpha=0.7)
axes[0].bar(x + w/2, bars_np, w, label="Non-pathogen", color="steelblue", alpha=0.7)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"L{i}" for i in top5_pos], rotation=45)
axes[0].set_ylabel("Mean activation")
axes[0].set_title("Top 5 pathogen-associated latents")
axes[0].legend()

# Non-pathogen-associated
top5_neg = top_neg_idx[:5]
x = np.arange(len(top5_neg))
bars_p = [features[labels == 1, i].mean() for i in top5_neg]
bars_np = [features[labels == 0, i].mean() for i in top5_neg]
axes[1].bar(x - w/2, bars_p, w, label="Pathogen", color="red", alpha=0.7)
axes[1].bar(x + w/2, bars_np, w, label="Non-pathogen", color="steelblue", alpha=0.7)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"L{i}" for i in top5_neg], rotation=45)
axes[1].set_ylabel("Mean activation")
axes[1].set_title("Top 5 non-pathogen-associated latents")
axes[1].legend()

fig.suptitle("Mean SAE Latent Activation by Pathogen Class")
fig.tight_layout()
fig.savefig(OUT_DIR / "top_latent_activations.png", dpi=150)
plt.close()

# Save summary
summary = {
    "accuracy": acc,
    "mcc": mcc,
    "auroc": auroc,
    "best_C": float(clf.C_[0]),
    "n_train": len(y_train),
    "n_test": len(y_test),
    "n_features": features.shape[1],
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Save API payload for backend/frontend
hist_counts, hist_edges = np.histogram(coefs, bins=40)
coef_distribution = [
    {
        "bin_center": float((hist_edges[i] + hist_edges[i + 1]) / 2.0),
        "count": int(hist_counts[i]),
    }
    for i in range(len(hist_counts))
]
roc_payload = [
    {"fpr": float(a), "tpr": float(b)}
    for a, b in zip(fpr, tpr)
]
top_latents_api = []
for item in top_latents:
    top_latents_api.append(
        {
            **item,
            "direction": "pathogen" if item["coefficient"] > 0 else "nonpathogen",
        }
    )

write_json(
    OUT_DIR / "api_results.json",
    {
        "summary": summary,
        "roc_curve": roc_payload,
        "coefficient_distribution": coef_distribution,
        "top_latents": top_latents_api,
    },
)

print(f"\nResults saved to {OUT_DIR}/")
print(f"  summary.json, top_latents.json")
print(f"  coefficient_distribution.png, roc_curve.png, top_latent_activations.png")
print(f"  api_results.json")
