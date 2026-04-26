"""
Experiment 6: Cross-Delivery Generalization
Train probe on class 1, test on class 2.
See experiment_plans/06_cross_delivery.md for details.
"""

import csv
import json
import os
import signal
import sys
import threading
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import fisher_exact, pearsonr, spearmanr

# ---------- Memory monitor ----------
RAM_LIMIT_GB = 125
RAM_WARN_GB = 100

def _get_rss_gb():
    """Get current process RSS in GB (macOS/Linux)."""
    import resource
    # maxrss on macOS is in bytes
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss_bytes / (1024 ** 3)
    return rss_bytes / (1024 ** 2)  # Linux: in KB

def _monitor_memory(interval=10):
    """Background thread that checks RSS every `interval` seconds."""
    warned = False
    while not _monitor_stop.is_set():
        rss = _get_rss_gb()
        if rss >= RAM_LIMIT_GB:
            print(f"\n*** MEMORY LIMIT EXCEEDED: {rss:.1f} GB >= {RAM_LIMIT_GB} GB. ABORTING. ***", flush=True)
            os.kill(os.getpid(), signal.SIGTERM)
            return
        elif rss >= RAM_WARN_GB and not warned:
            print(f"\n*** WARNING: memory at {rss:.1f} GB (limit: {RAM_LIMIT_GB} GB) ***", flush=True)
            warned = True
        elif rss < RAM_WARN_GB:
            warned = False
        _monitor_stop.wait(interval)

_monitor_stop = threading.Event()
_monitor_thread = threading.Thread(target=_monitor_memory, daemon=True)
_monitor_thread.start()
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    f1_score,
    roc_curve,
    classification_report,
)

DATA_DIR = Path("data/sae_model")
LABEL_FILE_C1 = Path("data/human_virus_class1_labeled.jsonl")
LABEL_FILE_C2 = Path("data/human_virus_class2_labeled.jsonl")
OUT_DIR = Path("results/cross_delivery")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Helper: load features + labels, aligned by sequence_id
# ============================================================
def load_features_and_labels(feature_path, ids_path, label_path):
    """Load features.npy and align with labels from JSONL."""
    features = np.load(feature_path)
    with open(ids_path) as f:
        sequence_ids = json.load(f)
    label_lookup = {}
    with open(label_path) as f:
        for line in f:
            row = json.loads(line)
            label_lookup[row["sequence_id"]] = int(row["source"])
    labels = np.array([label_lookup[sid] for sid in sequence_ids])
    return features, labels, sequence_ids


def compute_enrichment(features, labels):
    """Compute Fisher's exact test OR for each latent."""
    active = (features > 0).astype(np.int64)
    pathogen = (labels == 1).astype(np.int64)
    nonpathogen = (labels == 0).astype(np.int64)

    # Vectorized contingency table computation
    a = active.T @ pathogen
    b = active.T @ nonpathogen
    c = pathogen.sum() - a
    d = nonpathogen.sum() - b

    n_latents = features.shape[1]
    ors = np.zeros(n_latents)
    pvals = np.zeros(n_latents)

    start = time.time()
    for i in range(n_latents):
        if i % 5000 == 0:
            elapsed = time.time() - start
            print(f"  {i}/{n_latents} ({elapsed:.0f}s)")
        try:
            ors[i], pvals[i] = fisher_exact([[a[i], b[i]], [c[i], d[i]]])
        except Exception:
            ors[i], pvals[i] = 1.0, 1.0

    print(f"  Done: {n_latents} latents in {time.time() - start:.0f}s")
    return ors, pvals


# ============================================================
# Step 0: Load data
# ============================================================
print("=" * 60)
print("EXPERIMENT 6: Cross-Delivery Generalization")
print("=" * 60)

print("\nLoading class 1 features and labels...")
X_c1, y_c1, ids_c1 = load_features_and_labels(
    DATA_DIR / "features.npy",
    DATA_DIR / "sequence_ids.json",
    LABEL_FILE_C1,
)
print(f"  Class 1: {X_c1.shape[0]} samples, {X_c1.shape[1]} features")
print(f"  Balance: pathogen={y_c1.sum()}, non-pathogen={(1 - y_c1).sum()}")

print("\nLoading class 2 features and labels...")
X_c2, y_c2, ids_c2 = load_features_and_labels(
    DATA_DIR / "features_class2.npy",
    DATA_DIR / "sequence_ids_class2.json",
    LABEL_FILE_C2,
)
print(f"  Class 2: {X_c2.shape[0]} samples, {X_c2.shape[1]} features")
print(f"  Balance: pathogen={y_c2.sum()}, non-pathogen={(1 - y_c2).sum()}")

# ============================================================
# Step 1: Train probe on 100% of class 1
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: Train logistic regression on class 1 (full)")
print("=" * 60)

t0 = time.time()
clf = LogisticRegressionCV(
    Cs=5,
    cv=3,
    scoring="accuracy",
    max_iter=2000,
    random_state=42,
    n_jobs=1,
)
clf.fit(X_c1, y_c1)
print(f"  Training took {time.time() - t0:.1f}s")

# Class 1 train performance (for comparison)
y_pred_c1 = clf.predict(X_c1)
y_prob_c1 = clf.predict_proba(X_c1)[:, 1]

acc_c1 = accuracy_score(y_c1, y_pred_c1)
mcc_c1 = matthews_corrcoef(y_c1, y_pred_c1)
auroc_c1 = roc_auc_score(y_c1, y_prob_c1)
auprc_c1 = average_precision_score(y_c1, y_prob_c1)
f1_c1 = f1_score(y_c1, y_pred_c1)

print(f"\nClass 1 (train) performance:")
print(f"  Accuracy:  {acc_c1:.4f}")
print(f"  MCC:       {mcc_c1:.4f}")
print(f"  AUROC:     {auroc_c1:.4f}")
print(f"  AUPRC:     {auprc_c1:.4f}")
print(f"  F1:        {f1_c1:.4f}")
print(f"  Best C:    {clf.C_[0]:.4f}")

# ============================================================
# Step 2: Evaluate on class 2
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Evaluate on class 2 (held-out delivery)")
print("=" * 60)

y_pred_c2 = clf.predict(X_c2)
y_prob_c2 = clf.predict_proba(X_c2)[:, 1]

acc_c2 = accuracy_score(y_c2, y_pred_c2)
mcc_c2 = matthews_corrcoef(y_c2, y_pred_c2)
auroc_c2 = roc_auc_score(y_c2, y_prob_c2)
auprc_c2 = average_precision_score(y_c2, y_prob_c2)
f1_c2 = f1_score(y_c2, y_pred_c2)

print(f"\nClass 2 (test) performance:")
print(f"  Accuracy:  {acc_c2:.4f}")
print(f"  MCC:       {mcc_c2:.4f}")
print(f"  AUROC:     {auroc_c2:.4f}")
print(f"  AUPRC:     {auprc_c2:.4f}")
print(f"  F1:        {f1_c2:.4f}")
print(f"\n{classification_report(y_c2, y_pred_c2, target_names=['non-pathogen', 'pathogen'])}")

# Comparison table
print("\n" + "=" * 60)
print("COMPARISON: Class 1 (train) vs Class 2 (test)")
print("=" * 60)
print(f"{'Metric':<12} {'Class 1':>10} {'Class 2':>10} {'Delta':>10}")
print("-" * 44)
for name, v1, v2 in [
    ("Accuracy", acc_c1, acc_c2),
    ("MCC", mcc_c1, mcc_c2),
    ("AUROC", auroc_c1, auroc_c2),
    ("AUPRC", auprc_c1, auprc_c2),
    ("F1", f1_c1, f1_c2),
]:
    print(f"{name:<12} {v1:>10.4f} {v2:>10.4f} {v2 - v1:>+10.4f}")

# ROC curve for class 2
fpr_c2, tpr_c2, _ = roc_curve(y_c2, y_prob_c2)
fpr_c1, tpr_c1, _ = roc_curve(y_c1, y_prob_c1)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr_c1, tpr_c1, color="steelblue", lw=2, alpha=0.5,
        label=f"Class 1 train (AUROC={auroc_c1:.3f})")
ax.plot(fpr_c2, tpr_c2, color="red", lw=2,
        label=f"Class 2 test (AUROC={auroc_c2:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC: Cross-Delivery Generalization (Train C1 -> Test C2)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT_DIR / "roc_curve_class2.png", dpi=150)
plt.close()
print(f"\nSaved: {OUT_DIR / 'roc_curve_class2.png'}")

# ============================================================
# Step 3: Feature stability analysis
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Feature stability analysis (Fisher's OR)")
print("=" * 60)

n_latents = X_c1.shape[1]

print("\nComputing enrichment for class 1...")
t0 = time.time()
or_c1, pval_c1 = compute_enrichment(X_c1, y_c1)
print(f"  Class 1 enrichment took {time.time() - t0:.1f}s")

print("\nComputing enrichment for class 2...")
t0 = time.time()
or_c2, pval_c2 = compute_enrichment(X_c2, y_c2)
print(f"  Class 2 enrichment took {time.time() - t0:.1f}s")

# Log2 transform ORs (handle zeros/inf)
log2_or_c1 = np.log2(np.clip(or_c1, 1e-10, 1e10))
log2_or_c2 = np.log2(np.clip(or_c2, 1e-10, 1e10))

# Significance: p < 0.05 in EITHER class (after simple Bonferroni for display)
alpha = 0.05 / n_latents
sig_c1 = pval_c1 < alpha
sig_c2 = pval_c2 < alpha
sig_either = sig_c1 | sig_c2
sig_both = sig_c1 & sig_c2
n_sig_c1 = sig_c1.sum()
n_sig_c2 = sig_c2.sum()
n_sig_either = sig_either.sum()
n_sig_both = sig_both.sum()

print(f"\nSignificant latents (Bonferroni p < {alpha:.2e}):")
print(f"  Class 1 only:   {n_sig_c1}")
print(f"  Class 2 only:   {n_sig_c2}")
print(f"  Either class:   {n_sig_either}")
print(f"  Both classes:    {n_sig_both}")

# Correlation of enrichment vectors
finite_mask = np.isfinite(log2_or_c1) & np.isfinite(log2_or_c2)

pearson_all, pearson_p = pearsonr(log2_or_c1[finite_mask], log2_or_c2[finite_mask])
spearman_all, spearman_p = spearmanr(log2_or_c1[finite_mask], log2_or_c2[finite_mask])

print(f"\nEnrichment correlation (all {finite_mask.sum()} finite latents):")
print(f"  Pearson r:  {pearson_all:.4f} (p={pearson_p:.2e})")
print(f"  Spearman r: {spearman_all:.4f} (p={spearman_p:.2e})")

if n_sig_either > 2:
    sig_mask = sig_either & finite_mask
    pearson_sig, _ = pearsonr(log2_or_c1[sig_mask], log2_or_c2[sig_mask])
    spearman_sig, _ = spearmanr(log2_or_c1[sig_mask], log2_or_c2[sig_mask])
    print(f"\nEnrichment correlation (significant latents only, n={sig_mask.sum()}):")
    print(f"  Pearson r:  {pearson_sig:.4f}")
    print(f"  Spearman r: {spearman_sig:.4f}")
else:
    pearson_sig = spearman_sig = float("nan")

# ============================================================
# Enrichment scatter plot
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))

# Plot non-significant latents faintly
nonsig = ~sig_either & finite_mask
ax.scatter(
    log2_or_c1[nonsig], log2_or_c2[nonsig],
    s=1, alpha=0.05, c="gray", label=f"Non-significant ({nonsig.sum():,})",
    rasterized=True,
)

# Plot significant-in-one-class latents
sig_one = sig_either & ~sig_both & finite_mask
if sig_one.sum() > 0:
    ax.scatter(
        log2_or_c1[sig_one], log2_or_c2[sig_one],
        s=4, alpha=0.3, c="orange", label=f"Significant in one class ({sig_one.sum():,})",
        rasterized=True,
    )

# Plot significant-in-both latents
if sig_both.sum() > 0:
    sig_b = sig_both & finite_mask
    ax.scatter(
        log2_or_c1[sig_b], log2_or_c2[sig_b],
        s=6, alpha=0.5, c="red", label=f"Significant in both ({sig_b.sum():,})",
        rasterized=True,
    )

# Reference lines
lim = max(abs(log2_or_c1[finite_mask]).max(), abs(log2_or_c2[finite_mask]).max())
lim = min(lim, 15)  # cap for display
ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5, label="y = x")
ax.axhline(0, color="gray", lw=0.3)
ax.axvline(0, color="gray", lw=0.3)

ax.set_xlabel("Class 1 log2(OR)")
ax.set_ylabel("Class 2 log2(OR)")
ax.set_title(
    f"Feature Enrichment Stability: Class 1 vs Class 2\n"
    f"Pearson r = {pearson_all:.3f}, Spearman r = {spearman_all:.3f}"
)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "enrichment_scatter.png", dpi=150)
plt.close()
print(f"\nSaved: {OUT_DIR / 'enrichment_scatter.png'}")

# ============================================================
# Save feature stability CSV
# ============================================================
csv_path = OUT_DIR / "feature_stability.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "latent_id", "or_class1", "pval_class1", "log2or_class1",
        "or_class2", "pval_class2", "log2or_class2",
        "sig_class1", "sig_class2",
    ])
    for i in range(n_latents):
        writer.writerow([
            i,
            f"{or_c1[i]:.6f}", f"{pval_c1[i]:.6e}", f"{log2_or_c1[i]:.6f}",
            f"{or_c2[i]:.6f}", f"{pval_c2[i]:.6e}", f"{log2_or_c2[i]:.6f}",
            int(sig_c1[i]), int(sig_c2[i]),
        ])
print(f"Saved: {csv_path}")

# ============================================================
# Save comparison table (markdown)
# ============================================================
table_path = OUT_DIR / "comparison_table.md"
with open(table_path, "w") as f:
    f.write("# Cross-Delivery Generalization: Class 1 vs Class 2\n\n")
    f.write("| Metric | Class 1 (train) | Class 2 (test) | Delta |\n")
    f.write("|--------|---------------:|---------------:|------:|\n")
    for name, v1, v2 in [
        ("Accuracy", acc_c1, acc_c2),
        ("MCC", mcc_c1, mcc_c2),
        ("AUROC", auroc_c1, auroc_c2),
        ("AUPRC", auprc_c1, auprc_c2),
        ("F1", f1_c1, f1_c2),
    ]:
        f.write(f"| {name} | {v1:.4f} | {v2:.4f} | {v2 - v1:+.4f} |\n")
    f.write(f"\n## Feature Stability\n\n")
    f.write(f"- Enrichment correlation (all latents): Pearson r = {pearson_all:.4f}, Spearman r = {spearman_all:.4f}\n")
    if not np.isnan(pearson_sig):
        f.write(f"- Enrichment correlation (significant latents): Pearson r = {pearson_sig:.4f}, Spearman r = {spearman_sig:.4f}\n")
    f.write(f"- Significant in class 1: {n_sig_c1}\n")
    f.write(f"- Significant in class 2: {n_sig_c2}\n")
    f.write(f"- Significant in both: {n_sig_both}\n")
    f.write(f"- Best regularization C: {clf.C_[0]:.4f}\n")
print(f"Saved: {table_path}")

# ============================================================
# Save summary JSON
# ============================================================
summary = {
    "class1_train": {
        "accuracy": acc_c1,
        "mcc": mcc_c1,
        "auroc": auroc_c1,
        "auprc": auprc_c1,
        "f1": f1_c1,
        "n_samples": int(len(y_c1)),
    },
    "class2_test": {
        "accuracy": acc_c2,
        "mcc": mcc_c2,
        "auroc": auroc_c2,
        "auprc": auprc_c2,
        "f1": f1_c2,
        "n_samples": int(len(y_c2)),
    },
    "delta": {
        "accuracy": acc_c2 - acc_c1,
        "mcc": mcc_c2 - mcc_c1,
        "auroc": auroc_c2 - auroc_c1,
        "auprc": auprc_c2 - auprc_c1,
        "f1": f1_c2 - f1_c1,
    },
    "feature_stability": {
        "pearson_all": pearson_all,
        "spearman_all": spearman_all,
        "pearson_significant": pearson_sig if not np.isnan(pearson_sig) else None,
        "spearman_significant": spearman_sig if not np.isnan(spearman_sig) else None,
        "n_sig_class1": int(n_sig_c1),
        "n_sig_class2": int(n_sig_c2),
        "n_sig_both": int(n_sig_both),
        "n_sig_either": int(n_sig_either),
    },
    "best_C": float(clf.C_[0]),
    "n_features": int(X_c1.shape[1]),
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {OUT_DIR / 'summary.json'}")

print("\n" + "=" * 60)
print("DONE. All results saved to results/cross_delivery/")
print("=" * 60)
