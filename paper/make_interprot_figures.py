"""Generate InterProt-style figures for the paper."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 12
import json, os

OUT = "paper/figures_interprot"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Figure 1: Organism-specific feature count across layers
# (InterProt Fig 3d equivalent)
# ============================================================
layers = [8, 16, 24, 32]
specific_counts = [52, 7, 10, 4]  # F1 > 0.7

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(layers, specific_counts, 'o-', color='#2563eb', linewidth=2.5, markersize=10, zorder=5)
for i, (l, c) in enumerate(zip(layers, specific_counts)):
    ax.annotate(str(c), (l, c), textcoords="offset points", xytext=(0, 12),
                ha='center', fontsize=13, fontweight='bold', color='#2563eb')
ax.set_xlabel("Layer", fontsize=14)
ax.set_ylabel("Pathogen-specific latent count (F1 > 0.7)", fontsize=13)
ax.set_title("Organism-Specific Features Peak in Early Layers", fontsize=14, fontweight='bold')
ax.set_xticks(layers)
ax.set_xlim(4, 36)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/specific_features_by_layer.png", dpi=200)
plt.close()
print(f"Saved specific_features_by_layer.png")

# ============================================================
# Figure 2: Bimodal activation histogram for organism detector
# (InterProt Fig 3a equivalent)
# Pick a high-confidence astrovirus latent from layer 32
# ============================================================
features = np.load("data/sae_model/features.npy")
with open("data/sae_model/sequence_ids.json") as f:
    seq_ids = json.load(f)

# Load labels
labels_df = pd.read_json("data/human_virus_class1_labeled.jsonl", lines=True)
# Build label array aligned to features
label_map = {row['sequence_id']: row['source'] for _, row in labels_df.iterrows()}
labels = np.array([label_map.get(sid, -1) for sid in seq_ids])

# Load organism detector results to find a good astrovirus latent
org_df = pd.read_csv("results/organism_detectors/organism_labels.csv")
astrovirus_latents = org_df[(org_df.confidence == 'high') & (org_df.dominant_organism.str.contains('astrovirus', case=False, na=False))]
latent_id = astrovirus_latents.iloc[0]['latent_id']

activations = features[:, latent_id]
pathogen_mask = labels == 1
non_pathogen_mask = labels == 0

fig, ax = plt.subplots(figsize=(6, 4))
bins = np.linspace(0, activations.max() * 1.05, 50)

ax.hist(activations[non_pathogen_mask], bins=bins, alpha=0.7, label='Non-pathogen',
        color='#94a3b8', edgecolor='white', linewidth=0.5)
ax.hist(activations[pathogen_mask], bins=bins, alpha=0.7, label='Pathogen',
        color='#ef4444', edgecolor='white', linewidth=0.5)

org_name = astrovirus_latents.iloc[0]['dominant_organism']
ax.set_xlabel("Mean activation", fontsize=13)
ax.set_ylabel("Number of sequences", fontsize=13)
ax.set_title(f"Organism-specific latent L{latent_id}: {org_name}", fontsize=13, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/organism_detector_histogram.png", dpi=200)
plt.close()
print(f"Saved organism_detector_histogram.png (latent {latent_id})")

# ============================================================
# Figure 3: Probe AUROC across layers
# (InterProt Fig 5 equivalent)
# ============================================================
layer_aurocs = {8: 0.9912, 16: 0.9906, 24: 0.9914, 32: 0.9874}
layer_accs = {8: 0.9545, 16: 0.9540, 24: 0.9535, 32: 0.9455}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# AUROC
ax1.plot(layers, [layer_aurocs[l] for l in layers], 'o-', color='#2563eb', linewidth=2.5, markersize=9)
for l in layers:
    ax1.annotate(f"{layer_aurocs[l]:.4f}", (l, layer_aurocs[l]),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)
ax1.set_xlabel("Layer", fontsize=13)
ax1.set_ylabel("AUROC", fontsize=13)
ax1.set_title("Probe AUROC by Layer", fontsize=13, fontweight='bold')
ax1.set_xticks(layers)
ax1.set_ylim(0.984, 0.994)
ax1.grid(True, alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Accuracy
ax2.plot(layers, [layer_accs[l] for l in layers], 's-', color='#16a34a', linewidth=2.5, markersize=9)
for l in layers:
    ax2.annotate(f"{layer_accs[l]:.4f}", (l, layer_accs[l]),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)
ax2.set_xlabel("Layer", fontsize=13)
ax2.set_ylabel("Accuracy", fontsize=13)
ax2.set_title("Probe Accuracy by Layer", fontsize=13, fontweight='bold')
ax2.set_xticks(layers)
ax2.set_ylim(0.940, 0.960)
ax2.grid(True, alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/probe_auroc_by_layer.png", dpi=200)
plt.close()
print("Saved probe_auroc_by_layer.png")

# ============================================================
# Figure 4: Enrichment symmetry across layers
# (bonus — shows how representation changes with depth)
# ============================================================
enrichment_data = {
    8:  {'pathogen': 6566, 'non_pathogen': 6476, 'specific': 52},
    16: {'pathogen': 5855, 'non_pathogen': 5740, 'specific': 7},
    24: {'pathogen': 6482, 'non_pathogen': 4926, 'specific': 10},
    32: {'pathogen': 16519, 'non_pathogen': 2534, 'specific': 4},
}

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(layers))
width = 0.35

path_counts = [enrichment_data[l]['pathogen'] for l in layers]
nonpath_counts = [enrichment_data[l]['non_pathogen'] for l in layers]

bars1 = ax.bar(x - width/2, path_counts, width, label='Pathogen-enriched', color='#ef4444', alpha=0.85)
bars2 = ax.bar(x + width/2, nonpath_counts, width, label='Non-pathogen-enriched', color='#3b82f6', alpha=0.85)

ax.set_xlabel("Layer", fontsize=13)
ax.set_ylabel("Number of enriched latents (FDR < 0.01)", fontsize=12)
ax.set_title("Enrichment Symmetry Shifts Across Layers", fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers])
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.2, axis='y')

# Add ratio annotations
for i, l in enumerate(layers):
    ratio = enrichment_data[l]['pathogen'] / enrichment_data[l]['non_pathogen']
    ax.annotate(f"{ratio:.1f}x", (i, max(path_counts[i], nonpath_counts[i])),
                textcoords="offset points", xytext=(0, 8), ha='center', fontsize=10, fontstyle='italic')

plt.tight_layout()
plt.savefig(f"{OUT}/enrichment_symmetry_by_layer.png", dpi=200)
plt.close()
print("Saved enrichment_symmetry_by_layer.png")

print(f"\nAll figures saved to {OUT}/")
