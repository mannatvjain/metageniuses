# InterProt Reference

Key methodology and decisions from "From Mechanistic Interpretability to Mechanistic Biology" (Adams, Bai, Lee, Yu, AlQuraishi — ICML 2025). This is our primary prior art.

## What They Did

Trained TopK SAEs on the residual stream of ESM-2 (650M params, 33 layers) to extract interpretable features from a protein language model. Built InterProt visualizer to interpret latents. Used linear probes on SAE features for downstream tasks.

## SAE Architecture

- **Type**: TopK SAE (not ReLU-based — TopK gives explicit control over L0 sparsity)
- **Encoder**: `z = TopK(W_enc(x - b_pre))`
- **Decoder**: `x_hat = W_dec @ z + b_pre`
- **Loss**: Reconstruction MSE: `L = ||x - x_hat||_2^2`
- **TopK activation**: Only the k largest latents are non-zero; rest zeroed out
- **Key advantage over ReLU SAEs**: Directly set L0-norm of encoding via k, better reconstruction at same sparsity (Gao et al., 2024)

## Training Details

- **Training data**: 1M random sequences under 1022 residues from UniRef50
- **Hidden dimensions tested**: 2048, 4096, 8192, 16384 (expansion factors over ESM-2's hidden dim)
- **k values tested**: 16, 32, 64, 128, 256
- **Layers**: Trained separate SAEs on each of ESM-2's 33 layers
- **Best results for family-specific features**: middle layers, lower k (higher sparsity), larger hidden dim

## Key Hyperparameter Effects

### k (sparsity)
- Lower k = higher sparsity = more family-specific features
- At fixed hidden dim, reducing k increases the proportion of features that are family-specific
- Family-specific features are the most salient signals used in reconstruction

### Expansion factor (hidden dimension)
- Larger hidden dim increases number of family-specific features
- Activation pattern distribution (point, motif, domain, etc.) stays roughly consistent across sizes

### Layer choice
- Family-specific features peak in early-to-mid layers, then decline
- Long contiguous activation features (motif/domain) more common in earlier layers
- Later layers have shorter, more specific activations (specializing for final logit computation)

## Feature Classification Schemes

### By Activation Pattern
| Category | Criteria |
|----------|---------|
| Dead Latent | Never activated by any test sequence |
| Not Enough Data | <5 sequences activate it |
| Periodic | Regular intervals; >50% same distance between activations; >10 activation regions; short contigs |
| Point | Single prominent activation site (median length of highest activating region = 1) |
| Short Motif (1-20) | Contiguous region, median length 1-20, <80% mean activation coverage |
| Med Motif (20-50) | Contiguous region, median length 20-50, <80% mean activation coverage |
| Long Motif (50-300) | Contiguous region, median length 50-300, <80% mean activation coverage |
| Whole | >80% mean activation coverage |
| Other | Doesn't fit above categories |

### By Family Specificity
- For each latent, check: can its activations predict membership in a specific protein family?
- Label "family-specific" if F1 > 0.7 at some activation threshold
- Evaluation uses Swiss-Prot sequences clustered at 30% identity, categorized into InterPro families

## Interpretation Method

1. For a given SAE latent, find top-activating protein sequences
2. Visualize activations overlaid on protein structure (which residues activate?)
3. Align top-activating sequences to find shared motifs
4. Categorize by activation pattern + family specificity

## Linear Probing (Downstream Tasks)

They trained linear probes on SAE embeddings (mean-pooled for protein-level tasks) and compared to probes on raw ESM embeddings.

| Task | Type | Level | Metric | Implementation |
|------|------|-------|--------|---------------|
| Secondary structure | Classification | Residue | Accuracy | PyTorch linear classifier |
| Subcellular localization | Classification | Protein | Accuracy | Sklearn logistic regression |
| Thermostability | Regression | Protein | Spearman's rho | Sklearn ridge regression |
| CHO cell expression | Classification | Protein | Accuracy | Sklearn logistic regression |

**Key finding**: SAE probes perform competitively with ESM probes across all layers. For secondary structure, SAE consistently outperforms ESM.

## Key Findings for Our Project

1. **~80% of SAE features rated interpretable by humans** (vs much lower for raw ESM neurons)
2. **Family-specific latents exist** — many latents activate strongly only on proteins from a specific family
3. **Steering works**: clamping a family-specific latent to multiples of its max activation and continuing the forward pass changes fewer residues than steering random latents — family features capture evolutionary constraints
4. **Middle layers are richest** for family-specific features
5. **SAE features can discover unknown biology** — some predictive features don't match known concepts

## Differences from Our Setting

| InterProt | Our Project |
|-----------|------------|
| ESM-2 (protein LM, masked LM objective) | MetaGene-1 (metagenomic LM) |
| Single protein sequences | Metagenomic reads (mixed organisms) |
| Protein families as ground truth | Pathogen/organism taxonomies as ground truth |
| UniRef50 training data for SAE | Metagenomic sequences for SAE |
| Known protein structure for visualization | No single-sequence structure prediction |

## Code Structure (vendor/interprot/interprot/)

The SAE training pipeline we'll adapt:

```
interprot/
  sae_model.py       # SparseAutoencoder class (TopK SAE implementation)
  sae_module.py       # PyTorch Lightning module wrapping SAE + ESM-2
  esm_wrapper.py      # ESM2Model: loads ESM-2, extracts layer activations
  data_module.py      # SequenceDataModule: loads parquet of protein sequences
  training.py         # CLI training script (argparse + PL Trainer)
  validation_metrics.py  # diff_cross_entropy metric
  utils.py            # train/val/test split
```

### Key Implementation Details

- **SAE input**: ESM-2 residual stream at a chosen layer, shape `(batch, seq_len, 1280)`
- **Layer norm**: SAE applies its own LN before encoding (not relying on model's LN)
- **Dead neuron handling**: Auxiliary loss on dead neurons (auxk) per Gao et al. 2024
  - `dead_steps_threshold=2000`: neuron is "dead" if inactive for 2000 examples
  - Auxiliary top-k of `min(d_model/2, num_dead)` on dead neurons only
  - `auxk_coeff = 1/32`
- **Weight normalization**: decoder columns normalized after each backward pass
- **Gradient projection**: decoder gradients projected to remove component along decoder columns
- **Training**: AdamW, lr=2e-4, gradient clip 1.0, 1 epoch, log every 10 steps, val every 100 steps
- **Default hyperparams**: d_model=1280, d_hidden=16384, k=128, auxk=256, batch_size=48
- **Data**: parquet file with columns `id` and `sequence` (protein sequences from UniRef50)

### What We Need to Adapt

1. **Replace ESM-2 wrapper with MetaGene-1 wrapper** — need `get_layer_activations(sequences, layer_idx)` equivalent
2. **Change d_model from 1280 to 1536** (MetaGene-1's hidden dim)
3. **Replace protein sequence data with metagenomic reads**
4. **Handle autoregressive vs bidirectional** — ESM-2 is BERT-style, MetaGene-1 is causal. Activation extraction still works the same way (just hook into residual stream), but feature interpretation differs
5. **Tokenizer**: ESM-2 uses amino acid alphabet; MetaGene-1 uses 1024-token BPE on nucleotides

## External Resources

- **InterProt code**: github.com/etowahadams/interprot (cloned to `vendor/interprot`)
- **Pre-trained SAE weights**: huggingface.co/liambai/InterProt-ESM2-SAEs
- **Visualizer**: interprot.com
