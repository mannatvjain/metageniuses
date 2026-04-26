# Plan

## Phase 0: Setup
- [x] Scaffold project structure
- [x] Write CONTEXT.md
- [x] Read InterProt paper, extract methodology into reference doc
- [x] Clone InterProt code (`vendor/interprot`)
- [x] Clone MetaGene-1 pretrain code (`vendor/metagene-pretrain`)
- [x] Document MetaGene-1 architecture (32 layers, d_model=4096, 1024 BPE vocab — per paper Table 1)
- [x] Model weights accessible at `metagene-ai/METAGENE-1` on HuggingFace
- [ ] Set up RunPod environment with GPU (runbook at `docs/architecture/runpod_setup.md`)

## Phase 1: MetaGene-1 Wrapper
- [x] Write model adapter for residual extraction (`src/metageniuses/extraction/model_adapter.py`)
  - [x] `TransformersModelAdapter`: loads via HF `AutoModelForCausalLM`, extracts hidden states
  - [x] `FakeModelAdapter`: deterministic adapter for local tests without GPU/downloads
  - [x] `extract_batch(sequences, transformer_layers, max_length)` → token IDs + hidden states by layer
  - [x] Auto dtype/device resolution (bf16 on CUDA, fp32 on CPU)
- [x] Configurable layer selection (`LayerSelectionConfig`: explicit layers or last_n_layers)
- [x] CLI entry point (`metageniuses.extraction.cli`)
- [ ] Test with real MetaGene-1 weights on RunPod (smoke config ready: `configs/extraction/metagene-smoke-local.json`)

## Phase 2: Data Pipeline
- [x] Collect/curate metagenomic sequences
  - [x] Human virus datasets (classes 1-4, 20k sequences each)
  - [x] Human microbiome datasets (disease, sex, source, ~7.8k each)
  - [x] HVR default dataset (370 sequences)
  - [x] Curated forward-pass dataset (~85k unique sequences, deduplicated)
- [x] Build extraction pipeline (`src/metageniuses/extraction/`)
  - [x] Input I/O: JSONL + FASTA readers
  - [x] Preprocessing: uppercase, invalid char replacement, length filtering
  - [x] Sharded activation storage (binary f32 + JSONL index per layer)
  - [x] Resume support for interrupted runs
  - [x] Manifest + contracts for downstream SAE consumption
  - [x] `iter_layer_batches()` loader for SAE training
- [x] Configs: tiny-test, local smoke, cloud prod (`configs/extraction/`)
- [x] Tests: contracts, pipeline (fake adapter), preprocessing, resume
- [x] Run full extraction on RunPod (cloud prod config: 4 layers, 20k sequences)

## Phase 3: SAE Training
- [x] Adapt InterProt's SAE for MetaGene-1
  - [x] Change `d_model` from 1280 to 4096
  - [x] Keep TopK architecture, auxk dead neuron handling
  - [x] Wire SAE input from `iter_layer_batches()` contract
  - [x] Adjust hyperparameters: k=64, d_hidden=32768, expansion=8x
- [x] Choose target layer — layer 32 (last layer)
- [x] Train SAE on RunPod (10 epochs, batch_size=4096, lr=0.0002)
- [x] Evaluate: reconstruction MSE converged ~5e-5, dead features recovered to ~0%
- [x] SAE artifacts saved: `data/sae_model/` (sae_final.pt, features.npy, sae_config.json, training curves)

## Phase 4: Feature Analysis & Interpretation
- [x] SAE health check (experiment 3)
  - [x] 803 dead (2.4%), 31,965 alive, ~892 active features/sequence
  - [x] Activation distribution histograms, latent stats CSV
- [x] Linear probe — pathogen detection (experiment 2)
  - [x] 94.6% accuracy, 0.892 MCC, 0.987 AUROC on source label
  - [x] Top predictive latents identified (coefficients + activation distributions)
  - [x] Cumulative importance analysis: signal distributed across many latents
- [x] Sequence UMAP (experiment 4)
  - [x] Clear pathogen/non-pathogen separation in SAE feature space
  - [x] 49 sub-clusters via HDBSCAN, many near-pure for one class
- [x] Feature clustering (experiment 5)
  - [x] 12 clusters, pathogen-enriched latents cluster together (bottom-left blue → upper-right red gradient)
  - [x] 12,894 pathogen-enriched latents (log2 OR > 1), 2,451 pathogen-depleted
- [x] Peyton's analysis pipeline merged (PR #1)
  - [x] Fisher's exact test + BH-FDR enrichment
  - [x] Linear probe with AUROC/AUPRC/F1
  - [x] K-mer enrichment analysis
  - [x] Differential signature (mean pathogen - mean non-pathogen vector)
  - [x] Volcano plot, PCA/t-SNE/UMAP projections
- [ ] Organism-specific pathogen detectors (experiment 1) — THE MAIN RESULT
  - [ ] Enrichment scan → top pathogen-specific latents
  - [ ] Pull top-activating sequences per latent
  - [ ] BLAST against NCBI → organism identification
  - [ ] Label latents: "Latent X is an Influenza A detector"
- [ ] Cross-delivery generalization (experiment 6)
  - [ ] Encode class 2 sequences through SAE (needs GPU)
  - [ ] Test probe + enrichment stability across deliveries

## Phase 5: Writeup & Presentation
- [ ] Write hackathon submission
  - [ ] Motivation: interpretable pandemic surveillance
  - [ ] Method: SAE on MetaGene-1 (adapted from InterProt)
  - [ ] Results: organism-specific pathogen detector features + BLAST evidence
- [ ] Prepare figures
- [ ] Present
