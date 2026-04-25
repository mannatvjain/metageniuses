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
- [ ] Run full extraction on RunPod (cloud prod config: 4 layers, 20k sequences)

## Phase 3: SAE Training
- [ ] Adapt InterProt's SAE for MetaGene-1
  - [ ] Change `d_model` from 1280 to 4096
  - [ ] Keep TopK architecture, auxk dead neuron handling
  - [ ] Wire SAE input from `iter_layer_batches()` contract
  - [ ] Adjust hyperparameters: k, d_hidden, auxk (start with InterProt defaults, tune)
- [ ] Choose target layer(s) — start with middle layers (12-20), based on InterProt findings
- [ ] Train SAE on RunPod
- [ ] Evaluate: reconstruction MSE, diff cross-entropy, num dead neurons

## Phase 4: Feature Analysis & Interpretation
- [ ] Extract and classify learned features
  - [ ] By activation pattern (point, motif, domain, periodic, whole)
  - [ ] By specificity (does feature activate on specific organism/pathogen types?)
- [ ] Map features to known biology
  - [ ] Viral vs bacterial vs eukaryotic sequences
  - [ ] Known pathogen signatures
  - [ ] Gene function annotations (if available)
- [ ] Build simple visualization (activation heatmaps over sequences)
- [ ] Linear probes on SAE features for downstream classification (pathogen detection?)

## Phase 5: Writeup & Presentation
- [ ] Write hackathon submission
  - [ ] Motivation: interpretable pandemic surveillance
  - [ ] Method: SAE on MetaGene-1 (adapted from InterProt)
  - [ ] Results: discovered features, biological interpretation
- [ ] Prepare figures
- [ ] Present
