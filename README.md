# Sparse Autoencoder Interpretability of the METAGENE-1 Genomic Foundation Model

This package provides an implementation of the analyses described in *Sparse
Autoencoder Interpretability of the METAGENE-1 Genomic Foundation Model*
(AIxBio Hackathon, Apart Research, April 2026).

We train BatchTopK sparse autoencoders on the residual stream of
[METAGENE-1](https://huggingface.co/metagene-ai/METAGENE-1), a 7B-parameter
metagenomic foundation model, and study whether its internal features encode
biologically meaningful structure for human-infecting viral sequences. The
resulting sparse dictionary supports a 94.55%-accuracy linear probe for
pathogen versus non-pathogen classification, transfers to a held-out
sequencing delivery without retraining, and contains individual latents that
BLAST identifies as organism-specific detectors for *Human astrovirus*,
*Norovirus GI*, *Norovirus GII*, *Sapovirus GI*, *Mamastrovirus sp.*, and
*Astrovirus MLB1*.

- **Paper:** [`viz/public/paper.pdf`](viz/public/paper.pdf)
- **Interactive site:** https://mannatvjain.github.io/metageniuses
- **Model:** [`metagene-ai/METAGENE-1`](https://huggingface.co/metagene-ai/METAGENE-1) (Liu et al., 2025)

## Key Results

All numbers below are from the trained layer-32 SAE unless otherwise noted.

| Result | Value |
|---|---|
| SAE alive latents | 31,965 / 32,768 (2.45% dead) |
| Mean nonzero latents per sequence (after mean pooling) | 892.10 |
| Linear probe (mean-pooled features, 80/20 split) | 94.55% acc, MCC 0.8916, AUROC 0.9874 |
| Latents covering 50% / 90% of probe coefficient mass | 2,068 / 10,847 |
| Pathogen-enriched latents (FDR < 0.01, OR > 1) | 16,519 |
| Non-pathogen-enriched latents | 2,534 |
| BLAST-validated organism detectors (layer 32) | 12 high-confidence + 15 medium-confidence |
| Organism detectors (layer 16, broader taxonomy) | 30 high-confidence + 13 medium-confidence |
| Cross-delivery probe (train Class 1, test Class 2) | 93.96% acc, AUROC 0.9840 |
| Per-latent enrichment Spearman across deliveries | 0.79 (all latents), 0.87 (significant) |
| Probe AUROC at layer 8 / 16 / 24 / 32 | 0.9912 / 0.9906 / 0.9914 / 0.9874 |

The high-confidence layer-32 detectors all had zero false-positive activations
on non-pathogen sequences under the scan threshold and BLAST mean percent
identity between 97.7% and 99.9%.

## Repository structure

```
metageniuses/
  src/metageniuses/
    extraction/           Residual-stream extraction pipeline for METAGENE-1
    sae/                  BatchTopK SAE: model, training, encoding, analysis
  experiments/            Analysis scripts (probe, UMAP, clustering, BLAST, ...)
  experiment_plans/       Per-experiment specs (numbered 01-09 + master list)
  future_experiments/     GPU-blocked follow-ups (token-level, multi-class)
  configs/extraction/     Extraction run configs (tiny test, smoke, cloud prod)
  data/
    curated_sequences/    Deduplicated forward-pass inputs
    human_virus_*.jsonl   Class 1 / 2 / 3 / 4 viral datasets (20,000 each)
    hmpd_*.jsonl          Human Microbiome Project datasets
    sae_model/            Layer-32 SAE config + training curves (large
                          artifacts are gitignored, see "Reproducing")
    sae_layer{8,16,24}/   Per-layer SAE features (gitignored)
  viz/                    React 19 + Vite 7 interactive site
  backend/                Optional FastAPI server (see "Backend")
  paper/                  Paper figures and helper scripts
  papers/                 Reference PDFs (InterProt, METAGENE-1, SURF)
  vendor/                 InterProt and METAGENE-1 source as git submodules
  tests/                  Unit and integration tests
```

A full file map lives in [INDEX.md](INDEX.md). Open work and the project
history live in [PLAN.md](PLAN.md) and [LOG.md](LOG.md).

## Installation

The analysis scripts require Python 3.10 or newer.

```bash
git clone --recurse-submodules https://github.com/mannatvjain/metageniuses
cd metageniuses
pip install -r requirements.txt
```

The extraction pipeline and SAE trainer have additional dependencies (PyTorch,
transformers, the METAGENE-1 weights). Install those via the project package:

```bash
pip install -e .
```

## Quickstart

### 1. Browse the interactive site (no install)

The deployed site is the simplest way to see the results:
https://mannatvjain.github.io/metageniuses

To run the same site locally:

```bash
cd viz
npm install
npm run dev
```

The viz reads its data from static JSONs in `viz/public/data/`, so it works on
a fresh clone with no backend, no model weights, and no GPU.

### 2. Reproduce the analysis figures

The `experiments/` scripts run end-to-end on the trained SAE features. They
expect the layer-32 SAE artifacts under `data/sae_model/`:

- `sae_final.pt` (~1 GB, gitignored)
- `features.npy` (~2.4 GB, gitignored)
- `sequence_ids.json` (gitignored)
- `sae_config.json` (tracked)
- `sae_training_curves.png` (tracked)

These are not committed because of size. Contact the authors to obtain a copy
or set `METAGENIUSES_SAE_DIR` to point at an existing local directory:

```bash
export METAGENIUSES_SAE_DIR=/path/to/sae_model

python experiments/sae_health_check.py
python experiments/linear_probe_pathogen.py
python experiments/sequence_umap.py
python experiments/feature_clustering.py
python experiments/organism_detectors.py    # requires NCBI BLAST API access
python experiments/cross_delivery.py        # requires features_class2.npy
```

Outputs land in `results/<experiment_name>/`.

### 3. Run the unified analysis CLI

`metageniuses-analyze-sae` is the analysis CLI that produced the per-layer
outputs in the paper (enrichment, probe, k-mer, differential signature, and
projection plots):

```bash
metageniuses-analyze-sae \
  --dataset_jsonl data/human_virus_class1_labeled.jsonl \
  --activation_path data/sae_model \
  --output_dir results/analyze \
  --label_field source --positive_label 1
```

### 4. Re-train the SAE from scratch (GPU required)

End-to-end training requires a GPU node and the METAGENE-1 weights. See
[`docs/architecture/runpod_setup.md`](docs/architecture/runpod_setup.md) for
the configuration we used. The high-level flow is:

```bash
# 1. Extract residual-stream activations
PYTHONPATH=src python -m metageniuses.extraction.cli \
  --config configs/extraction/metagene-cloud-prod.json

# 2. Train a BatchTopK SAE on the resulting activations
metageniuses-train-sae --config configs/sae/layer32.json   # adjust per layer

# 3. Encode all sequences through the trained SAE
metageniuses-encode-features --sae path/to/sae_final.pt
```

The published configuration is Adam, learning rate 2e-4, batch size 4096
token activations, 10 epochs, expansion factor 8, k = 64, with an auxiliary
dead-feature loss weighted at 0.03125.

## Backend

A FastAPI server in [`backend/`](backend) is available for setups where the
viz should pull from a live API rather than the bundled static JSONs:

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

The server reads from the gitignored `results/` directory, so it only returns
data after you have run the experiment scripts. The deployed site does not
use it. The Vite dev server proxies `/api/*` to `http://localhost:8000` if
you want to wire the two together during local development.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests/extraction -p 'test_*.py'
python -m pytest tests/sae/
```

The extraction tests use a fake model adapter and run without GPU or model
downloads. The SAE analysis tests require a small fixture; see
[`tests/sae/test_analyze.py`](tests/sae/test_analyze.py) for details.

## Citing this work

If you use this code or build on these analyses, please cite the paper. Author
list and BibTeX entry are on the title page of
[`viz/public/paper.pdf`](viz/public/paper.pdf).

## Acknowledgements

This project builds directly on
[InterProt](https://github.com/etowahadams/interprot) (Adams et al., 2025),
which applied SAEs to the protein language model ESM-2. We use METAGENE-1
(Liu et al., 2025) as the target foundation model. We thank Apart Research
for hosting the AIxBio Hackathon.

## License

Released under the [MIT License](LICENSE).
