# Metageniuses

**Interpretable pandemic surveillance via Sparse Autoencoders on MetaGene-1.**

Apart Research AI x Bio Hackathon project. We train a Sparse Autoencoder on the residual stream of MetaGene-1 (a 7B metagenomic foundation model) to extract interpretable features that reveal what the model has learned about pathogens.

## Key Results

- **Linear probe**: 94.6% accuracy (0.892 MCC) predicting pathogen vs non-pathogen from SAE features alone
- **SAE health**: 31,965 alive features out of 32,768 (2.4% dead), well-trained
- **Sequence UMAP**: clear pathogen/non-pathogen separation with 49 sub-clusters
- **Feature clustering**: pathogen-enriched latents cluster spatially together
- **Organism detectors** (in progress): BLAST-based labeling of pathogen-specific latents

## Project Structure

```
metageniuses/
├── src/metageniuses/
│   ├── extraction/          # MetaGene-1 residual stream extraction pipeline
│   └── sae/                 # SAE model, training, encoding, analysis
├── experiments/             # Experiment scripts (linear probe, UMAP, clustering, etc.)
├── experiment_plans/        # Detailed specs for each experiment
├── future_experiments/      # Ideas that need GPU re-runs
├── results/                 # Experiment outputs (gitignored)
├── data/
│   ├── sae_model/           # Trained SAE weights + features.npy
│   ├── human_virus_*.jsonl  # Labeled sequence datasets
│   └── curated_sequences/   # Deduplicated forward-pass inputs
├── viz/                     # React frontend (feature explorer + experiment viz)
├── backend/                 # FastAPI backend serving experiment results
├── configs/extraction/      # Extraction run configs
├── papers/                  # InterProt, MetaGene-1, SURF papers
└── vendor/                  # InterProt + MetaGene-1 code (git submodules)
```

## Quickstart

### Run experiments
```bash
pip install numpy scipy scikit-learn matplotlib statsmodels umap-learn hdbscan

python experiments/linear_probe_pathogen.py
python experiments/sae_health_check.py
python experiments/sequence_umap.py
python experiments/feature_clustering.py
```

### Run Peyton's analysis pipeline
```bash
pip install -e .
metageniuses-analyze-sae \
  --dataset_jsonl data/human_virus_class1_labeled.jsonl \
  --activation_path data/sae_model \
  --output_dir results/analyze \
  --label_field source --positive_label 1
```

### Frontend + Backend
```bash
# Backend
cd backend && pip install -r requirements.txt && uvicorn app:app --reload

# Frontend (separate terminal)
cd viz && npm install && npm run dev
```

### Run tests
```bash
PYTHONPATH=src python3 -m unittest discover -s tests/extraction -p 'test_*.py'
python -m pytest tests/sae/
```

## Papers

- [InterProt](papers/InterProt.pdf) — SAE on ESM-2 protein language model (Adams et al., 2025)
- [MetaGene-1](papers/metagene-1.pdf) — Metagenomic foundation model (Liu et al., 2025)
- [SURF](papers/SURF_Paper.docx) — PBD family specificity via InterProt SAE (Liu & Rogers, 2025)

## Team

Mannat Jain, Peyton Jackson, Bridget Liu, Ciaran, Astrid
