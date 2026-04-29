# Index

## Documentation
| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project instructions, architecture, agent protocol |
| `INDEX.md` | This file — map of the repository |
| `PLAN.md` | Task checklist with progress tracking |
| `LOG.md` | Reverse-chronological session changelog |
| `CONTEXT.md` | Problem domain, approach, key concepts, team |
| `INTERPROT_REFERENCE.md` | Methodology and patterns from the InterProt paper + code |
| `METAGENE_REFERENCE.md` | MetaGene-1 architecture, tokenizer, and code structure |
| `README.md` | Project overview, structure, quickstart |
| `docs/architecture/residual_extraction.md` | Extraction component contract and design |
| `docs/architecture/runpod_setup.md` | RunPod deployment/run instructions |
| `docs/datasets/forward_pass_dataset.md` | How raw uploads were organized and curated |
| `docs/datasets/data_sources.docx` | Original data source documentation |

## Code — Extraction pipeline
| File | Purpose |
|------|---------|
| `src/metageniuses/extraction/__init__.py` | Extraction module exports |
| `src/metageniuses/extraction/cli.py` | CLI entry point for extraction |
| `src/metageniuses/extraction/config.py` | Dataclass configs (input, preprocess, model, layers, runtime) |
| `src/metageniuses/extraction/contracts.py` | SAE-facing loader: `iter_layer_batches()`, `load_manifest()` |
| `src/metageniuses/extraction/extractor.py` | `ResidualExtractionPipeline` — main orchestrator |
| `src/metageniuses/extraction/input_io.py` | JSONL + FASTA sequence readers |
| `src/metageniuses/extraction/model_adapter.py` | `ModelAdapter` ABC, `TransformersModelAdapter`, `FakeModelAdapter` |
| `src/metageniuses/extraction/preprocess.py` | Sequence cleaning + validation |
| `src/metageniuses/extraction/schemas.py` | Shared dataclasses (SequenceRecord, ModelDescription, etc.) |
| `src/metageniuses/extraction/storage.py` | Sharded activation writer (`ActivationStore`, `_LayerWriter`) |

## Code — SAE training & analysis
| File | Purpose |
|------|---------|
| `src/metageniuses/sae/__init__.py` | SAE module exports |
| `src/metageniuses/sae/model.py` | TopK SAE model definition |
| `src/metageniuses/sae/train.py` | SAE training loop (`metageniuses-train-sae` CLI) |
| `src/metageniuses/sae/config.py` | SAE training config dataclasses |
| `src/metageniuses/sae/encode_features.py` | Encode sequences through trained SAE → features.npy |
| `src/metageniuses/sae/plot_training.py` | Plot training loss curves |
| `src/metageniuses/sae/analyze.py` | Full analysis CLI: enrichment, probe, k-mer, differential signature (Peyton's) |

## Code — Backend API
| File | Purpose |
|------|---------|
| `backend/app.py` | FastAPI server: serves experiment results + feature explorer data |
| `backend/dummy_data.py` | Dummy data generators for when real results aren't available |
| `backend/requirements.txt` | Backend Python dependencies |

## Tests
| File | Purpose |
|------|---------|
| `tests/extraction/test_contracts.py` | Contract tests for `iter_layer_batches` |
| `tests/extraction/test_pipeline_fake.py` | End-to-end pipeline test with fake adapter |
| `tests/extraction/test_preprocess.py` | Preprocessing unit tests |
| `tests/extraction/test_resume.py` | Resume/interrupted-run tests |
| `tests/fixtures/tiny_reads.jsonl` | Tiny test data fixture |
| `tests/sae/test_analyze.py` | Tests for Peyton's analyze.py |

## Configs
| File | Purpose |
|------|---------|
| `configs/extraction/default.json` | Baseline config (curated sequences, `results/extraction/`) |
| `configs/extraction/tiny-test.json` | Local no-download test config |
| `configs/extraction/metagene-smoke-local.json` | Real-weights local smoke run |
| `configs/extraction/metagene-cloud-prod.json` | Cloud production: 4 layers, 20k seqs |
| `configs/extraction/metagene1.json` | MetaGene-1 legacy test config |
| `configs/extraction/metagene-local-small.json` | Small local extraction config |

## Data
| File | Purpose |
|------|---------|
| `data/curated_sequences/forward_pass_unique_sequences.jsonl` | ~85k deduplicated sequences (default extraction input) |
| `data/curated_sequences/forward_pass_all_rows.jsonl` | All rows before dedup (~112k) |
| `data/curated_sequences/forward_pass_summary.json` | Curation stats |
| `data/human_virus_class{1-4}.jsonl` | Human virus datasets, no labels (20k each) |
| `data/human_virus_class1_labeled.jsonl` | Class 1 with `source` + `class` labels (train set) |
| `data/human_virus_class2_labeled.jsonl` | Class 2 with `source` + `class` labels (test set) |
| `data/hmpd_{disease,sex,source}.jsonl` | Human microbiome project datasets |
| `data/hvr_default.jsonl` | HVR default dataset (370 seqs) |
| `data/raw_sources/*.csv` | Original uploaded CSVs |
| `data/sae_model/sae_config.json` | SAE hyperparameters (tracked) |
| `data/sae_model/sae_training_curves.png` | Training loss + dead feature % plot (tracked) |
| `data/sae_model/sae_final.pt` | Trained SAE weights, layer 32, d_model=4096, 32,768 latents, k=64 (gitignored, ~1 GB) |
| `data/sae_model/features.npy` | (20k x 32,768) sequence-level SAE activations for class 1 (gitignored) |
| `data/sae_model/features_class2.npy` | (20k x 32,768) sequence-level SAE activations for class 2 (gitignored) |
| `data/sae_model/sequence_ids.json` | Maps features.npy row index to sequence_id (gitignored) |
| `data/sae_layer{8,16,24}/features.npy` | Per-layer sequence-level SAE features for the multi-layer comparison (gitignored) |

## Vendor (cloned repos)
| Directory | Purpose |
|-----------|---------|
| `vendor/interprot/` | InterProt SAE training code + visualizer (submodule: etowahadams/interprot) |
| `vendor/metagene-pretrain/` | MetaGene-1 pretraining code (submodule: metagene-ai/metagene-pretrain) |

## Experiment Plans
| File | Purpose |
|------|---------|
| `experiment_plans/EXPERIMENTS.md` | Master experiment list with priorities and status |
| `experiment_plans/01_organism_detectors.md` | Enrichment scan + BLAST to find organism-specific pathogen detector latents |
| `experiment_plans/02_linear_probe.md` | Logistic regression on SAE features for pathogen classification |
| `experiment_plans/03_sae_health_check.md` | Descriptive stats on SAE: dead/alive, sparsity, activation distributions |
| `experiment_plans/04_sequence_umap.md` | UMAP projection of sequences colored by pathogen label |
| `experiment_plans/05_feature_clustering.md` | Cluster latents by co-activation, check if pathogen features group together |
| `experiment_plans/06_cross_delivery.md` | Train on class 1, test on class 2 (blocked on class 2 features) |
| `experiment_plans/07_layer_wise_probe_comparison.md` | SAE vs raw activation probes at layers 8, 16, 24, 32 |
| `experiment_plans/08_activation_pattern_classification.md` | Classify latents as point/motif/periodic/whole across layers |
| `experiment_plans/09_token_level_pathogen_localization.md` | Per-token pathogen scoring + BLAST on hotspot subsequences |
| `experiment_plans/feature_labeling_mvp.md` | Top-100 feature labeling via specificity ranking + BLAST + LLM synthesis |

## Future Experiments (need GPU)
| File | Purpose |
|------|---------|
| `future_experiments/token_level_activations.md` | Re-run SAE encoder to get per-token activations (unlocks activation patterns, precise BLAST, heatmaps) |
| `future_experiments/virus_species_classifier.md` | Multi-class virus species classifier from SAE features (target: ICML, not hackathon) |

## Experiments (code + results)
| File | Purpose |
|------|---------|
| `experiments/linear_probe_pathogen.py` | Linear probe: logistic regression on SAE features → pathogen detection |
| `experiments/probe_visualizations.py` | Activation distributions + cumulative coefficient importance plots |
| `experiments/sae_health_check.py` | Dead/alive census, sparsity stats, activation distribution figures |
| `experiments/sequence_umap.py` | PCA → UMAP → scatter plot colored by pathogen label + HDBSCAN |
| `experiments/feature_clustering.py` | Transpose features → PCA → UMAP → HDBSCAN on 32k latents |
| `experiments/organism_detectors.py` | Enrichment scan + BLAST + organism labeling (Exp 1, the main result) |
| `experiments/cross_delivery.py` | Train probe on class 1, test on class 2 (Exp 6) |

## Results (gitignored, generated by experiment scripts)
| Directory | Contents |
|-----------|---------|
| `results/linear_probe_pathogen/` | Layer 32: 94.55% acc, 0.8916 MCC, 0.9874 AUROC. ROC curve, coefficient plots, top latents JSON, api_results.json |
| `results/sae_health_check/` | 803 dead, 31,965 alive. Activation histograms, latent stats CSV, api_results.json |
| `results/sequence_umap/` | UMAP scatter (clear separation), PCA variance, 49 sub-clusters, api_results.json |
| `results/feature_clustering/` | Latent UMAP colored by cluster/enrichment/activation count, api_results.json |
| `results/organism_detectors/` | Layer 32: 12 high-conf + 15 medium-conf organism detectors. BLAST results, enrichment CSV, volcano plot, bar chart |
| `results/organism_detectors_layer16/` | Layer 16: 30 high-conf + 13 medium-conf organism detectors |
| `results/cross_delivery/` | Cross-delivery probe generalization (class 1 -> class 2): 93.96% acc, AUROC 0.9840 |
| `results/analyze_layer{8,16,24,32}/` | Peyton's pipeline outputs (probe, k-mer, differential signature, projections) per layer |

## Viz (Frontend)
| File | Purpose |
|------|---------|
| `viz/package.json` | React 19 + Vite 7 + Tailwind 4 + Lucide + Recharts |
| `viz/vite.config.js` | Vite config (base `/metageniuses/`, dev proxy to backend) |
| `viz/public/data/*.json` | Static result payloads consumed by the live viz routes |
| `viz/public/paper.pdf` | Hackathon paper served from the landing page |
| `viz/src/main.jsx` | React entry point |
| `viz/src/App.jsx` | Top-level router: landing + four experiment pages |
| `viz/src/LandingPage.jsx` | Hero with rotating words, stats bar, feature cards |
| `viz/src/components/CanvasScatter.jsx` | Canvas-based scatter component |
| `viz/src/components/LoadingState.jsx` | Loading/error state component |
| `viz/src/hooks/useApi.js` | Maps `/api/...` URLs to static JSONs in `viz/public/data/` |
| `viz/src/pages/ExperimentsLayout.jsx` | Shared layout wrapper for experiment pages |
| `viz/src/pages/Experiment1.jsx` | Organism detectors page (layer 32 + layer 16) |
| `viz/src/pages/EncodedEarly.jsx` | Multi-layer enrichment view (layers 8/16/24/32) |
| `viz/src/pages/PathogenicityVector.jsx` | Differential signature page |
| `viz/src/pages/SAEHealth.jsx` | SAE health page |

Unrouted scaffolding (kept in tree as a starting point for a future per-feature explorer; not reachable from the live app): `ExplorerView.jsx`, `Sidebar.jsx`, `FeaturePanel.jsx`, `DetailsPanel.jsx`, `SequenceStrip.jsx`.

## SAE Analysis (Peyton's pipeline)
| File | Purpose |
|------|---------|
| `src/metageniuses/sae/analyze.py` | Full SAE analysis CLI: enrichment, probe, k-mer, differential signature, plots |
| `tests/sae/test_analyze.py` | Tests for analyze.py |

## Paper Writeup
| File | Purpose |
|------|---------|
| `viz/public/paper.pdf` | The hackathon paper (also served by the landing page) |
| `paper/figures_interprot/*.png` | InterProt-style comparison figures (specific features by layer, AUROC by layer, enrichment symmetry, organism detector histogram) |
| `paper/make_interprot_figures.py` | Script that produces the four figures above |
| `paper/results/` | Local-only handoff package built for the Overleaf agent (gitignored) |

## Papers
| File | Purpose |
|------|---------|
| `papers/InterProt.pdf` | InterProt paper, SAE on protein language model (prior art) |
| `papers/metagene-1.pdf` | MetaGene-1 paper, the target metagenomic foundation model |
| `papers/SURF.pdf` | SURF paper, PBD family specificity analysis using InterProt SAE |
