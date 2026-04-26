# Log

Reverse-chronological. Each session appends what changed, what's unfinished, what to pick up next.

### 2026-04-26 (session 6 — experiment execution)
- **Experiment 1 (organism detectors) — COMPLETE**
  - Part A: enrichment scan across 32,768 latents — 16,519 pathogen-enriched (Fisher FDR<0.01, OR>1), 2,534 non-pathogen-enriched, 4 pathogen-specific (F1>0.7), 19,533 Wilcoxon significant
  - Part B: selected top 50 latents (4 specific + 46 enriched), pulled top 10 pathogen sequences per latent (500 total)
  - Part C: BLAST all 500 sequences against NCBI nt — ~100% hit rate. Parallelized submission with round-robin polling.
  - Part D: 12 high-confidence + 15 medium-confidence organism labels. Top detectors: Human astrovirus (multiple latents, 99%+ identity), Norovirus GI (10/10 consistency), Norovirus GII (7/10), Human adenovirus (5/10), Sapovirus
  - Part E: volcano plot, organism detector bar chart, enrichment histogram
  - Bugs fixed during run: NCBI returns ZIP not raw JSON for JSON2 format; BlastOutput2 can be dict or list; IncompleteRead not caught by retry logic; inf ORs break matplotlib histogram
  - Script: `experiments/organism_detectors.py` (1049 lines, CLI with `--parts` and `--blast-test` flags, checkpoint/resume)
- **Experiment 6 (cross-delivery) — COMPLETE**
  - Probe: 93.96% acc on class 2 (vs 94.76% train), -0.8% delta. AUROC 0.984. Features generalize.
  - Feature stability: Spearman r=0.79 (all latents), 0.87 (significant only). 10,510 latents significant in both classes.
  - Outputs: summary.json, comparison_table.md, enrichment_scatter.png, roc_curve_class2.png, feature_stability.csv
  - Bugs fixed: ProcessPoolExecutor spawn error (no `__main__` guard) → replaced with simple loop; n_jobs=1 for sklearn (disk); added RAM monitor thread (warn 100GB, kill 125GB)
- Deleted redundant zips: `sae_final.pt.zip` (941M) and `features.npy.zip` (104M) — freed ~1GB
- Cleaned up 5 stale worktrees (~900MB)
- **Lesson learned**: Don't print verbose output to terminal for long-running jobs — redirect to log files. Ghostty buffers all scrollback in RAM.
- **Next up**: Finish Exp 6, update docs with results. Then Phase 5 writeup.

### 2026-04-26 (session 5 — multi-layer planning)
- Reviewed full SAE training pipeline (`src/metageniuses/sae/model.py`, `train.py`, `config.py`, `encode_features.py`) — confirmed it supports arbitrary layers with zero code changes. Just pass `--layer N`.
- Confirmed existing extraction only covers layers 29-32 (`last_n_layers: 4` in cloud prod config). Layers 8, 16, 24 need fresh extraction on RunPod.
- Peyton is running the multi-layer extraction + SAE training + per-token encoding on RunPod now.
- Wrote 3 detailed experiment plans for post-multi-layer-data work:
  - `experiment_plans/07_layer_wise_probe_comparison.md` — SAE vs raw activation probes at layers 8, 16, 24, 32. Replicates InterProt Section 4.3 / Table 3. ~10 min local.
  - `experiment_plans/08_activation_pattern_classification.md` — classify all 32k latents × 4 layers as point/motif/periodic/whole. Replicates InterProt Table 2. ~30 min local. Requires per-token sparse activations.
  - `experiment_plans/10_token_level_pathogen_localization.md` — per-token pathogen scoring via probe_coef · SAE_activations, BLAST hotspot subsequences. Hero figure for paper. ~30 min local + BLAST.
- All plans include "Instructions for Codex" sections since Peyton uses Codex, not Claude.
- Per-token data format specified: scipy sparse CSR `.npz` + `token_metadata.jsonl` per layer, ~1.5 GB/layer, ~6 GB total for 4 layers.
- Created `future_experiments/virus_species_classifier.md` — multi-class virus classifier from SAE features, targeting ICML (not hackathon). Requires BLASTing all 10k pathogen sequences for species labels.
- Set Signal reminders for Mannat: (1) run middle/early layer extraction + SAE vs MetaGene linear probe, (2) save per-token activations during RunPod session.
- Updated `experiment_plans/EXPERIMENTS.md` with Exp 7, 8, 10 descriptions and updated priority table with completion status.
- Updated `INDEX.md` with new plan files + virus species classifier future experiment.
- Updated `PLAN.md` with new Phase 4b (multi-layer analysis).
- **Unfinished**: Experiments 1 and 6 scripts still need to be run (from session 4). Experiments 7, 8, 10 blocked on Peyton's RunPod data.
- **Next up**: (1) Run Exp 1 (organism detectors) and Exp 6 (cross-delivery) — scripts exist, just need execution with proper timeouts. (2) Once Peyton delivers multi-layer data, run Exp 7, 8, 10. (3) Phase 5 writeup.
- **Plan impact**: Added Phase 4b (multi-layer analysis) to PLAN.md with 9 new items.

### 2026-04-26 (session 4)
- Received class 2 SAE features from Bridget: `data/sae_model/features_class2.npy` (20k x 32768) + `sequence_ids_class2.json` — unblocks Experiment 6
- Attempted to run Experiments 1 and 6 via worktree agents — both failed due to 2-min bash timeout on long-running enrichment scans (32k Fisher tests + F1 sweeps)
- Agents wrote complete scripts before dying:
  - `experiments/organism_detectors.py` (955 lines) — in worktree `agent-a142044a`, not yet copied to main
  - `experiments/cross_delivery.py` (416 lines) — in worktree `agent-abf05bf8`, not yet copied to main
- Neither script has been successfully run yet — no results produced
- **Next up**: Copy scripts from worktrees to main, review for correctness, run with proper timeouts (10 min). Exp 1 Parts A+B are local (~3-5 min), Part C is BLAST (~15-30 min). Exp 6 probe fitting may need PCA or saga solver to avoid timeout on 32k features.
- **Lesson learned**: Worktree agents need explicit `timeout=600000` on bash calls for compute-heavy scripts. Default 2-min timeout is insufficient for 32k-latent enrichment scans.

### 2026-04-26 (session 3 — experiment blitz)
- Reviewed all three papers (InterProt, MetaGene-1, SURF/Bridget's PBD paper) to map analyses onto our project
- Wrote 6 experiment plans in `experiment_plans/01-06_*.md` + master list `EXPERIMENTS.md`
- Ran 4 experiments in parallel via worktree agents:
  - **Exp 2 (linear probe)**: 94.6% accuracy, 0.892 MCC, 0.987 AUROC on pathogen detection. Top latents have high coefficients but enrichment ~1.0 — they fire on nearly all sequences with subtle magnitude differences. Code: `experiments/linear_probe_pathogen.py`, `experiments/probe_visualizations.py`
  - **Exp 3 (SAE health check)**: 803 dead (2.4%), 31,965 alive. ~892 active features/sequence (aggregated across tokens, not raw TopK). Code: `experiments/sae_health_check.py`
  - **Exp 4 (sequence UMAP)**: clear pathogen/non-pathogen separation. 49 HDBSCAN sub-clusters, many pure. Sub-clusters within pathogen class may correspond to different organisms. Code: `experiments/sequence_umap.py`
  - **Exp 5 (feature clustering)**: 12 clusters of latents. Pathogen-enriched latents cluster together spatially in UMAP. 12,894 pathogen-enriched (log2 OR > 1), 2,451 depleted. Code: `experiments/feature_clustering.py`
- Merged Peyton's SAE analysis pipeline (PR #1): `src/metageniuses/sae/analyze.py` — Fisher's exact test, linear probe, k-mer enrichment, differential signature, volcano plot, projections
- Updated PLAN.md: marked Phase 2 extraction and Phase 3 SAE training as complete, detailed Phase 4 progress
- **Key insight**: linear probe top latents have enrichment ~1.0 (fire on everything). The enrichment analysis (Fisher's/Wilcoxon) finds different latents — ones with skewed firing frequency. Both lenses are complementary.
- **Unfinished**: Experiment 1 (organism-specific pathogen detectors via BLAST) — the main result. All statistical infrastructure is ready; need to run enrichment scan → pull top sequences → BLAST against NCBI.
- **Also unfinished**: Experiment 6 (cross-delivery generalization) — blocked on encoding class 2 sequences through the SAE (needs GPU)
- Fleshed out Experiment 1 spec (`experiment_plans/01_organism_detectors.md`) with full implementation detail: BLAST API mechanics, batching/checkpointing, organism labeling rules, edge cases, exact output formats
- **Next up**: Implement and run Experiment 1 (organism detectors). Then write up results for hackathon submission.

### 2026-04-25 (session 2)
- Discussed Phase 4 approach: what analyses to run on trained SAE (feature classification, linear probing, visualization)
- Wrote experiment plan: feature labeling MVP — rank top 100 features by specificity, BLAST activating subsequences via NCBI API, LLM-synthesize labels (~$2 total cost)
- Wrote experiment plan: activation pattern classification — categorize all latents as point/motif/periodic/whole/dead with nucleotide-recalibrated thresholds
- Ran Munger-style inversion on agent labeling at scale — concluded: skip mass labeling, focus on top 100 with BLAST grounding
- Committed both plans to `experiment_plans/` on `ui/feature-explorer` branch
- **Unfinished**: no code written yet for Phase 4 — plans only
- **Next up**: get SAE weights + extraction outputs from Bridget/Peyton; review Peyton's `modeling/sae` branch; start implementing activation pattern classification (cheapest experiment, $0, feeds into labeling MVP)
- **Plan impact**: no PLAN.md changes yet — Phase 3 status unclear (Bridget says SAE is trained but PLAN.md still shows unchecked items; confirm with team what's done before updating)

### 2026-04-25
- Train/test split: train set is class 1 of `human_virus_infecting` (`data/human_virus_class1.jsonl`, 20k seqs), test set is class 2 (`data/human_virus_class2.jsonl`, 20k seqs)
- Bridget ran inference on the class 1 file (the label-stripped version Ciaran and Astrid prepped) using the cloud prod config
- Labels (`source`, `class`) exist in `data/raw_sources/human_virus_infecting.csv` and `data/curated_sequences/forward_pass_unique_sequences.jsonl` but were not included in the per-class JSONL files
- Created labeled versions: `data/human_virus_class1_labeled.jsonl` and `data/human_virus_class2_labeled.jsonl` with flat `source` and `class` fields (replaced origin's nested-metadata version `human_virus_class1_restored_labels.jsonl`)
- Merged origin/main, converted `vendor/interprot` and `vendor/metagene-pretrain` from embedded repos to git submodules
- Added project docs (CONTEXT.md, INTERPROT_REFERENCE.md, METAGENE_REFERENCE.md) and papers to repo
- Peyton has SAE training pipeline on `origin/modeling/sae` branch (not merged) — includes SAE model, trainer, config, feature encoding script, class 2 extraction config, pyproject.toml
- Next session: review Peyton's `modeling/sae` branch when ready to merge; run extraction with real weights on RunPod
