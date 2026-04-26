# Log

Reverse-chronological. Each session appends what changed, what's unfinished, what to pick up next.

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
