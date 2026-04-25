# Log

Reverse-chronological. Each session appends what changed, what's unfinished, what to pick up next.

### 2026-04-25
- Train/test split: train set is class 1 of `human_virus_infecting` (`data/human_virus_class1.jsonl`, 20k seqs), test set is class 2 (`data/human_virus_class2.jsonl`, 20k seqs)
- Bridget ran inference on the class 1 file (the label-stripped version Ciaran and Astrid prepped) using the cloud prod config
- Labels (`source`, `class`) exist in `data/raw_sources/human_virus_infecting.csv` and `data/curated_sequences/forward_pass_unique_sequences.jsonl` but were not included in the per-class JSONL files
- Created labeled versions: `data/human_virus_class1_labeled.jsonl` and `data/human_virus_class2_labeled.jsonl` with flat `source` and `class` fields (replaced origin's nested-metadata version `human_virus_class1_restored_labels.jsonl`)
- Merged origin/main, converted `vendor/interprot` and `vendor/metagene-pretrain` from embedded repos to git submodules
- Added project docs (CONTEXT.md, INTERPROT_REFERENCE.md, METAGENE_REFERENCE.md) and papers to repo
- Peyton has SAE training pipeline on `origin/modeling/sae` branch (not merged) — includes SAE model, trainer, config, feature encoding script, class 2 extraction config, pyproject.toml
- Next session: review Peyton's `modeling/sae` branch when ready to merge; run extraction with real weights on RunPod
