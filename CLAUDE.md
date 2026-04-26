# Metageniuses

Project for Apart Research's AI x Bio Hackathon. Interpretable pandemic surveillance via Sparse Autoencoders on MetaGene-1.

## Stack
- **Language**: Python 3.10+
- **Key dependencies**: torch, transformers, scikit-learn, scipy, statsmodels, numpy, pandas, matplotlib, umap-learn, hdbscan
- **Frontend**: React 19 + Vite 7 + Tailwind 4 (in `viz/`)
- **Backend**: FastAPI (in `backend/`)

## Architecture

```
MetaGene-1 (7B transformer, layer 32)
        ↓ residual stream activations
    SAE (TopK, 32768 latents, k=64)
        ↓ sparse feature vectors
    Experiments (enrichment, probes, UMAP, BLAST)
        ↓
    Results → Backend API → Frontend visualizer
```

## Experiment organization
- `experiment_plans/` — plans for experiments we can run now with existing data
- `future_experiments/` — experiments that require GPU re-runs or new SAE inference (e.g., token-level activations, additional layer SAEs). Put any idea that needs RunPod / SAE encoder / new model inference here, not in `experiment_plans/`.
- `experiments/` — Python scripts that implement the experiment plans
- `results/` — output from experiments (gitignored — large CSVs, PNGs)

## Conventions
- **No self-attribution**: Claude is a tool, not a person. Never add Co-Authored-By lines or credit Claude as a co-author in commits or anywhere else.
- **Owner tracking**: This is a multi-person project. Every commit must include the name of the human who instructed the Claude agent (e.g., `Instructed by: Mannat Jain` in the commit message). Ask if unclear.
- **Don't touch Peyton's code**: `src/metageniuses/sae/analyze.py`, `tests/sae/test_analyze.py`, and `pyproject.toml` are maintained separately. Don't edit them unless coordinating with Peyton.

## Dev commands
```bash
# Run an experiment
python experiments/linear_probe_pathogen.py
python experiments/sae_health_check.py
python experiments/sequence_umap.py
python experiments/feature_clustering.py

# Run Peyton's analysis pipeline
metageniuses-analyze-sae --dataset_jsonl data/human_virus_class1_labeled.jsonl \
  --activation_path data/sae_model --output_dir results/analyze \
  --label_field source --positive_label 1

# Run extraction (needs GPU)
PYTHONPATH=src python3 -m metageniuses.extraction.cli --config configs/extraction/tiny-test.json --adapter fake

# Run tests
PYTHONPATH=src python3 -m unittest discover -s tests/extraction -p 'test_*.py'
python -m pytest tests/sae/

# Frontend
cd viz && npm install && npm run dev

# Backend
cd backend && pip install -r requirements.txt && uvicorn app:app --reload
```

---

## Agent Protocol

You are one of many short-lived Claude sessions working on this project. The user relies on Claude to write code — knowledge must transfer between sessions via docs, not memory. You will not remember prior conversations. The docs are your memory.

### Before starting work
1. Read `INDEX.md` to understand the repo layout.
2. Read `PLAN.md` to see what's done and what's next.
3. Read `LOG.md` (latest entry) to understand where the last session left off.
4. Read any `*_REFERENCE.md` files listed in `INDEX.md` before writing code that touches those domains. Do not guess at APIs or platform behavior — it's documented there for a reason.
5. Read `CONTEXT.md` if it exists — it describes the problem domain and any external systems.

### During work

#### Planning (required for non-trivial tasks)
Before making changes, write a short ASCII plan and show it to the user:

```
+-------------------------------------+
| Task: <short description>           |
+-------------------------------------+
| 1. <step>                           |
| 2. <step>                           |
|    - <substep>                      |
| 3. <step>                           |
+-------------------------------------+
```

Wait for confirmation before proceeding. Keep plans concise.

#### Recap (required after completing each action)
After completing work, show an ASCII recap:

```
+-------------------------------------+
| Recap: <short description>          |
+-------------------------------------+
| Files edited:                       |
|  * path/to/file                     |
|    - <what changed>                 |
| Insights saved:                     |
|  > REFERENCE_DOC.md                 |
|    - <what was documented>          |
+-------------------------------------+
```

#### Documenting new knowledge
When you learn something non-obvious — a platform gotcha, an API quirk, a pattern that works — write it to the appropriate `*_REFERENCE.md` file immediately. If no reference doc exists for that domain yet, create one and add it to `INDEX.md`. This is how you pass knowledge to the next session.

### Handoff (end of every conversation)
When the user runs `/close` or the conversation is ending, complete this checklist:
1. Update any `*_REFERENCE.md` files with patterns learned this session.
2. Update `PLAN.md` — mark completed items `[x]`, add new items if the plan changed.
3. Append a dated entry to `LOG.md`: what changed, what's unfinished, what the next session should pick up.
4. Update `INDEX.md` if any files were added or removed.
5. If anything is half-finished, note it clearly in `LOG.md` so the next agent doesn't have to guess.
