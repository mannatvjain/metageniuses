# Context

## Problem

Black-box AI models used in pandemic surveillance (e.g. wastewater metagenomic sequencing) are hard to trust in high-stakes biosecurity settings. If a model flags a novel pathogen, practitioners need to understand *why* — not just see a score.

## Approach

Train a **Sparse Autoencoder (SAE)** on the internal activations (residual stream) of **MetaGene-1**, a transformer trained on metagenomic sequencing data. The SAE learns a dictionary of sparse features — interpretable directions in activation space — that reveal what biological concepts the model has implicitly learned (e.g. "viral sequence", "known pandemic pathogen", specific gene functions).

This is directly inspired by the **InterProt** paper (Etowah et al.), which applied the same technique to a protein language model.

## Key Concepts

- **Metagenomic sequencing**: Raw genetic material sampled directly from environments (wastewater, soil, air). Captures all pathogens circulating in a population without assuming what organism sequences came from.
- **MetaGene-1**: LLM (transformer) trained on metagenomic sequencing data. Learns representations of genetic sequences without labeled genomes. Powerful for pandemic early warning because it can process signal from novel or poorly-characterized sequences.
- **Sparse Autoencoder (SAE)**: Interpretability tool trained on a model's internal activations. Decompresses dense, polysemantic representations into a sparse dictionary of human-readable features.
- **Residual stream**: The hidden representations a transformer builds up as it processes a sequence — what the SAE is trained on.

## Why This Matters

If MetaGene-1 has internally represented features corresponding to known dangerous pathogen classes, surfacing and labeling them provides:
1. **Validation** of the model's biological understanding
2. **Interpretability** for auditable AI in pandemic surveillance
3. **Biosecurity value** — practitioners get a window into what the model is actually detecting

## External Systems

- **MetaGene-1**: The target model. Need to access its weights and extract intermediate activations.
- **RunPod**: GPU compute for training the SAE.

## Prior Art

- InterProt (Etowah et al.) — SAE on protein language model internals. Paper in `papers/InterProt.pdf`.

## Team

- Mannat Jain
- Bridget
- (others TBD)

## Event

Apart Research AI x Bio Hackathon
