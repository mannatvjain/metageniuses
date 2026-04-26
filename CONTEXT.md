# Context

## Problem

Black-box AI models used in pandemic surveillance (e.g. wastewater metagenomic sequencing) are hard to trust in high-stakes biosecurity settings. If a model flags a novel pathogen, practitioners need to understand *why* — not just see a score.

## Approach

Train a **Sparse Autoencoder (SAE)** on the internal activations (residual stream) of **MetaGene-1**, a transformer trained on metagenomic sequencing data. The SAE learns a dictionary of sparse features — interpretable directions in activation space — that reveal what biological concepts the model has implicitly learned (e.g. "viral sequence", "known pandemic pathogen", specific gene functions).

This is directly inspired by the **InterProt** paper (Adams et al.), which applied the same technique to a protein language model (ESM-2), and **SURF** (Liu & Rogers), which used InterProt's SAE to find peptide-binding domain family-specific features.

## Key Concepts

- **Metagenomic sequencing**: Raw genetic material sampled directly from environments (wastewater, soil, air). Captures all pathogens circulating in a population without assuming what organism sequences came from.
- **MetaGene-1**: 7B-parameter decoder-only transformer (Llama-2 architecture) trained on 1.5T base pairs of wastewater metagenomic DNA/RNA sequences. BPE tokenized (vocab=1024), 32 layers, d_model=4096. SOTA on pathogen detection (92.96 MCC).
- **Sparse Autoencoder (SAE)**: Interpretability tool trained on a model's internal activations. Decompresses dense, polysemantic representations into a sparse dictionary of human-readable features. We use TopK SAEs (k=64) with 8x expansion (32,768 latents).
- **Residual stream**: The hidden representations a transformer builds up as it processes a sequence — what the SAE is trained on. We train on layer 32 (the last layer).

## Why This Matters

If MetaGene-1 has internally represented features corresponding to known dangerous pathogen classes, surfacing and labeling them provides:
1. **Validation** of the model's biological understanding
2. **Interpretability** for auditable AI in pandemic surveillance
3. **Biosecurity value** — practitioners get a window into what the model is actually detecting

## External Systems

- **MetaGene-1**: Weights at `metagene-ai/METAGENE-1` on HuggingFace
- **RunPod**: GPU compute for extraction and SAE training
- **NCBI BLAST**: Free REST API for identifying organisms from nucleotide sequences

## Prior Art

- **InterProt** (Adams et al., 2025) — SAE on ESM-2 protein language model. Found family-specific features, activation pattern taxonomy, interpretable probing. Paper in `papers/InterProt.pdf`.
- **MetaGene-1** (Liu et al., 2025) — The target model. Pathogen detection + genomic embedding benchmarks. Paper in `papers/metagene-1.pdf`.
- **SURF** (Liu & Rogers, 2025) — Used InterProt's pre-trained SAE to find PBD family-specific and cross-family features across 24 human peptide-binding domain families. Paper in `papers/SURF_Paper.docx`.

## Team

- Mannat Jain
- Peyton Jackson
- Bridget Liu
- Ciaran
- Astrid

## Event

Apart Research AI x Bio Hackathon
