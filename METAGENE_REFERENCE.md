# MetaGene-1 Reference

Architecture and implementation details from `vendor/metagene-pretrain`.

## Model Architecture

MetaGene-1 is a LLaMA-2-7B-style autoregressive transformer.

### Gotcha: Code config vs actual 7B model
**Wrong**: Trust `vendor/metagene-pretrain/train/litgpt/config.py` (`genomics-llama`) — it shows n_layer=24, n_embd=1536
**Right**: The paper (Table 1) describes the actual 7B model with different dimensions (see below)
**Why**: The code config appears to be a smaller variant or development config. The published model is Llama-2-7B architecture.

### Architecture (from paper, Table 1)

| Parameter | Paper (7B model) | Code config (`genomics-llama`) |
|-----------|-----------------|-------------------------------|
| Architecture | Llama-2-7B | — |
| Embedding size | 4096 | 1536 |
| Intermediate size (MLP) | 11008 | 6144 |
| Attention heads | 32 | 16 |
| Hidden layers | 32 | 24 |
| Vocab size | 1024 | 1024 |
| Sequence length | 512 | 512 |
| Normalization | RMSNorm | RMSNorm (eps=1e-5) |
| Position embedding | Rotary | Rotary (100%) |
| Bias | None | False |
| Regularization | z-loss | — |

**Training details** (paper):
- Batch size: 30,720 sequences
- Micro-batch size: 48
- Learning rate: 6e-4, cosine decay
- Warmup: 2000 steps
- Weight decay: 0.1
- Betas: (0.9, 0.95)
- Trained on 4 nodes x 8 H100 SXM5 GPUs
- MFU: ~40%

### Continual Pretraining (2nd stage)
- ~9% of total pretraining tokens
- Mixes in curated multi-species genomes (from Zhou et al. 2023 GUE dataset) at 1:8 ratio with metagenomic reads
- Includes: human genomes, fungi, mammalian, invertebrate, bacteria
- Uses warmup + cooldown learning rate schedule

### Context Stuffing
- Short reads packed into 512-token context window
- Attention mask prevents cross-attention between different reads
- Implemented via FlashAttention-2's `flash_attn_varlen_func`

### Key Differences from ESM-2 (InterProt's target)

| | ESM-2 (650M) | MetaGene-1 (7B) |
|--|-------------|------------|
| Architecture | BERT-style (masked LM) | LLaMA-style (autoregressive) |
| Layers | 33 | 32 |
| Hidden dim | 1280 | 4096 |
| Params | 650M | ~7B |
| Input | Protein sequences (amino acids) | Metagenomic reads (DNA/RNA, BPE tokenized) |
| Vocab | ~33 amino acid tokens | 1024 BPE tokens |
| Attention | Full bidirectional | Causal (left-to-right) |

### Implications for SAE Training

- **d_model = 4096** (not 1280 like ESM-2, not 1536 like code config) — SAE dimensions must match the actual model
- **Causal attention** means residual stream representations at position i only see tokens 0..i. Features may be more position-dependent than in ESM-2's bidirectional setting
- **BPE tokenization** means one token can span multiple nucleotides (e.g. "ATCC" is one token). SAE features may correspond to multi-nucleotide patterns
- **32 layers** — InterProt found middle layers richest; start with layers 12-20
- **Context stuffing** — activations near read boundaries may be noisy; consider masking padding/separator tokens when collecting SAE training data

## Tokenizer

BPE tokenizer trained on ~150M reads (2B base pairs). Vocab size 1024.
- Located at: `vendor/metagene-pretrain/train/minbpe/mgfm-1024/`
- Short tokens: AA, GG, TAC, AAAA, ATCC, etc.
- Long tokens: up to ~50+ nucleotides (e.g. ACCGTTGCCGGCGTACTCCCCAGGTGGATAGCTTAATGGTTTCCCTCAGGCACCC)

## Data

- Trained on metagenomic sequencing data from human wastewater (municipal influent)
- Captures DNA/RNA from tens of thousands of organisms
- Data streamed from S3 via MosaicML streaming
- **Dataset not yet public** — check for updates

## Model Weights

- **HuggingFace**: `huggingface.co/metagene-ai` (listed on paper page 1)
- **GitHub**: `github.com/metagene-ai` (code repository)
- The model config name is `genomics-llama`

## Code Structure

```
vendor/metagene-pretrain/
  train/
    litgpt/
      config.py        # Model configs (genomics-llama, genomics-llama-mini)
      model.py         # GPT model implementation
      pretrain.py       # Training loop
      tokenizer.py      # Tokenizer loading
      data/
        base.py         # NAODataset class
        nao.py          # NAO DataModule with S3 streaming
    minbpe/
      mgfm-1024/       # BPE tokenizer files (1024 vocab)
    config_hub/
      pretrain/
        genomicsllama.yml  # Pretrain config
  evaluation/
    data/
      sample_reads.csv  # Sample metagenomic reads for testing
```
