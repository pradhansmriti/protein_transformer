# protein_transformer

A from-scratch BERT-style transformer for masked language modeling (MLM) over
protein (amino-acid) sequences, built in phases.

## Status

- **Phase 1 — Data pipeline & tokenization** ✅
  - `config.py` — all hyperparameters
  - `tokenizer.py` — amino-acid tokenization
  - `data_pipeline.py` — generate toy sequences
  - `dataset.py` — PyTorch `Dataset` + `DataLoader` with MLM masking
- **Phase 2 — Transformer components** ✅
  - `models/embeddings.py` — token + learned absolute position
  - `models/attention.py` — multi-head self-attention, written out by hand
  - `models/feed_forward.py` — position-wise FFN
  - `models/block.py` — one encoder layer (post-LN residuals)
  - `models/encoder.py` — the stack: ids → contextual residue vectors (~4.8M params)
- Phase 3 — MLM head + full model
- Phase 4 — Training loop with validation
- Phase 5 — Evaluation & visualization

### Architecture choices

Encoder-only BERT, following the original papers, with the departures noted in
each module's docstring:

| | ours | note |
|---|---|---|
| positional encoding | learned absolute, as in BERT | Vaswani et al. used fixed sinusoidal |
| norm placement | post-LN, `LayerNorm(x + Sublayer(x))` | Vaswani et al. §3.1, kept by BERT |
| attention | manual `softmax(QKᵀ/√d_k)V` | both; written out rather than fused, so the weights stay available for Phase 5 |
| FFN activation | GELU, as in BERT | Vaswani et al. used ReLU |
| embedding scaling | none | Vaswani et al. scale by `√d_model`; BERT drops it, since the LayerNorm on the embedding sum makes it redundant |

`d_model=256`, `n_heads=8` → `d_k=32`, `n_layers=6`, `d_ff=1024`.

Rotary embeddings (ESM-2's choice) are a natural follow-up — they live inside
`MultiHeadAttention` rather than at the embedding layer, so they're a
self-contained change to one file once there's a baseline loss curve to compare
against.

## Layout

```
config.py            # hyperparameters
tokenizer.py         # amino-acid tokenizer
data_pipeline.py     # toy sequence generation
dataset.py           # Dataset / DataLoader + dynamic MLM masking
models/              # attention, transformer blocks, MLM head, full model (later phases)
train.py             # training loop (later phase)
evaluate.py          # analysis & visualization (later phase)
utils.py             # helpers
```

## Setup

Project-local conda environment (git-ignored):

```bash
conda create -p ./.conda-env python=3.12 -y
./.conda-env/bin/pip install torch numpy matplotlib
```

Run anything with `./.conda-env/bin/python <script>.py`, or activate it with
`conda activate ./.conda-env`.
