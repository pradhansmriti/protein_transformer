# protein_transformer

A from-scratch BERT-style transformer for masked language modeling (MLM) over
protein (amino-acid) sequences, built in phases.

## Status

- **Phase 1 — Data pipeline & tokenization** ✅
  - `config.py` — all hyperparameters
  - `tokenizer.py` — amino-acid tokenization
  - `data_pipeline.py` — generate toy sequences
  - `dataset.py` — PyTorch `Dataset` + `DataLoader` with MLM masking
- Phase 2 — Transformer components (attention, blocks, encoder)
- Phase 3 — MLM head + full model
- Phase 4 — Training loop with validation
- Phase 5 — Evaluation & visualization

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
