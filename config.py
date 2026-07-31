"""Central configuration for the protein transformer (MLM pretraining).

Every module imports the single `config` instance defined at the bottom so that
data generation, tokenization, training, and evaluation stay in sync.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class Config:
    # ---- Reproducibility ----
    seed: int = 42

    # ---- Data ----
    data_dir: str = "data"
    train_file: str = os.path.join("data", "train.txt")
    val_file: str = os.path.join("data", "val.txt")
    num_train_sequences: int = 8000
    num_val_sequences: int = 1000
    min_seq_len: int = 20          # residues, before special tokens
    # TODO(tune): 128 keeps toy runs fast; real proteins are often longer.
    #   Revisit once the pipeline works end-to-end (memory scales ~O(len^2)).
    max_seq_len: int = 128         # residues, sequences longer than this are truncated
    num_families: int = 6          # toy "protein families" with distinct statistics

    # ---- Masked language modeling ----
    mlm_probability: float = 0.15      # fraction of residues selected for prediction
    mask_token_prob: float = 0.80      # of selected: replaced with [MASK]
    random_token_prob: float = 0.10    # of selected: replaced with a random AA
    # remaining 0.10 of selected are left unchanged

    # ---- Model (consumed in later phases) ----
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1

    # ---- Training ----
    batch_size: int = 32
    # TODO(tune): 3e-4 is a reasonable starting LR for a small MLM;
    #   sweep once training is stable (try 1e-4 .. 5e-4, w/ warmup below).
    lr: float = 3e-4
    weight_decay: float = 0.01
    # TODO(tune): 20 is a placeholder for quick iteration; increase after
    #   we confirm the loss curve is learning and not overfitting the toy data.
    epochs: int = 20
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    num_workers: int = 0

    device: str = field(default_factory=_default_device)

    @property
    def block_size(self) -> int:
        """Max token length including [CLS] and [SEP]."""
        return self.max_seq_len + 2


config = Config()


if __name__ == "__main__":
    from pprint import pprint

    pprint(config.__dict__)
    print("block_size:", config.block_size)
