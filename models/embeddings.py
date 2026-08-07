"""Input embeddings: token identity + learned absolute position.

Self-attention is permutation-equivariant — shuffle the residues and the output
shuffles identically, because softmax(QK^T)V is a sum over positions and sums do
not care about order. So position has to be injected from outside, or the model
literally cannot tell MKTAY from YATKM. For protein data that would be fatal: a
motif like CXXC *is* an adjacency claim.

This uses BERT's learned absolute scheme — a second embedding table, one vector
per position index, added to the token vector. Two departures from Vaswani et al.
(2017) worth naming:

  * The paper uses fixed sinusoidal encodings; these are learned parameters.
  * The paper scales token embeddings by sqrt(d_model) before adding the
    positional signal. That exists to stop the embeddings (initialised small,
    and weight-tied to the output projection) from being swamped by a sinusoid of
    scale ~1. BERT drops it because the LayerNorm below re-standardises the sum
    anyway, which makes the fix-up redundant. We follow BERT.

Also absent: BERT's segment/token-type embeddings. Those distinguish sentence A
from sentence B in next-sentence prediction; we have one sequence and no NSP.

The learned table is the one real limitation: it has exactly block_size rows, so
unlike sinusoidal or rotary encodings it cannot extrapolate to a longer sequence
at all. Hence the explicit length check in forward().
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import config
from tokenizer import tokenizer as default_tokenizer


class ProteinEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int = default_tokenizer.vocab_size,
        d_model: int = config.d_model,
        block_size: int = config.block_size,
        dropout: float = config.dropout,
        pad_token_id: int = default_tokenizer.pad_token_id,
    ) -> None:
        super().__init__()
        self.block_size = block_size

        # padding_idx pins row [PAD] to zeros and, more importantly, excludes it
        # from gradient updates — so [PAD] contributes exactly nothing, forever,
        # rather than drifting into a learned "padding is meaningful" vector.
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(block_size, d_model)

        # LayerNorm before block 0, as in BERT: the sum of two independently
        # initialised embeddings has no particular scale, and normalising here
        # means block 0 receives the same distribution every later block does.
        self.ln = nn.LayerNorm(d_model)

        # Dropout on the embedding + positional sum. Unlike the attention-weight
        # dropout, this one *is* specified in the paper (§5.4).
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """(B, L) token ids -> (B, L, d_model)."""
        _, L = input_ids.shape
        if L > self.block_size:
            raise ValueError(
                f"sequence length {L} exceeds block_size {self.block_size}; "
                "learned absolute positions cannot extrapolate. Either raise "
                "config.max_seq_len (block_size = max_seq_len + 2) or truncate."
            )

        # arange on the input's device so this works unchanged on cpu/mps/cuda.
        positions = torch.arange(L, device=input_ids.device)

        # (B, L, d_model) + (L, d_model) -> broadcasts over the batch: every
        # sequence gets the same positional vector at the same index.
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        return self.dropout(self.ln(x))


if __name__ == "__main__":
    torch.manual_seed(config.seed)

    emb = ProteinEmbeddings().eval()
    pad_id = default_tokenizer.pad_token_id

    B, L_real, L_pad = 2, 8, 4
    ids = torch.randint(len(default_tokenizer.special_token_ids), len(default_tokenizer),
                        (B, L_real + L_pad))
    ids[:, L_real:] = pad_id

    out = emb(ids)
    print(f"vocab_size={default_tokenizer.vocab_size}  d_model={config.d_model}  "
          f"block_size={config.block_size}")
    print(f"input          : {tuple(ids.shape)}")
    print(f"output         : {tuple(out.shape)}")

    # padding_idx keeps the [PAD] *row of the table* at exactly zero. (The output
    # at padded positions is still nonzero — the positional vector is added, and
    # LayerNorm then shifts it. Attention is what ignores those positions.)
    print(f"\n[PAD] emb row  : max |value| = {emb.tok_emb.weight[pad_id].abs().max():.1e}")

    # Position actually reaches the model: the same residue at two positions must
    # produce different vectors, or nothing above can distinguish order.
    same_residue = torch.full((1, 4), 7, dtype=torch.long)
    vecs = emb(same_residue)[0]
    spread = (vecs[0] - vecs[1]).abs().max()
    print(f"positional sig : same residue at pos 0 vs 1 differs by {spread:.4f}")

    # LayerNorm standardises each position over the feature axis.
    print(f"output stats   : mean={out.mean(-1).abs().max():.2e}  std={out.std(-1).mean():.4f}")

    # Over-length input must fail loudly rather than silently indexing garbage.
    try:
        emb(torch.zeros(1, config.block_size + 1, dtype=torch.long))
    except ValueError as e:
        print(f"\nlength guard   : raised -> {str(e).split(';')[0]}")

    n_tok = emb.tok_emb.weight.numel()
    n_pos = emb.pos_emb.weight.numel()
    print(f"\nparameters     : {sum(p.numel() for p in emb.parameters()):,} "
          f"(token {n_tok:,} + position {n_pos:,} + LN)")
