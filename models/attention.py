"""Multi-head self-attention, written out as explicit matrix algebra.

`F.scaled_dot_product_attention` is faster, but it hides the arithmetic this
project exists to learn and never hands back the attention weights Phase 5 wants
to plot. So softmax(QK^T / sqrt(d_k))V is spelled out step by step.

Shapes, throughout: B = batch, L = padded token length, H = n_heads, and d_k =
d_model // n_heads — the paper's notation (§3.2.2: d_k = d_v = d_model/h).
Queries and keys must share d_k since they are dotted together; d_v could in
principle differ, but as in the paper we set them equal and carry one variable.

Attention scores are (B, H, L, L), where the first L indexes the *query* (the
residue looking) and the second the *key* (the residue being looked at).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from config import config


def expand_padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """(B, L) 1/0 padding mask -> (B, 1, 1, L) bool, broadcastable over scores.

    The collator emits one flag per token (`dataset.py`), but scores are
    (B, H, L, L). The two singleton axes broadcast across heads and across query
    rows; the trailing L lines up with the *key* axis. So the mask says "these
    positions are never attended **to**" — by any head, from any query.

    Padded query rows still compute an output. Nothing downstream reads it: the
    labels are IGNORE_INDEX there, so no loss flows back through those rows.
    """
    if attention_mask.dim() != 2:
        raise ValueError(
            f"expected a (B, L) padding mask, got shape {tuple(attention_mask.shape)}"
        )
    return attention_mask.bool()[:, None, None, :]


class MultiHeadAttention(nn.Module):
    """Bidirectional (unmasked) self-attention over H parallel heads.

    Splitting d_model into H heads lets different heads specialise — one may
    track adjacency, another a conserved motif — at no extra parameter cost,
    since the head dimension is carved out of d_model rather than added to it.
    """

    def __init__(
        self,
        d_model: int = config.d_model,
        n_heads: int = config.n_heads,
        dropout: float = config.dropout,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        # d_k = d_v = d_model / h, following the paper. The 1/sqrt(d_k) scale
        # exists because q·k sums over d_k terms: with unit-variance components
        # the dot product has variance d_k (std ~5.7 at d_k=32), which saturates
        # the softmax and kills its gradient. Dividing restores variance ~1.
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        # One (d_model -> d_model) projection per role; the head split is a
        # reshape afterwards, so all H heads are projected in a single matmul.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout on the attention *weights* — randomly severing "residue i may
        # look at residue j" during training. Note this is not in the paper's
        # §5.4, which describes dropout on sublayer outputs and on the
        # embedding+positional sums; weight dropout comes from the tensor2tensor
        # reference implementation and BERT. The sublayer-output dropout the
        # paper *does* specify lives in TransformerBlock, which owns the residual.
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model) -> (B, H, L, d_k).

        `view` splits the *last* axis into (H, d_k), so head h takes channels
        [h*d_k : (h+1)*d_k] and positions are untouched; `transpose` then moves
        H in front of L so the matmuls below batch over heads.

        Order matters: view(B, H, L, d_k) directly would have the same shape and
        the wrong meaning, carving the tensor along positions instead of channels.
        """
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, L, d_k) -> (B, L, d_model). Exactly _split_heads reversed."""
        B, _, L, _ = x.shape
        # .contiguous() because .transpose() only permutes strides, and .view()
        # needs a contiguous buffer to reinterpret.
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Attend over `x`.

        `attention_mask` accepts either the (B, L) form the collator emits or an
        already-expanded (B, 1, 1, L); the encoder expands once and passes the
        expanded form down to every layer.

        Returns (output, weights), where weights is the post-softmax,
        *pre-dropout* attention — the version worth plotting — or None.
        """
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # (B, H, L, d_k) @ (B, H, d_k, L) -> (B, H, L, L)
        scores = (q @ k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = expand_padding_mask(attention_mask)
            else:
                # `~` on an int tensor is bitwise, not logical: ~1 == -2, which
                # is truthy, and the fill would blank out every score. Force bool.
                attention_mask = attention_mask.bool()
            # finfo.min rather than -inf: if a row were ever fully masked, -inf
            # gives 0/0 = NaN out of the softmax, while a finite floor degrades
            # to a harmless uniform row. (Can't happen here — every sequence
            # keeps [CLS] — but the failure is silent enough to guard against.)
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

        # Softmax over the *key* axis: each query row becomes a distribution
        # over the positions it can look at, summing to 1.
        weights = scores.softmax(dim=-1)

        out = self._merge_heads(self.attn_dropout(weights) @ v)
        out = self.out_proj(out)

        return out, (weights if need_weights else None)


if __name__ == "__main__":
    torch.manual_seed(config.seed)

    B, L = 4, 12
    attn = MultiHeadAttention().eval()

    x = torch.randn(B, L, config.d_model)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[:, -4:] = 0  # last 4 positions are [PAD]

    out, weights = attn(x, mask, need_weights=True)
    print(f"d_model={attn.d_model}  n_heads={attn.n_heads}  d_k={attn.d_k}")
    print(f"input          : {tuple(x.shape)}")
    print(f"output         : {tuple(out.shape)}")
    print(f"weights        : {tuple(weights.shape)}  (B, H, L_query, L_key)")

    print(f"\nrow sums       : {weights.sum(-1).min():.6f} .. {weights.sum(-1).max():.6f}")
    print(f"weight on [PAD]: {weights[..., -4:].max():.3e}")

    print("\nhead 0, query 0 attention over keys:")
    print("  " + "  ".join(f"{w:.3f}" for w in weights[0, 0, 0]))
    print("  " + "  ".join("pad  " if m == 0 else "real " for m in mask[0]))

    n_params = sum(p.numel() for p in attn.parameters())
    print(f"\nparameters     : {n_params:,}  (4 x d_model^2 + biases)")
