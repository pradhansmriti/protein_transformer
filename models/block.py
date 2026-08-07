"""One transformer encoder layer: self-attention + feed-forward, each wrapped in
a residual connection and a LayerNorm.

Post-LN, following Vaswani et al. (2017) §3.1 and BERT:

    x = LayerNorm(x + Dropout(Sublayer(x)))

The two wrappers are what make depth trainable. The residual `x +` gives every
layer the option of doing nothing, so a 6-layer stack can't be worse than a
1-layer one by construction, and it gives gradients a short path back to the
embeddings. The LayerNorm keeps activations at a stable scale so that scale
doesn't compound multiplicatively as blocks stack.

Dropout sits on the sublayer *output*, before the residual add — the placement
the paper specifies in §5.4. Neither MultiHeadAttention nor FeedForward applies
it, so it lands exactly once.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from config import config
from models.attention import MultiHeadAttention
from models.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = config.d_model,
        n_heads: int = config.n_heads,
        d_ff: int = config.d_ff,
        dropout: float = config.dropout,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff)

        # LayerNorm normalises over the *feature* axis: for each (batch,
        # position) pair independently, it standardises that position's d_model
        # numbers to mean 0 / variance 1, then applies a learned scale and shift.
        #
        # That per-position independence is why LayerNorm and not BatchNorm.
        # BatchNorm would pool statistics across the batch and the sequence, so
        # [PAD] positions would contaminate the mean and variance of real ones,
        # and the statistics would shift with batch composition. LayerNorm is
        # blind to everything but the position it is normalising.
        #
        # torch defaults to eps=1e-5; BERT uses 1e-12. Immaterial at this scale.
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # One module used for both sublayers: nn.Dropout is stateless, and each
        # call draws a fresh mask, so sharing it is not the same as reusing a mask.
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """(B, L, d_model) -> (B, L, d_model), plus attention weights or None."""
        attn_out, weights = self.attn(x, attention_mask, need_weights=need_weights)
        x = self.ln1(x + self.dropout(attn_out))
        x = self.ln2(x + self.dropout(self.ff(x)))
        return x, weights

    # ---- Pre-LN alternative (deliberately not used; see Phase 2 plan) ----
    #
    #     x = x + self.dropout(self.attn(self.ln1(x), attention_mask)[0])
    #     x = x + self.dropout(self.ff(self.ln2(x)))
    #     ...plus a final LayerNorm after the last block, in TransformerEncoder.
    #
    # Normalising the sublayer *input* instead leaves the residual stream as an
    # unbroken identity path from embeddings to output, so gradients reach layer
    # 0 undiminished. Post-LN puts a LayerNorm on that path at every layer, which
    # is why its lower-layer gradients scale with depth at initialisation
    # (Xiong et al., 2020) and why the original Transformer needed warmup at all.
    # config.warmup_steps is therefore load-bearing here, not decorative.
    #
    # If Phase 4 training diverges or proves warmup-sensitive, switching to the
    # two lines above (plus the final LN) is the first thing to try.


if __name__ == "__main__":
    torch.manual_seed(config.seed)

    B, L_real, L_pad = 4, 10, 6
    L = L_real + L_pad
    block = TransformerBlock().eval()  # eval() so dropout is off and runs repeat

    x = torch.randn(B, L, config.d_model)
    mask = torch.zeros(B, L, dtype=torch.long)
    mask[:, :L_real] = 1

    out, weights = block(x, mask, need_weights=True)
    print(f"input          : {tuple(x.shape)}")
    print(f"output         : {tuple(out.shape)}")
    print(f"weights        : {tuple(weights.shape)}")
    print(f"finite         : {torch.isfinite(out).all().item()}")

    # Padding invariance: run the real prefix alone, and again inside the padded
    # batch with garbage in the padded slots. Real positions must be identical.
    solo, _ = block(x[:, :L_real], mask[:, :L_real])
    noisy = x.clone()
    noisy[:, L_real:] = torch.randn(B, L_pad, config.d_model) * 100
    padded, _ = block(noisy, mask)
    gap = (solo - padded[:, :L_real]).abs().max()
    print(f"\npad invariance : max drift at real positions = {gap:.2e}")

    # LayerNorm acts per position over the feature axis, so every row of the
    # output is standardised independently -> mean ~0, std ~1 across d_model.
    print(f"output stats   : mean={out.mean(-1).abs().max():.2e}  "
          f"std={out.std(-1).mean():.4f}")

    # The residual path should carry gradient all the way back to the input.
    # Note the loss must be .pow(2).sum(), not .sum(): LayerNorm centres each
    # position, so with weight=1/bias=0 the output sums to exactly zero over the
    # feature axis whatever the input is. A plain .sum() is a constant, and its
    # gradient is legitimately 0 — which looks exactly like a broken residual.
    x_grad = x.clone().requires_grad_(True)
    block(x_grad, mask)[0].pow(2).sum().backward()
    print(f"\ngradient to in : finite={torch.isfinite(x_grad.grad).all().item()}  "
          f"norm={x_grad.grad.norm():.4f}")

    n_params = sum(p.numel() for p in block.parameters())
    print(f"\nparameters     : {n_params:,} per block")
