"""Position-wise feed-forward network — the paper's FFN sublayer (§3.3).

    FFN(x) = act(x W_1 + b_1) W_2 + b_2

Two linear layers with a nonlinearity between them, applied to each position
*independently*: the same weights for every residue, no mixing across positions.
Attention is the only operation in a transformer that moves information between
positions; this sublayer is where each position privately transforms whatever
attention just gathered for it.

The inner width d_ff is 4x d_model (1024 vs 256), the ratio the paper used
(2048 vs 512) and that nearly everything since has kept. Most of a transformer's
parameters live here rather than in attention: 2 x d_model x d_ff for the FFN
versus 4 x d_model^2 for attention, so 2:1 at the 4x ratio.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import config


class FeedForward(nn.Module):
    """d_model -> d_ff -> d_model, applied identically at every position.

    Deliberately holds no dropout. The paper (§5.4) applies dropout to the
    *output* of each sublayer, just before the residual add — that belongs to
    TransformerBlock, which owns the residual. Adding another here would silently
    double it. (Some implementations do add an "activation dropout" after the
    nonlinearity; fairseq exposes it and defaults it to 0.)
    """

    def __init__(
        self,
        d_model: int = config.d_model,
        d_ff: int = config.d_ff,
    ) -> None:
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff)      # W_1: 256 -> 1024
        self.down_proj = nn.Linear(d_ff, d_model)    # W_2: 1024 -> 256

        # The 2017 paper used ReLU; BERT switched to GELU and it has stuck.
        # GELU weights the input by its percentile under a Gaussian instead of
        # hard-clipping at zero, so negatives keep a small gradient rather than
        # being zeroed outright — no "dead" units, and a smooth derivative.
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model) -> (B, L, d_model)."""
        return self.down_proj(self.act(self.up_proj(x)))


if __name__ == "__main__":
    torch.manual_seed(config.seed)

    B, L = 4, 12
    ff = FeedForward().eval()

    x = torch.randn(B, L, config.d_model)
    out = ff(x)

    ratio = config.d_ff // config.d_model
    print(f"d_model={config.d_model}  d_ff={config.d_ff}  ratio={ratio}x")
    print(f"input          : {tuple(x.shape)}")
    print(f"output         : {tuple(out.shape)}")

    # Genuinely nonlinear: an affine map would give f(2x) - 2f(x) = -bias only.
    gap = (ff(2 * x) - 2 * ff(x)).abs().max()
    print(f"\nnonlinear      : max |f(2x) - 2f(x)| = {gap:.4f}")

    # Position-wise: shuffling positions shuffles the output the same way, i.e.
    # row i of the output depends on row i of the input and nothing else.
    perm = torch.randperm(L)
    same = torch.allclose(ff(x[:, perm]), out[:, perm], atol=1e-6)
    print(f"position-wise  : permuting inputs permutes outputs = {same}")

    n_params = sum(p.numel() for p in ff.parameters())
    attn_params = 4 * config.d_model * config.d_model + 4 * config.d_model
    print(f"\nparameters     : {n_params:,}  (2 x d_model x d_ff + biases)")
    print(f"  vs attention : {attn_params:,}  -> FFN is {n_params / attn_params:.1f}x larger")
