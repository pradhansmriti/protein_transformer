"""The encoder stack: embeddings followed by N identical transformer blocks.

    input_ids (B, L)  ->  contextual residue vectors (B, L, d_model)

"Contextual" is the whole point. The embedding layer maps residue L to one fixed
vector regardless of where it sits; after six blocks, an L inside a conserved
GXGXXG motif and an L in a hydrophobic stretch have different vectors, because
each has mixed in information from the residues around it.

Phase 3 puts an MLM head on top to turn those vectors back into residue
predictions. Phase 2 stops here.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from config import config
from models.attention import expand_padding_mask
from models.block import TransformerBlock
from models.embeddings import ProteinEmbeddings
from tokenizer import tokenizer as default_tokenizer


class TransformerEncoder(nn.Module):
    """A stack of `n_layers` post-LN transformer blocks.

    Note there is no final LayerNorm. Under post-LN every block already ends in
    one, so the output of the last block is normalised. A trailing LN is required
    only under pre-LN, where the residual stream is never normalised in place —
    its absence here is deliberate, not an oversight.
    """

    def __init__(
        self,
        vocab_size: int = default_tokenizer.vocab_size,
        d_model: int = config.d_model,
        n_heads: int = config.n_heads,
        n_layers: int = config.n_layers,
        d_ff: int = config.d_ff,
        dropout: float = config.dropout,
        block_size: int = config.block_size,
        pad_token_id: int = default_tokenizer.pad_token_id,
    ) -> None:
        super().__init__()
        self.embeddings = ProteinEmbeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            block_size=block_size,
            dropout=dropout,
            pad_token_id=pad_token_id,
        )
        # Identical in structure, independent in weights — nothing is shared
        # between layers. ModuleList (not a plain list) so the parameters are
        # registered and .parameters() / .to(device) actually find them.
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """BERT's initialisation: normal(0, 0.02) for weights, zero for biases.

        0.02 is much tighter than PyTorch's default for nn.Linear (uniform with
        bound 1/sqrt(fan_in) ~= 0.06 at d_model=256). Post-LN is more sensitive
        to initialisation than pre-LN — with a LayerNorm sitting on the residual
        path at every layer, large initial activations compound through the
        stack — so the tighter init matters more here than it otherwise would.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            # nn.Embedding zeroes padding_idx at construction, but the line above
            # just overwrote it. Re-zero, or [PAD] starts as a random vector.
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """(B, L) ids -> (B, L, d_model), plus per-layer attention or None.

        `attention_mask` is the (B, L) 1/0 form the collator emits.
        """
        x = self.embeddings(input_ids)

        # Expand once here rather than in each block: the mask is identical for
        # every layer, and expand_padding_mask allocates a view either way.
        mask = expand_padding_mask(attention_mask) if attention_mask is not None else None

        attentions: Optional[List[torch.Tensor]] = [] if need_weights else None
        for layer in self.layers:
            x, weights = layer(x, mask, need_weights=need_weights)
            if need_weights:
                attentions.append(weights)

        return x, attentions


if __name__ == "__main__":
    from dataset import get_dataloaders

    torch.manual_seed(config.seed)

    encoder = TransformerEncoder().eval()
    print(f"layers={config.n_layers}  d_model={config.d_model}  heads={config.n_heads}  "
          f"d_ff={config.d_ff}")
    print(f"parameters     : {encoder.num_parameters():,}")

    # End-to-end on a real batch: this is the proof Phase 1 and Phase 2 connect.
    train_loader, _ = get_dataloaders()
    batch = next(iter(train_loader))
    input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]

    torch.set_grad_enabled(False)  # inference-only demo; keeps autograd out of the prints
    hidden, attentions = encoder(input_ids, attention_mask, need_weights=True)
    print(f"\ninput_ids      : {tuple(input_ids.shape)}")
    print(f"hidden         : {tuple(hidden.shape)}")
    print(f"attentions     : {len(attentions)} layers x {tuple(attentions[0].shape)}")
    print(f"finite         : {torch.isfinite(hidden).all().item()}")

    # Padding invariance, end to end: one sequence alone vs. the same sequence
    # sitting in a padded batch. Real positions must match.
    #
    # Pick the *most padded* row. Batches are padded to their longest member, so
    # row 0 may well have no padding at all — comparing it against itself would
    # pass trivially and prove nothing.
    row = int(attention_mask.sum(dim=1).argmin())
    n_real = int(attention_mask[row].sum())
    print(f"\n(invariance row {row}: {n_real} real of {attention_mask.shape[1]} tokens)")
    solo, _ = encoder(input_ids[row : row + 1, :n_real],
                      attention_mask[row : row + 1, :n_real])
    batched, _ = encoder(input_ids[row : row + 1], attention_mask[row : row + 1])
    gap = (solo - batched[:, :n_real]).abs().max()
    print(f"\npad invariance : max drift at real positions = {gap:.2e}")

    # No attention weight may land on a padded key, at any layer.
    pad_cols = ~attention_mask.bool()[:, None, None, :]
    worst = max(float(a.masked_select(pad_cols.expand_as(a)).max()) for a in attentions)
    print(f"weight on [PAD]: {worst:.1e}")

    # eval() must be deterministic — a second pass identical to the first.
    again, _ = encoder(input_ids, attention_mask)
    print(f"deterministic  : {torch.equal(hidden, again)}")

    # Contextualisation: the same residue at different positions starts as one
    # fixed embedding and ends as different vectors.
    ids = input_ids[row][:n_real]
    first = ids[1]
    where = (ids == first).nonzero().flatten()
    if len(where) >= 2:
        i, j = int(where[0]), int(where[1])
        aa = default_tokenizer.id_to_token[int(first)]
        emb_gap = (encoder.embeddings.tok_emb(ids[i]) - encoder.embeddings.tok_emb(ids[j])).abs().max()
        ctx_gap = (hidden[row, i] - hidden[row, j]).abs().max()
        print(f"\ncontextualised : residue '{aa}' at positions {i} and {j}")
        print(f"  token embedding differs by : {emb_gap:.4f}  (identical by construction)")
        print(f"  encoder output differs by  : {ctx_gap:.4f}  (context made them distinct)")
