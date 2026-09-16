"""The masked-language-modeling head: contextual vectors -> residue logits.

    (B, L, d_model)  ->  (B, L, vocab_size)

The encoder's job ended with a d_model vector per position. To ask "which residue
was here?" that vector has to be scored against all 31 tokens, which is one
matmul with a (vocab_size, d_model) matrix. Everything else in this file exists
around that matmul.

BERT calls this `BertLMPredictionHead` and puts a small MLP in front of the
projection:

    Linear(d_model, d_model) -> GELU -> LayerNorm -> Linear(d_model, vocab, bias=False) + bias

The transform block is not padding. Two reasons it is there:

  * **Task-specific scratch space.** The encoder output is the representation we
    actually want to keep — the thing a downstream classifier or an embedding
    plot will use. Prediction-specific reshaping (which directions separate C
    from S?) happens *here*, in a layer that gets thrown away after pretraining,
    so the encoder is not pushed to specialise for the MLM objective.
  * **It makes weight tying viable.** Tying forces the output projection to reuse
    the input embedding table, which was optimised for a different job. The
    transform is the adapter that lets one matrix serve both directions; without
    it, tying constrains the encoder output far harder.

The LayerNorm also fixes the scale of what enters the projection, which keeps the
logits — and therefore the softmax temperature — in a sane range from step 0.

Note this head scores *every* position, padding and all. Selecting the masked
ones is the loss function's job (Phase 3's `model.py`), via the IGNORE_INDEX
labels the collator already produces. Computing 130 rows and discarding most is
cheaper than the gather/scatter needed to avoid it at this size.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import config
from tokenizer import tokenizer as default_tokenizer


class MLMHead(nn.Module):
    """BERT's prediction head: transform, then project onto the vocabulary."""

    def __init__(
        self,
        vocab_size: int = default_tokenizer.vocab_size,
        d_model: int = config.d_model,
    ) -> None:
        super().__init__()
        # ---- transform: d_model -> d_model, task-specific scratch space ----
        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(d_model)

        # ---- projection: d_model -> vocab_size ----
        # bias=False because this weight may be *replaced* by the embedding
        # table in tie_to(). The embedding table is (vocab_size, d_model) and has
        # no bias of its own, so the output bias has to live outside the Linear
        # as an independent parameter — it is never shared, tied or not.
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Initialise here, in __init__, so the order is impossible to get wrong:
        # a caller that constructs the head and then calls tie_to() cannot
        # accidentally re-randomise the shared tensor afterwards. (Running an
        # init pass after tying would overwrite the embedding table *and* undo
        # the padding_idx zeroing that ProteinEmbeddings depends on.)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """BERT init, matching TransformerEncoder._init_weights in encoder.py.

        Kept as a local copy rather than imported: the head has to be
        initialisable on its own — see the __init__ comment on ordering — and
        importing the encoder just to borrow a staticmethod would pull the whole
        stack in for four lines.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def tie_to(self, embedding: nn.Embedding) -> None:
        """Share the output projection with the input embedding table.

        Both matrices are (vocab_size, d_model) and both are, loosely, a lookup
        of "what does residue v look like in model space" — the embedding reads
        it, the decoder scores against it. Press (2017) showed sharing them helps
        in NLP, where the vocabulary is tens of thousands of rows and the table
        is a large share of the parameters.

        Here it saves 7,936 params (0.16%), so this is a *hypothesis to test*,
        not a free win — hence `config.tie_word_embeddings`.

        One consequence to keep in mind: `padding_idx` only zeroes the gradient
        arriving through the embedding *lookup*. The decoder matmul produces
        gradient for every row of the shared matrix, [PAD] included, so under
        tying the [PAD] row drifts away from zero during training. Harmless —
        attention masks those positions out anyway — but the "[PAD] is exactly
        zero forever" guarantee from Phase 2 holds only when untied.
        """
        # Assignment, not copy_: the two modules must reference the *same*
        # tensor, so one gradient accumulates into one parameter. A copy would
        # drift apart on the first optimizer step and silently stop being tied.
        self.decoder.weight = embedding.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model) -> (B, L, vocab_size) raw logits (no softmax).

        Logits, not probabilities: `F.cross_entropy` wants them unnormalised, and
        doing log_softmax inside it is numerically safer than softmax-then-log.
        """
        x = self.ln(self.act(self.dense(hidden_states)))
        return self.decoder(x) + self.bias


if __name__ == "__main__":
    import math

    import torch.nn.functional as F

    torch.manual_seed(config.seed)

    V = default_tokenizer.vocab_size
    head = MLMHead().eval()

    B, L = 4, 12
    hidden = torch.randn(B, L, config.d_model)
    logits = head(hidden)

    print(f"d_model={config.d_model}  vocab_size={V}")
    print(f"input          : {tuple(hidden.shape)}")
    print(f"output         : {tuple(logits.shape)}")

    # The acceptance criterion for the whole phase, checked here first: a model
    # that knows nothing should spread probability mass uniformly over the
    # vocabulary, giving -log(1/V) = log(V) nats per token. Anything far from
    # this at init means the logits are not centred — usually a scale bug.
    labels = torch.randint(0, V, (B, L))
    loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1))
    print(f"\nloss at init   : {loss:.4f}   (ln({V}) = {math.log(V):.4f})")
    print(f"logit spread   : max |logit| = {logits.abs().max():.4f}")

    # Position-wise, like the FFN: the head sees one vector at a time and has no
    # way to move information between positions.
    perm = torch.randperm(L)
    same = torch.allclose(head(hidden[:, perm]), logits[:, perm], atol=1e-6)
    print(f"position-wise  : permuting inputs permutes outputs = {same}")

    # ---- tying ----
    from models.embeddings import ProteinEmbeddings

    emb = ProteinEmbeddings()
    tied = MLMHead()

    # Count the *pair*, not the head alone: the duplicate is across the two
    # modules, so tying changes nothing about head.parameters() on its own.
    def pair_params(*modules) -> int:
        # Deduplicate by identity — after tying, decoder.weight and
        # tok_emb.weight are one tensor and must be counted once.
        return sum(p.numel() for p in
                   {id(p): p for m in modules for p in m.parameters()}.values())

    before = pair_params(emb, tied)
    tied.tie_to(emb.tok_emb)
    after = pair_params(emb, tied)

    shared = tied.decoder.weight is emb.tok_emb.weight
    print(f"\ntying          : decoder.weight is tok_emb.weight = {shared}")
    print(f"  embeddings + head, untied: {before:,}")
    print(f"  embeddings + head, tied  : {after:,}   (saves {before - after:,}, "
          f"{(before - after) / before:.1%})")

    # Tying must survive the optimizer: one gradient, one parameter, one update.
    out = tied(torch.randn(2, 5, config.d_model)).sum()
    out.backward()
    print(f"  one gradient : emb.tok_emb.weight.grad is not None = "
          f"{emb.tok_emb.weight.grad is not None}")

    # ---- gradient reaches the input ----
    x = torch.randn(B, L, config.d_model, requires_grad=True)
    head(x).sum().backward()
    print(f"\ngradient       : finite={torch.isfinite(x.grad).all().item()}  "
          f"max |dL/dx| = {x.grad.abs().max():.4f}")

    n_dense = sum(p.numel() for p in head.dense.parameters())
    n_ln = sum(p.numel() for p in head.ln.parameters())
    n_dec = head.decoder.weight.numel() + head.bias.numel()
    print(f"\nparameters     : {n_dense + n_ln + n_dec:,}  "
          f"(transform {n_dense:,} + LN {n_ln:,} + decoder {n_dec:,})")
