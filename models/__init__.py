"""Transformer components for the protein MLM (Phase 2).

The stack, bottom to top:

    ProteinEmbeddings   token ids -> vectors (+ learned absolute positions)
    MultiHeadAttention  softmax(QK^T / sqrt(d_k))V, written out by hand
    FeedForward         position-wise MLP
    TransformerBlock    the two sublayers above, wired post-LN
    TransformerEncoder  embeddings + N blocks -> contextual residue vectors

The MLM head that turns those vectors back into residue predictions is Phase 3.

Deliberately empty of imports: re-exporting the submodules here would make
`python -m models.attention` load that module twice — once through this package,
once as __main__ — which Python flags as an unpredictable-behaviour warning.
Import by full path instead:

    from models.encoder import TransformerEncoder
"""
