"""Amino-acid tokenizer for protein sequences.

A character-level tokenizer over the 20 canonical amino acids plus the common
ambiguity codes, with the usual BERT-style special tokens. Kept dependency-free
(pure Python) so it can be used anywhere in the pipeline.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

# Order matters: special tokens occupy the low ids so [PAD] == 0.
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# 20 canonical amino acids (single-letter codes).
CANONICAL_AA = list("ACDEFGHIKLMNPQRSTVWY")

# Ambiguity / non-standard codes kept as real tokens so they don't all collapse
# to [UNK]: B (Asx), Z (Glx), J (Xle/Leu-or-Ile), U (Sec), O (Pyl), X (any).
AMBIGUOUS_AA = list("BZJUOX")


class ProteinTokenizer:
    def __init__(self) -> None:
        self.vocab: List[str] = SPECIAL_TOKENS + CANONICAL_AA + AMBIGUOUS_AA
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        # Convenience handles for special ids.
        self.pad_token_id = self.token_to_id["[PAD]"]
        self.unk_token_id = self.token_to_id["[UNK]"]
        self.cls_token_id = self.token_to_id["[CLS]"]
        self.sep_token_id = self.token_to_id["[SEP]"]
        self.mask_token_id = self.token_to_id["[MASK]"]

        self.special_token_ids = {
            self.pad_token_id,
            self.unk_token_id,
            self.cls_token_id,
            self.sep_token_id,
            self.mask_token_id,
        }

        # Ids MLM is allowed to sample as "random" replacements (real residues only).
        self.regular_token_ids = [
            self.token_to_id[a] for a in (CANONICAL_AA + AMBIGUOUS_AA)
        ]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ---- core encode / decode ----
    def encode(
        self,
        sequence: str,
        add_special_tokens: bool = True,
        max_len: int | None = None,
    ) -> List[int]:
        """Convert an amino-acid string to token ids.

        `max_len`, if given, bounds the number of *residues* (special tokens are
        added on top). Unknown characters map to [UNK].
        """
        residues = sequence.strip().upper()
        if max_len is not None:
            residues = residues[:max_len]

        ids = [self.token_to_id.get(ch, self.unk_token_id) for ch in residues]
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        out = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i in self.special_token_ids:
                continue
            out.append(self.id_to_token.get(i, "[UNK]"))
        return "".join(out)

    def convert_tokens_to_ids(self, tokens: Iterable[str]) -> List[int]:
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens]

    def is_special(self, token_id: int) -> bool:
        return int(token_id) in self.special_token_ids

    def __len__(self) -> int:
        return self.vocab_size


# Shared instance used across the pipeline.
tokenizer = ProteinTokenizer()


if __name__ == "__main__":
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    ids = tokenizer.encode(seq)
    print("vocab_size:", tokenizer.vocab_size)
    print("seq       :", seq)
    print("ids       :", ids)
    print("decoded   :", tokenizer.decode(ids))
    assert tokenizer.decode(ids) == seq, "round-trip failed"

    # unknown char handling
    assert tokenizer.encode("MK*", add_special_tokens=False)[-1] == tokenizer.unk_token_id
    print("round-trip OK")
