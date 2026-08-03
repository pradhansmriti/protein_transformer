"""Generate toy protein sequences with learnable structure.

Real MLM pretraining needs statistical signal: if residues were i.i.d. uniform,
there would be nothing to predict from context. So we synthesize a handful of
"protein families," each with:

  * its own amino-acid background distribution (biased composition), and
  * a set of conserved motifs inserted at semi-fixed positions.

That gives the model both local (motif) and compositional (family) structure to
learn, while keeping the data fully self-contained and reproducible.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Tuple

from config import config
from tokenizer import CANONICAL_AA


@dataclass
class Family:
    name: str
    # Relative weights over the 20 canonical AAs (biased composition).
    weights: List[float]
    # Conserved motifs as (fractional_position, motif_string).
    motifs: List[Tuple[float, str]]


def _biased_weights(favored: str, boost: float = 6.0) -> List[float]:
    """Base weight 1.0 for every AA, boosted for a favored subset."""
    favored_set = set(favored)
    return [boost if aa in favored_set else 1.0 for aa in CANONICAL_AA]


def build_families(n: int) -> List[Family]:
    """Deterministically construct `n` distinct toy families."""
    # A few hand-picked flavors; extra families reuse flavors with fresh motifs.
    flavors = [
        ("hydrophobic",  "AVLIMF", ["GXGXXG", "LLLL"]),
        ("charged",      "DEKR",   ["DDGD", "RKRK"]),
        ("polar",        "STNQ",   ["NPST", "QQSQ"]),
        ("aromatic",     "FWY",    ["WXWXW", "YFY"]),
        ("glycine_rich", "GPAS",   ["GPGP", "GGSGG"]),
        ("cysteine",     "CST",    ["CXXC", "CCXCC"]),
    ]
    families: List[Family] = []
    for i in range(n):
        name, favored, motif_strs = flavors[i % len(flavors)]
        weights = _biased_weights(favored)
        # Place each motif at a stable fractional position for this family.
        motifs: List[Tuple[float, str]] = []
        for j, m in enumerate(motif_strs):
            frac = (0.2 + 0.5 * (j + i * 0.13)) % 0.9  # spread motifs out
            motifs.append((frac, m))
        families.append(Family(f"{name}_{i}", weights, motifs))
    return families


def _instantiate_motif(rng: random.Random, motif: str) -> str:
    """Resolve wildcard 'X' positions in a motif to random canonical AAs."""
    return "".join(rng.choice(CANONICAL_AA) if ch == "X" else ch for ch in motif)


def sample_sequence(rng: random.Random, family: Family) -> str:
    """Draw one sequence from a family: background composition + inserted motifs."""
    length = rng.randint(config.min_seq_len, config.max_seq_len)
    residues = rng.choices(CANONICAL_AA, weights=family.weights, k=length)

    for frac, motif in family.motifs:
        resolved = _instantiate_motif(rng, motif)
        if len(resolved) >= length:
            continue
        pos = int(frac * (length - len(resolved)))
        residues[pos:pos + len(resolved)] = list(resolved)

    return "".join(residues)


def generate(n_sequences: int, families: List[Family], rng: random.Random) -> List[str]:
    seqs = []
    for _ in range(n_sequences):
        fam = rng.choice(families)
        seqs.append(sample_sequence(rng, fam))
    return seqs


def write_sequences(path: str, sequences: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(sequences) + "\n")


def main() -> None:
    rng = random.Random(config.seed)
    families = build_families(config.num_families)

    train = generate(config.num_train_sequences, families, rng)
    val = generate(config.num_val_sequences, families, rng)

    write_sequences(config.train_file, train)
    write_sequences(config.val_file, val)

    lengths = [len(s) for s in train]
    print(f"families      : {[f.name for f in families]}")
    print(f"train         : {len(train)} seqs -> {config.train_file}")
    print(f"val           : {len(val)} seqs -> {config.val_file}")
    print(f"length (train): min={min(lengths)} max={max(lengths)} "
          f"mean={sum(lengths) / len(lengths):.1f}")
    print(f"example       : {train[0][:60]}...")


if __name__ == "__main__":
    main()
