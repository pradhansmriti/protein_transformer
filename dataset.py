"""PyTorch `Dataset` / `DataLoader` with dynamic MLM masking.

The dataset stores only encoded sequences; masking happens in the collator, so a
sequence gets a *different* mask every epoch ("dynamic masking", which RoBERTa
showed beats masking once up front). Batches are padded to the longest member
rather than to `block_size`, since our lengths vary from 20 to 128 residues and
padding to the max would waste most of the compute on [PAD].
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from config import config
from tokenizer import ProteinTokenizer, tokenizer as default_tokenizer

# Ignored by `F.cross_entropy` — positions the model is not asked to predict.
IGNORE_INDEX = -100


def read_sequences(path: str) -> List[str]:
    """Read one amino-acid sequence per line, skipping blanks."""
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


class ProteinDataset(Dataset):
    """Encoded protein sequences, ready for the MLM collator.

    Sequences are tokenized once up front (the corpus is small and
    character-level encoding is cheap). Truncation to `max_seq_len` residues
    happens before special tokens are added, so [CLS]/[SEP] always survive.
    """

    def __init__(
        self,
        path: str,
        tokenizer: ProteinTokenizer = default_tokenizer,
        max_seq_len: int = config.max_seq_len,
    ) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.sequences = read_sequences(path)
        self.encoded: List[List[int]] = [
            tokenizer.encode(s, add_special_tokens=True, max_len=max_seq_len)
            for s in self.sequences
        ]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.encoded[idx], dtype=torch.long)


class MLMCollator:
    """Pad a batch and apply the 80/10/10 BERT masking scheme.

    Produces a dict of:
      input_ids      (B, L) corrupted tokens fed to the model
      attention_mask (B, L) 1 for real tokens, 0 for [PAD]
      labels         (B, L) original ids at predicted positions, IGNORE_INDEX elsewhere
    """

    def __init__(
        self,
        tokenizer: ProteinTokenizer = default_tokenizer,
        mlm_probability: float = config.mlm_probability,
        mask_token_prob: float = config.mask_token_prob,
        random_token_prob: float = config.random_token_prob,
    ) -> None:
        if mask_token_prob + random_token_prob > 1.0:
            raise ValueError("mask_token_prob + random_token_prob must be <= 1.0")

        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_prob = mask_token_prob
        # Probability of a *random* replacement conditioned on "not [MASK]", so
        # the two independent draws below compose to the intended 80/10/10:
        # 0.10 / (1 - 0.80) = 0.5 with the defaults.
        remainder = 1.0 - mask_token_prob
        self.random_given_not_mask = (
            random_token_prob / remainder if remainder > 0 else 0.0
        )

        self.special_ids = torch.tensor(
            sorted(tokenizer.special_token_ids), dtype=torch.long
        )
        # Random replacements come from real residues only — never a special token.
        self.regular_ids = torch.tensor(tokenizer.regular_token_ids, dtype=torch.long)

    def __call__(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        max_len = max(seq.size(0) for seq in batch)
        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, seq in enumerate(batch):
            input_ids[i, : seq.size(0)] = seq
            attention_mask[i, : seq.size(0)] = 1

        input_ids, labels = self.mask_tokens(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def mask_tokens(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Corrupt `mlm_probability` of the non-special tokens and build labels."""
        labels = input_ids.clone()

        # [PAD]/[CLS]/[SEP]/[MASK]/[UNK] are never predicted or corrupted.
        special_mask = torch.isin(input_ids, self.special_ids)
        prob_matrix = torch.full(labels.shape, self.mlm_probability)
        prob_matrix.masked_fill_(special_mask, 0.0)
        selected = torch.bernoulli(prob_matrix).bool()

        labels[~selected] = IGNORE_INDEX

        # 80% of selected -> [MASK]
        replaced = (
            torch.bernoulli(torch.full(labels.shape, self.mask_token_prob)).bool()
            & selected
        )
        input_ids[replaced] = self.tokenizer.mask_token_id

        # 10% of selected -> a random real residue (half of the remaining 20%)
        randomized = (
            torch.bernoulli(
                torch.full(labels.shape, self.random_given_not_mask)
            ).bool()
            & selected
            & ~replaced
        )
        n_random = int(randomized.sum())
        if n_random:
            picks = torch.randint(len(self.regular_ids), (n_random,))
            input_ids[randomized] = self.regular_ids[picks]

        # The remaining 10% of selected keep their original token.
        return input_ids, labels


def make_dataloader(
    path: str,
    shuffle: bool,
    tokenizer: ProteinTokenizer = default_tokenizer,
    batch_size: int = config.batch_size,
    num_workers: int = config.num_workers,
) -> DataLoader:
    dataset = ProteinDataset(path, tokenizer=tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=MLMCollator(tokenizer=tokenizer),
        drop_last=False,
    )


def get_dataloaders(
    tokenizer: ProteinTokenizer = default_tokenizer,
) -> Tuple[DataLoader, DataLoader]:
    """Train (shuffled) and validation (ordered) loaders from the config paths."""
    return (
        make_dataloader(config.train_file, shuffle=True, tokenizer=tokenizer),
        make_dataloader(config.val_file, shuffle=False, tokenizer=tokenizer),
    )


if __name__ == "__main__":
    torch.manual_seed(config.seed)

    train_loader, val_loader = get_dataloaders()
    print(f"train sequences : {len(train_loader.dataset)}")
    print(f"val sequences   : {len(val_loader.dataset)}")

    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    print(f"input_ids       : {tuple(input_ids.shape)}")
    print(f"attention_mask  : {tuple(attention_mask.shape)}")
    print(f"labels          : {tuple(labels.shape)}")

    # ---- invariants ----
    predicted = labels != IGNORE_INDEX
    real = attention_mask.bool()
    assert not (predicted & ~real).any(), "padding was selected for prediction"

    special = torch.isin(labels, torch.tensor(sorted(default_tokenizer.special_token_ids)))
    assert not (predicted & special).any(), "a special token was selected for prediction"

    unchanged_ok = (input_ids[~predicted] != default_tokenizer.mask_token_id).all()
    assert unchanged_ok, "an unselected position was corrupted to [MASK]"

    kept = predicted & (input_ids == labels)
    masked = predicted & (input_ids == default_tokenizer.mask_token_id)
    randomized = predicted & ~kept & ~masked

    n_pred, n_real = int(predicted.sum()), int(real.sum())
    print(f"\nselected        : {n_pred}/{n_real} = {n_pred / n_real:.3f} "
          f"(target {config.mlm_probability})")
    print(f"  -> [MASK]     : {int(masked.sum()) / n_pred:.3f} "
          f"(target {config.mask_token_prob})")
    print(f"  -> random     : {int(randomized.sum()) / n_pred:.3f} "
          f"(target {config.random_token_prob})")
    print(f"  -> unchanged  : {int(kept.sum()) / n_pred:.3f} "
          f"(target {1 - config.mask_token_prob - config.random_token_prob:.2f})")

    # Dynamic masking: the same sequence gets a different mask on a second pass.
    collator = MLMCollator()
    sample = train_loader.dataset[0]
    first = collator([sample])["input_ids"]
    second = collator([sample])["input_ids"]
    print(f"\ndynamic masking : two passes differ = {not torch.equal(first, second)}")

    print("\nexample (first 40 tokens of row 0)")
    print("  input  :", default_tokenizer.decode(input_ids[0][:40], skip_special_tokens=False))
    print("  labels :", "".join(
        default_tokenizer.id_to_token[int(t)] if t != IGNORE_INDEX else "."
        for t in labels[0][:40]
    ))
