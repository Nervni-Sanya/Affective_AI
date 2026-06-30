"""Torch datasets/collation for the two trainable phases.

Affect-state semantics used throughout:

* ``s_before`` — the assistant's state as the user's message arrives.
* ``s_after``  — the state after reading the message; this is the state that
  conditions the generated reply (call it ``S_t``).

Phase 2 (Affect Core) learns ``(s_before, user_text) -> s_after`` (predict ΔS).
Phase 3 (SFT) conditions generation on ``s_after`` and learns the reply tokens.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch.utils.data import Dataset

from emotive_llm.config import DataConfig
from emotive_llm.data.schema import AffectRecord


# --- Phase 2: Affect Core (ΔS) ------------------------------------------
class AffectCoreDataset(Dataset):
    def __init__(self, records: Sequence[AffectRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records[i]
        return {
            "user_text": r.user_text,
            "s_before": torch.tensor(r.s_before, dtype=torch.float32),
            "s_after": torch.tensor(r.s_after, dtype=torch.float32),
        }


def affect_collate(batch):
    return {
        "texts": [b["user_text"] for b in batch],
        "s_before": torch.stack([b["s_before"] for b in batch]),
        "s_after": torch.stack([b["s_after"] for b in batch]),
    }


# --- Phase 3: SFT (conditioned cross-entropy) ---------------------------
class SFTDataset(Dataset):
    def __init__(self, records: Sequence[AffectRecord], tokenizer, max_seq_len: int = 128):
        self.records = list(records)
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_id = tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records[i]
        prompt_ids = self.tok.encode(r.user_text + "\n", add_special_tokens=False)
        reply_ids = self.tok.encode(r.assistant_text, add_special_tokens=False)
        if self.eos_id is not None:
            reply_ids = reply_ids + [self.eos_id]

        # Truncate from the left of the prompt if the pair is too long, keeping
        # the whole reply (which carries the loss-bearing tokens).
        budget = self.max_seq_len
        reply_ids = reply_ids[:budget]
        prompt_ids = prompt_ids[: max(0, budget - len(reply_ids))]

        input_ids = prompt_ids + reply_ids
        labels = [-100] * len(prompt_ids) + reply_ids
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            # s_after conditions the reply.
            "affect_state": torch.tensor(r.s_after, dtype=torch.float32),
        }


def make_sft_collate(pad_id: int):
    def sft_collate(batch):
        width = max(b["input_ids"].shape[0] for b in batch)
        n = len(batch)
        input_ids = torch.full((n, width), pad_id, dtype=torch.long)
        labels = torch.full((n, width), -100, dtype=torch.long)
        attention_mask = torch.zeros((n, width), dtype=torch.long)
        for i, b in enumerate(batch):
            L = b["input_ids"].shape[0]
            input_ids[i, :L] = b["input_ids"]
            labels[i, :L] = b["labels"]
            attention_mask[i, :L] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "affect_state": torch.stack([b["affect_state"] for b in batch]),
        }

    return sft_collate


# --- Record assembly from config ----------------------------------------
def build_records(data_cfg: DataConfig) -> List[AffectRecord]:
    """Produce annotated records per the data config (synthetic or DailyDialog)."""
    if data_cfg.source == "synthetic":
        from emotive_llm.data.synthetic import generate_records

        return generate_records(data_cfg.num_synthetic, seed=data_cfg.seed)

    if data_cfg.source == "dailydialog":
        from emotive_llm.data.annotate import annotate_pairs
        from emotive_llm.data.real import load_dailydialog_pairs

        pairs = load_dailydialog_pairs(max_dialogues=data_cfg.max_dialogues)
        return annotate_pairs(
            pairs, mode=data_cfg.annotate, judge_model=data_cfg.judge_model, seed=data_cfg.seed
        )

    raise ValueError(f"Unknown data source: {data_cfg.source}")
