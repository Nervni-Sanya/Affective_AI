"""Tokenizer plumbing shared by the encoder and the backbone.

Two modes:

* **pretrained** – real HuggingFace tokenizers (``distilbert-base-uncased`` /
  ``gpt2``). Requires a one-time download.
* **tiny / offline** – :class:`SimpleByteTokenizer`, a dependency-free byte-level
  tokenizer so tests and the demo run with no network access.

Both expose the minimal surface the rest of the codebase relies on:
``encode``, ``decode``, ``pad_token_id``, ``eos_token_id``, ``vocab_size``.
"""

from __future__ import annotations

from typing import List, Sequence

import torch


class SimpleByteTokenizer:
    """A tiny, offline, byte-level tokenizer.

    Text is encoded to its UTF-8 bytes (ids 0-255); ids 256-259 are reserved
    special tokens. Lossless for any UTF-8 string. Produces gibberish when fed
    through a random-init model, which is expected in tiny mode — the point is
    that the pipeline runs end-to-end without downloads.
    """

    PAD_ID = 256
    BOS_ID = 257
    EOS_ID = 258
    SEP_ID = 259

    def __init__(self) -> None:
        self.pad_token_id = self.PAD_ID
        self.bos_token_id = self.BOS_ID
        self.eos_token_id = self.EOS_ID
        self.sep_token_id = self.SEP_ID
        self.vocab_size = 260

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = list(text.encode("utf-8"))
        if add_special_tokens:
            ids = [self.BOS_ID] + ids + [self.EOS_ID]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        out = bytearray()
        for i in ids:
            i = int(i)
            if i < 256:
                out.append(i)
            elif not skip_special_tokens:
                # Render specials as nothing meaningful; keep decode total.
                continue
        return out.decode("utf-8", errors="replace")


def build_encoder_tokenizer(model_cfg):
    """Tokenizer for the Affect Core encoder."""
    if model_cfg.encoder_pretrained:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_cfg.encoder_name)
    return SimpleByteTokenizer()


def build_backbone_tokenizer(model_cfg):
    """Tokenizer for the generative backbone (pad token guaranteed)."""
    if model_cfg.backbone_pretrained:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_cfg.backbone_name)
        if tok.pad_token_id is None:
            # GPT-2 has no pad token; reuse EOS so batching/padding works.
            tok.pad_token = tok.eos_token
        return tok
    return SimpleByteTokenizer()


def batch_encode(
    tokenizer,
    texts: Sequence[str],
    max_length: int,
    device=None,
):
    """Encode a batch of strings into padded ``(input_ids, attention_mask)``.

    Works for both HF tokenizers and :class:`SimpleByteTokenizer` by going
    through ``encode`` + manual right-padding, so behaviour is identical across
    backends (no reliance on per-tokenizer ``__call__`` padding semantics).
    """
    pad_id = tokenizer.pad_token_id
    seqs = []
    for t in texts:
        ids = tokenizer.encode(t or "", add_special_tokens=False)[:max_length]
        if not ids:  # never feed an empty sequence to the encoder
            ids = [pad_id]
        seqs.append(ids)
    width = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, : len(s)] = 1
    if device is not None:
        input_ids = input_ids.to(device)
        attn = attn.to(device)
    return input_ids, attn
