"""Conditioned generative backbone (spec §4.2).

Wraps a GPT-like causal LM and injects the affect state as a soft prompt:
projector(S_t) produces ``n_prefix`` prefix embeddings that are prepended to the
token embeddings. Training is plain cross-entropy on the reply tokens only —
the affect signal enters purely as *input* conditioning, never as a loss
coefficient (spec §5/§8).

Generation uses a self-contained greedy/sampling decode loop (re-running the
full sequence each step) rather than ``model.generate`` with ``inputs_embeds``,
which keeps it robust across transformers versions and tiny CPU models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from emotive_llm.config import ModelConfig
from emotive_llm.conditioning.projector import AffectProjector
from emotive_llm.tokenization import build_backbone_tokenizer


def _build_backbone(model_cfg: ModelConfig, vocab_size: int):
    """Return (lm_module, d_model). Pretrained or tiny random-init GPT-2."""
    if model_cfg.backbone_pretrained:
        from transformers import AutoModelForCausalLM

        lm = AutoModelForCausalLM.from_pretrained(model_cfg.backbone_name)
        cfg = lm.config
        d_model = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd")
        return lm, int(d_model)

    from transformers import GPT2Config, GPT2LMHeadModel
    from emotive_llm.tokenization import SimpleByteTokenizer

    dim = model_cfg.backbone_dim
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=model_cfg.backbone_max_positions,
        n_embd=dim,
        n_layer=model_cfg.backbone_layers,
        n_head=model_cfg.backbone_heads,
        n_inner=4 * dim,
        bos_token_id=SimpleByteTokenizer.BOS_ID,
        eos_token_id=SimpleByteTokenizer.EOS_ID,
    )
    return GPT2LMHeadModel(cfg), dim


@dataclass
class GenerationOutput:
    text: List[str]
    token_ids: List[List[int]]


class ConditionedLM(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.tokenizer = build_backbone_tokenizer(model_cfg)
        vocab_size = (
            self.tokenizer.vocab_size
            if not model_cfg.backbone_pretrained
            else len(self.tokenizer)
        )
        self.backbone, self.d_model = _build_backbone(model_cfg, vocab_size)
        self.projector = AffectProjector(
            self.d_model, hidden=model_cfg.proj_hidden, n_prefix=model_cfg.n_prefix
        )
        self.n_prefix = model_cfg.n_prefix

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    # -- internals ---------------------------------------------------------
    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_input_embeddings()(input_ids)

    def _prefixed(self, input_ids, attention_mask, affect_state, labels):
        """Build (inputs_embeds, attention_mask, labels) with the affect prefix."""
        device = self.device
        input_ids = input_ids.to(device)
        token_embeds = self._embed_tokens(input_ids)                 # (B, T, d)
        prefix = self.projector(affect_state).to(token_embeds.dtype)  # (B, P, d)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

        b, p = input_ids.shape[0], self.n_prefix
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device)
        prefix_mask = torch.ones((b, p), dtype=attention_mask.dtype, device=device)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        full_labels = None
        if labels is not None:
            labels = labels.to(device)
            prefix_labels = torch.full((b, p), -100, dtype=labels.dtype, device=device)
            full_labels = torch.cat([prefix_labels, labels], dim=1)

        return inputs_embeds, full_mask, full_labels

    # -- training ----------------------------------------------------------
    def forward(self, input_ids, attention_mask=None, affect_state=None, labels=None):
        """Forward pass with affect conditioning; returns the HF output (``.loss``)."""
        inputs_embeds, full_mask, full_labels = self._prefixed(
            input_ids, attention_mask, affect_state, labels
        )
        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
        )

    # -- generation --------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        affect_state: torch.Tensor,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Greedy/sampling decode. Returns newly generated token ids ``(B, n)``."""
        self.eval()
        device = self.device
        cur = input_ids.to(device)
        b = cur.shape[0]
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        prefix = self.projector(affect_state).to(self._embed_tokens(cur).dtype)

        finished = torch.zeros(b, dtype=torch.bool, device=device)
        generated: List[List[int]] = [[] for _ in range(b)]

        for _ in range(max_new_tokens):
            token_embeds = self._embed_tokens(cur)
            inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
            attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
            logits = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attn).logits
            next_logits = logits[:, -1, :]                          # (B, vocab)

            if do_sample:
                next_logits = next_logits / max(temperature, 1e-5)
                if top_k and top_k > 0:
                    k = min(top_k, next_logits.shape[-1])
                    kth = torch.topk(next_logits, k, dim=-1).values[:, -1, None]
                    next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = next_logits.argmax(dim=-1)

            for i in range(b):
                if not finished[i]:
                    tok = int(next_token[i].item())
                    generated[i].append(tok)
                    if eos_token_id is not None and tok == eos_token_id:
                        finished[i] = True
            cur = torch.cat([cur, next_token.unsqueeze(1)], dim=1)
            if bool(finished.all().item()):
                break

        max_len = max((len(g) for g in generated), default=0)
        out = torch.full((b, max_len), eos_token_id or 0, dtype=torch.long, device=device)
        for i, g in enumerate(generated):
            if g:
                out[i, : len(g)] = torch.tensor(g, dtype=torch.long, device=device)
        return out

    @torch.no_grad()
    def generate_text(
        self, prompt: str, affect_state: torch.Tensor, **kwargs
    ) -> str:
        """Tokenise ``prompt``, generate, and decode only the new tokens."""
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            ids = [self.tokenizer.eos_token_id or 0]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        gen = self.generate(input_ids, affect_state, **kwargs)
        toks = [t for t in gen[0].tolist()]
        # Trim a trailing EOS for cleaner output.
        if toks and self.tokenizer.eos_token_id is not None and toks[-1] == self.tokenizer.eos_token_id:
            toks = toks[:-1]
        return self.tokenizer.decode(toks, skip_special_tokens=True)
