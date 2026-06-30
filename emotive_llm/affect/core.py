"""Affect Core: the recurrent block that evolves the 7-D state (spec §4.1).

A light encoder turns the latest utterance into an embedding ``h_t``; a
decay cell blends it with the previous state using per-axis inertia ``alpha``:

    S_t = bounded( alpha_eff * S_{t-1} + (1 - alpha_eff) * (W h_t + b) )

* ``alpha`` is a learnable per-axis vector (init: Valence 0.9 "sticky",
  Arousal 0.6 "cools fast").
* ``alpha_eff`` is modulated by the **resilience** axis to model fatigue:
  when depleted, inertia drops and the state becomes more volatile.
* ``resilience`` itself has hand-designed dynamics: it is spent by emotional
  exertion (high arousal / negative valence) and recovers when calm.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from emotive_llm.affect import space
from emotive_llm.config import ModelConfig
from emotive_llm.tokenization import batch_encode, build_encoder_tokenizer


def _build_encoder(model_cfg: ModelConfig):
    """Return (encoder_module, hidden_dim). Pretrained or tiny random-init."""
    if model_cfg.encoder_pretrained:
        from transformers import AutoModel

        enc = AutoModel.from_pretrained(model_cfg.encoder_name)
        cfg = enc.config
        dim = getattr(cfg, "hidden_size", None) or getattr(cfg, "dim")
        return enc, int(dim)

    # Tiny random-init DistilBERT-shaped encoder (no download).
    from transformers import DistilBertConfig, DistilBertModel
    from emotive_llm.tokenization import SimpleByteTokenizer

    dim = model_cfg.encoder_dim
    cfg = DistilBertConfig(
        vocab_size=SimpleByteTokenizer().vocab_size,
        dim=dim,
        hidden_dim=4 * dim,
        n_layers=model_cfg.encoder_layers,
        n_heads=model_cfg.encoder_heads,
        max_position_embeddings=512,
    )
    return DistilBertModel(cfg), dim


class AffectCore(nn.Module):
    """Recurrent affect-dynamics module."""

    # Resilience dynamics (heuristic, documented constants).
    EXERTION_DECAY = 0.10   # resilience spent per unit exertion
    CALM_RECOVERY = 0.05    # resilience regained per unit calm

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.tokenizer = build_encoder_tokenizer(model_cfg)
        self.encoder, enc_dim = _build_encoder(model_cfg)
        self.enc_dim = enc_dim
        self.max_len = 64

        # W h_t + b  ->  per-axis drive toward a target state.
        self.proj = nn.Linear(enc_dim, space.DIM)

        # alpha = sigmoid(alpha_raw), initialised to the configured base inertia.
        alpha_init = torch.tensor(model_cfg.alpha_init, dtype=torch.float32).clamp(1e-3, 1 - 1e-3)
        alpha_raw = torch.log(alpha_init / (1 - alpha_init))  # inverse sigmoid
        self.alpha_raw = nn.Parameter(alpha_raw)
        self.fatigue_beta = float(model_cfg.fatigue_beta)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def alpha_base(self) -> torch.Tensor:
        """Per-axis base inertia in (0, 1), shape ``(7,)``."""
        return torch.sigmoid(self.alpha_raw)

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Mean-pool the encoder over tokens -> ``(B, enc_dim)``."""
        input_ids, attn = batch_encode(self.tokenizer, texts, self.max_len, self.device)
        out = self.encoder(input_ids=input_ids, attention_mask=attn)
        hidden = out.last_hidden_state                      # (B, T, dim)
        mask = attn.unsqueeze(-1).to(hidden.dtype)          # (B, T, 1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return pooled

    def forward(
        self, s_prev: torch.Tensor, texts: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance the state given the previous state and the latest utterance.

        Returns ``(s_new, delta)`` where ``delta = s_new - s_prev``.
        """
        s_prev = s_prev.to(self.device)
        h = self.encode(texts)
        target = self.proj(h)                               # (B, 7) drive

        # Fatigue gate: low resilience -> lower effective inertia -> volatility.
        resilience_prev = s_prev[:, space.RESILIENCE_IDX]   # (B,)
        gate = self.fatigue_beta + (1.0 - self.fatigue_beta) * resilience_prev
        alpha_eff = self.alpha_base().unsqueeze(0) * gate.unsqueeze(1)  # (B, 7)

        raw = alpha_eff * s_prev + (1.0 - alpha_eff) * target
        s_new = space.bounded_activation(raw)               # tanh / sigmoid per axis

        # Override resilience with its own fatigue/recovery dynamics.
        valence_new = s_new[:, space.AXIS_INDEX["valence"]]
        arousal_new = s_new[:, space.AXIS_INDEX["arousal"]]
        exertion = 0.5 * arousal_new.abs() + 0.5 * (-valence_new).clamp(min=0.0)  # [0,1]
        resilience_new = (
            resilience_prev
            - self.EXERTION_DECAY * exertion
            + self.CALM_RECOVERY * (1.0 - exertion)
        ).clamp(0.0, 1.0)

        s_new = s_new.clone()
        s_new[:, space.RESILIENCE_IDX] = resilience_new
        s_new = space.clamp_to_bounds(s_new)

        return s_new, s_new - s_prev

    @torch.no_grad()
    def step(self, s_prev: torch.Tensor, text: str) -> torch.Tensor:
        """Convenience single-step update used at inference time."""
        self.eval()
        s_new, _ = self.forward(s_prev, [text])
        return s_new
