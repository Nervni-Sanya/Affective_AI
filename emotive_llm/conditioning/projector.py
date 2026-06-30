"""Affect -> soft prompt projector (spec §4.2).

The 7-D state is mapped through a small MLP into one or more prefix embeddings
that live in the backbone's ``d_model`` space:

    S_t (7) -> Linear(7, hidden) -> GELU -> Linear(hidden, n_prefix * d_model)
            -> reshape (B, n_prefix, d_model)

These prefix embeddings are prepended to the token embeddings, so the model
learns to associate affect configurations (e.g. high Arousal + low Valence)
with lexical/stylistic patterns. This is the "soft prompting via projection"
integration the specification prefers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from emotive_llm.affect import space


class AffectProjector(nn.Module):
    def __init__(self, d_model: int, hidden: int = 128, n_prefix: int = 1):
        super().__init__()
        self.d_model = d_model
        self.n_prefix = n_prefix
        self.net = nn.Sequential(
            nn.Linear(space.DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_prefix * d_model),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """``(B, 7)`` -> ``(B, n_prefix, d_model)`` prefix embeddings."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        device = self.net[0].weight.device
        state = state.to(device=device, dtype=self.net[0].weight.dtype)
        flat = self.net(state)                       # (B, n_prefix * d_model)
        return flat.view(state.shape[0], self.n_prefix, self.d_model)
