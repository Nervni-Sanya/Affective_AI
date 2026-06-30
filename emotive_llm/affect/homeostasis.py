"""Affect safety / homeostasis (spec §7).

Two mechanisms keep the state from getting "stuck" in an extreme:

* :func:`homeostasis_pull` — a gentle per-step relaxation toward a light-positive
  neutral target. Also usable as a training regulariser.
* :class:`ExtremeDwellGuard` — a stateful runtime guard that applies a stronger
  pull once the state has dwelt in the high-distress corner for more than ``n``
  consecutive steps.
"""

from __future__ import annotations

import torch

from emotive_llm.affect import space


def homeostasis_pull(state: torch.Tensor, strength: float = 0.05) -> torch.Tensor:
    """Move ``state`` a fraction ``strength`` of the way toward the neutral target.

    ``strength`` of 0 leaves the state unchanged; 1 snaps it to the target.
    """
    target = space.neutral_target(device=state.device, dtype=state.dtype)
    pulled = state + strength * (target - state)
    return space.clamp_to_bounds(pulled)


def homeostasis_reg(state: torch.Tensor) -> torch.Tensor:
    """Mean squared distance from the neutral target — a scalar reg term.

    Added (weighted) to the Phase 2 loss to discourage runaway dynamics.
    """
    target = space.neutral_target(device=state.device, dtype=state.dtype)
    return ((state - target) ** 2).mean()


class ExtremeDwellGuard:
    """Force a relaxation once the state stays extreme for too long.

    Tracks a per-call counter of consecutive extreme steps. Once it exceeds
    ``max_steps``, a strong homeostasis pull is applied and the counter resets.
    Intended for the single-stream inference loop (batch size 1).
    """

    def __init__(self, max_steps: int = 3, relax_strength: float = 0.5):
        self.max_steps = max_steps
        self.relax_strength = relax_strength
        self._dwell = 0

    def reset(self) -> None:
        self._dwell = 0

    @property
    def dwell(self) -> int:
        return self._dwell

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """Update the dwell counter and relax the state if it has been extreme too long."""
        extreme = bool(space.is_extreme(state).any().item())
        if extreme:
            self._dwell += 1
        else:
            self._dwell = 0

        if self._dwell > self.max_steps:
            state = homeostasis_pull(state, strength=self.relax_strength)
            self._dwell = 0
        return state
