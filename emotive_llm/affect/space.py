"""The Affect Space: a 7-dimensional, interpretable emotional state.

Axes (VAD + social drivers + a service inertia axis), per the specification:

    0 valence     [-1, 1]  negative -> positive
    1 arousal     [-1, 1]  sleepy   -> alert
    2 dominance   [-1, 1]  submissive -> in control
    3 warmth      [-1, 1]  hostile  -> affectionate   (social)
    4 interest    [-1, 1]  bored    -> curious        (social)
    5 honesty     [-1, 1]  guarded  -> candid         (social)
    6 resilience  [ 0, 1]  depleted -> fresh          (service: fatigue/inertia)

The first six are bipolar; ``resilience`` is unipolar and modulates the inertia
of the recurrent core (fatigue). The neutral start state ``S_0`` is a mild
positive social baseline.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

# --- Axis layout ---------------------------------------------------------
AXES: List[str] = [
    "valence",
    "arousal",
    "dominance",
    "warmth",
    "interest",
    "honesty",
    "resilience",
]
AXIS_INDEX = {name: i for i, name in enumerate(AXES)}
DIM = len(AXES)
RESILIENCE_IDX = AXIS_INDEX["resilience"]

# Per-axis bounds. Axes 0-5 are bipolar; resilience is unipolar.
BOUNDS_LOW: tuple = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0)
BOUNDS_HIGH: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

# Neutral starting state S_0 (spec §6): mild positive social baseline, fresh.
S0: tuple = (0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0)

# Light-positive homeostatic target the system gently relaxes toward (spec §7).
NEUTRAL_TARGET: tuple = (0.15, 0.0, 0.0, 0.5, 0.4, 0.5, 1.0)

# Human-readable affect prototypes for labelling / metrics. Each is a full
# 7-vector; nearest-prototype matching uses the six expressive axes only.
PROTOTYPES: List[tuple] = [
    ("neutral", S0),
    ("calm / content", (0.4, -0.2, 0.1, 0.5, 0.3, 0.6, 0.9)),
    ("enthusiastic", (0.7, 0.7, 0.3, 0.6, 0.9, 0.6, 0.8)),
    ("affectionate", (0.6, 0.2, -0.1, 0.9, 0.5, 0.7, 0.8)),
    ("irritated but polite", (-0.4, 0.5, 0.3, 0.0, 0.2, 0.5, 0.6)),
    ("angry", (-0.8, 0.8, 0.5, -0.7, 0.3, 0.6, 0.4)),
    ("anxious", (-0.4, 0.6, -0.5, 0.1, 0.3, 0.3, 0.4)),
    ("sad / withdrawn", (-0.6, -0.5, -0.4, -0.1, -0.3, 0.4, 0.5)),
    ("bored", (-0.1, -0.6, 0.0, 0.1, -0.8, 0.4, 0.7)),
]

# Thresholds for the "stuck in an extreme" safety guard (spec §7).
EXTREME_VALENCE = -0.8
EXTREME_AROUSAL = 0.8


def _bounds_tensors(device=None, dtype=torch.float32):
    low = torch.tensor(BOUNDS_LOW, device=device, dtype=dtype)
    high = torch.tensor(BOUNDS_HIGH, device=device, dtype=dtype)
    return low, high


def to_tensor(values: Sequence[float], device=None, dtype=torch.float32) -> torch.Tensor:
    """Turn a length-7 sequence into a ``(1, 7)`` tensor."""
    t = torch.as_tensor(values, device=device, dtype=dtype)
    if t.shape[-1] != DIM:
        raise ValueError(f"Affect vector must have {DIM} dims, got {tuple(t.shape)}")
    return t.reshape(-1, DIM)


def default_state(batch_size: int = 1, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return ``S_0`` broadcast to ``(batch_size, 7)``."""
    return to_tensor(S0, device=device, dtype=dtype).expand(batch_size, DIM).clone()


def clamp_to_bounds(state: torch.Tensor) -> torch.Tensor:
    """Clamp each axis into its valid range."""
    low, high = _bounds_tensors(device=state.device, dtype=state.dtype)
    return torch.maximum(torch.minimum(state, high), low)


def bounded_activation(raw: torch.Tensor) -> torch.Tensor:
    """Squash a raw ``(B, 7)`` pre-activation into the valid ranges.

    ``tanh`` -> (-1, 1) for the six bipolar axes; ``sigmoid`` -> (0, 1) for
    resilience. This keeps every produced state inside bounds by construction.
    """
    out = torch.tanh(raw)
    res = torch.sigmoid(raw[..., RESILIENCE_IDX])
    out = out.clone()
    out[..., RESILIENCE_IDX] = res
    return out


def is_extreme(state: torch.Tensor) -> torch.Tensor:
    """Boolean mask ``(B,)``: state is in the high-distress / aggression corner."""
    valence = state[..., AXIS_INDEX["valence"]]
    arousal = state[..., AXIS_INDEX["arousal"]]
    return (valence < EXTREME_VALENCE) & (arousal > EXTREME_AROUSAL)


def nearest_prototype(state: Sequence[float] | torch.Tensor) -> str:
    """Label a single affect vector by its nearest prototype (6 expressive axes)."""
    vec = torch.as_tensor(state, dtype=torch.float32).reshape(-1)[:DIM]
    protos = torch.tensor([p[1] for p in PROTOTYPES], dtype=torch.float32)
    # Distance over the six expressive axes; resilience excluded.
    dist = torch.linalg.vector_norm(protos[:, :RESILIENCE_IDX] - vec[:RESILIENCE_IDX], dim=1)
    idx = int(torch.argmin(dist).item())
    return PROTOTYPES[idx][0]


def neutral_target(device=None, dtype=torch.float32) -> torch.Tensor:
    """Return the light-positive homeostatic target as ``(1, 7)``."""
    return to_tensor(NEUTRAL_TARGET, device=device, dtype=dtype)


class AffectSpace:
    """Namespace bundling the affect-space constants and helpers.

    Provided for ergonomic imports; every method delegates to the module-level
    functions above.
    """

    AXES = AXES
    DIM = DIM
    S0 = S0
    NEUTRAL_TARGET = NEUTRAL_TARGET
    PROTOTYPES = PROTOTYPES

    default_state = staticmethod(default_state)
    clamp_to_bounds = staticmethod(clamp_to_bounds)
    bounded_activation = staticmethod(bounded_activation)
    is_extreme = staticmethod(is_extreme)
    nearest_prototype = staticmethod(nearest_prototype)
    neutral_target = staticmethod(neutral_target)
    to_tensor = staticmethod(to_tensor)
