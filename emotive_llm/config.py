"""Configuration objects for EmotiveLLM.

This module is deliberately free of heavy imports (no ``torch``) so it can be
imported cheaply by tooling and tests. Device/AMP helpers import ``torch``
lazily inside the function body.

Configs are plain dataclasses that can be built from a YAML file via
:func:`load_config`. Three reference configs live in ``configs/``:

* ``tiny``  – random-init mini models, runs offline on CPU (tests/demo).
* ``base``  – real ``distilbert-base`` + ``gpt2``, single-GPU defaults.
* ``prod``  – production-scale recipe (real data, AMP, grad-accum).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional

# Dimensionality of the affect vector (VAD + social drivers + resilience).
AFFECT_DIM = 7


@dataclass
class ModelConfig:
    """Architecture sizing for the encoder, backbone, and conditioning."""

    # --- Affect Core encoder (DistilBERT-shaped) ---
    encoder_name: str = "distilbert-base-uncased"
    encoder_pretrained: bool = False
    # Tiny random-init dims (used when ``encoder_pretrained`` is False).
    encoder_dim: int = 64
    encoder_layers: int = 2
    encoder_heads: int = 2

    # --- Generative backbone (GPT-2-shaped) ---
    backbone_name: str = "gpt2"
    backbone_pretrained: bool = False
    # Tiny random-init dims (used when ``backbone_pretrained`` is False).
    backbone_dim: int = 64
    backbone_layers: int = 2
    backbone_heads: int = 2
    backbone_max_positions: int = 256

    # --- Conditioning projector (S_t -> soft prompt) ---
    proj_hidden: int = 128
    n_prefix: int = 1

    # --- Affect Core dynamics ---
    # Per-axis base inertia alpha. Valence is "stickier" (0.9), Arousal cools
    # fast (0.6); resilience itself is very inertial (0.95).
    alpha_init: tuple = (0.9, 0.6, 0.7, 0.8, 0.7, 0.8, 0.95)
    # Fatigue gain: alpha_eff = alpha_base * (fatigue_beta + (1-fatigue_beta)*resilience).
    fatigue_beta: float = 0.5

    @property
    def pretrained(self) -> bool:
        """True only when *both* sub-models load real weights."""
        return self.encoder_pretrained and self.backbone_pretrained

    def set_pretrained(self, value: bool) -> None:
        self.encoder_pretrained = value
        self.backbone_pretrained = value


@dataclass
class TrainConfig:
    """Optimisation + bookkeeping shared by Phase 2 and Phase 3."""

    lr: float = 5e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    epochs: int = 1
    max_steps: int = -1            # -1 = run full epochs
    grad_accum: int = 1
    grad_clip: float = 1.0
    amp: str = "off"               # off | fp16 | bf16
    warmup_steps: int = 0
    log_every: int = 10
    seed: int = 42
    max_seq_len: int = 128
    out_dir: str = "checkpoints"
    # Weight of the homeostasis regulariser added to the Phase 2 MSE loss.
    homeostasis_reg: float = 0.0
    device: str = "auto"           # auto | cpu | cuda


@dataclass
class DataConfig:
    """Where dialogue data comes from and how it is annotated."""

    source: str = "synthetic"      # synthetic | dailydialog
    num_synthetic: int = 256       # records to generate when source == synthetic
    max_dialogues: int = 256       # cap when loading a real corpus
    annotate: str = "auto"         # auto | llm | synthetic
    judge_model: str = "claude-opus-4-8"
    data_dir: str = "data_out"
    seed: int = 42


@dataclass
class Config:
    """Top-level bundle: model + training + data, plus a human-readable name."""

    name: str = "tiny"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def _build_dataclass(cls, payload: Optional[dict]):
    """Construct ``cls`` from a (possibly nested) dict, ignoring unknown keys."""
    if payload is None:
        return cls()
    known = {f.name: f for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        f = known[key]
        # YAML has no tuple type; coerce lists into tuples for tuple-typed fields.
        if isinstance(f.default, tuple) and isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str) -> Config:
    """Load a :class:`Config` from a YAML file.

    The YAML may set any subset of fields; unspecified fields fall back to the
    dataclass defaults. Top-level keys are ``name``, ``model``, ``train``,
    ``data``.
    """
    import yaml  # local import keeps module import cheap

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return Config(
        name=raw.get("name", "config"),
        model=_build_dataclass(ModelConfig, raw.get("model")),
        train=_build_dataclass(TrainConfig, raw.get("train")),
        data=_build_dataclass(DataConfig, raw.get("data")),
    )


def resolve_device(prefer: str = "auto") -> str:
    """Return a concrete device string ('cpu' or 'cuda')."""
    import torch

    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def amp_dtype(amp: str):
    """Map an AMP setting to a torch dtype, or None when disabled."""
    import torch

    return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
