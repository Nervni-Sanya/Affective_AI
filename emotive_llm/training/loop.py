"""Shared training utilities: seeding, optimisation loop, AMP, checkpoints.

The loop is generic over a ``loss_fn(model, batch, device) -> scalar tensor`` so
Phase 2 and Phase 3 share the same optimisation/AMP/grad-accum machinery.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, List

import torch

from emotive_llm.config import Config, TrainConfig, amp_dtype, load_config, resolve_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )


def train_loop(
    model: torch.nn.Module,
    dataloader,
    loss_fn: Callable,
    cfg: TrainConfig,
    device: str,
    desc: str = "train",
) -> List[float]:
    """Run optimisation; returns the per-micro-batch loss history."""
    model.to(device)
    model.train()
    opt = get_optimizer(model, cfg)

    use_amp = cfg.amp in ("fp16", "bf16") and device.startswith("cuda")
    dtype = amp_dtype(cfg.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp == "fp16" and use_amp))

    losses: List[float] = []
    opt_step = 0
    opt.zero_grad(set_to_none=True)

    for epoch in range(cfg.epochs):
        for i, batch in enumerate(dataloader):
            if use_amp:
                with torch.amp.autocast("cuda", dtype=dtype):
                    loss = loss_fn(model, batch, device)
            else:
                loss = loss_fn(model, batch, device)

            losses.append(float(loss.detach().item()))
            scaled = loss / cfg.grad_accum
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            if (i + 1) % cfg.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % cfg.log_every == 0:
                    recent = sum(losses[-cfg.log_every:]) / min(len(losses), cfg.log_every)
                    print(f"[{desc}] step {opt_step} loss {recent:.4f}")

                if cfg.max_steps > 0 and opt_step >= cfg.max_steps:
                    print(f"[{desc}] reached max_steps={cfg.max_steps}")
                    return losses
    return losses


def save_checkpoint(model: torch.nn.Module, path: str, extra: dict | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"Saved checkpoint -> {path}")


def load_checkpoint(model: torch.nn.Module, path: str, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["state_dict"])
    return payload


# --- config resolution for CLIs -----------------------------------------
def _repo_configs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs"


def resolve_config_path(name_or_path: str) -> str:
    """Resolve a config given a name ('tiny') or an explicit path."""
    if os.path.exists(name_or_path):
        return name_or_path
    candidate = _repo_configs_dir() / f"{name_or_path}.yaml"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"No config '{name_or_path}' (looked in {candidate})")


def load_run_config(name_or_path: str = "tiny", steps: int = 0) -> Config:
    """Load a :class:`Config`, optionally overriding ``train.max_steps``."""
    cfg = load_config(resolve_config_path(name_or_path))
    if steps and steps > 0:
        cfg.train.max_steps = steps
    return cfg


def pick_device(cfg: TrainConfig) -> str:
    return resolve_device(cfg.device)
