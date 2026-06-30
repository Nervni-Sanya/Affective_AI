"""Evaluation metrics (spec §7).

* :func:`consistency` — how close the predicted next-step affect change is to the
  annotated reaction (MSE + cosine over ΔS).
* :func:`profile_adherence` — an automatable proxy for the human "does this match
  the stated affective profile" Likert check: distance to a target state and the
  rate at which produced states share the target's prototype label.
* :func:`evaluate_affect_core` — run a trained Affect Core over held-out records
  and report consistency.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from emotive_llm.affect import space


def consistency(pred_delta: torch.Tensor, true_delta: torch.Tensor) -> Dict[str, float]:
    """MSE and mean cosine similarity between predicted and annotated ΔS."""
    pred_delta = pred_delta.float()
    true_delta = true_delta.float()
    mse = torch.mean((pred_delta - true_delta) ** 2).item()
    cos = torch.nn.functional.cosine_similarity(pred_delta, true_delta, dim=-1, eps=1e-8)
    return {"delta_mse": float(mse), "delta_cosine": float(cos.mean().item())}


def profile_adherence(
    states: torch.Tensor, target: Sequence[float]
) -> Dict[str, float]:
    """How well a sequence of states matches a target affect profile.

    Returns mean L2 distance to ``target`` and the fraction of states whose
    nearest prototype equals the target's prototype.
    """
    states = states.float().reshape(-1, space.DIM)
    target_t = torch.tensor(target, dtype=torch.float32).reshape(space.DIM)
    l2 = torch.linalg.vector_norm(states - target_t, dim=1).mean().item()

    target_label = space.nearest_prototype(target_t)
    matches = sum(1 for s in states if space.nearest_prototype(s) == target_label)
    return {
        "mean_l2": float(l2),
        "label_match_rate": matches / max(1, states.shape[0]),
        "target_label": target_label,
    }


@torch.no_grad()
def evaluate_affect_core(model, records, batch_size: int = 16) -> Dict[str, float]:
    """Run the Affect Core over records and report ΔS consistency."""
    from torch.utils.data import DataLoader

    from emotive_llm.data.dataset import AffectCoreDataset, affect_collate

    model.eval()
    device = next(model.parameters()).device
    dl = DataLoader(
        AffectCoreDataset(records), batch_size=batch_size, collate_fn=affect_collate
    )
    pred_deltas: List[torch.Tensor] = []
    true_deltas: List[torch.Tensor] = []
    for batch in dl:
        s_before = batch["s_before"].to(device)
        s_after = batch["s_after"].to(device)
        s_pred, delta = model(s_before, batch["texts"])
        pred_deltas.append(delta.cpu())
        true_deltas.append((s_after - s_before).cpu())
    return consistency(torch.cat(pred_deltas), torch.cat(true_deltas))
