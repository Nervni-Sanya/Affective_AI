"""Phase 2: pre-train the Affect Core to predict ΔS (spec §5 Phase 2).

Minimises MSE between the predicted next state and the annotated ``s_after``,
optionally plus a small homeostasis regulariser. This teaches the recurrent
core realistic emotion-transition dynamics before any generation is involved.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from emotive_llm.affect.core import AffectCore
from emotive_llm.affect.homeostasis import homeostasis_reg
from emotive_llm.data.dataset import AffectCoreDataset, affect_collate, build_records
from emotive_llm.data.schema import read_jsonl
from emotive_llm.training.loop import (
    load_run_config,
    pick_device,
    save_checkpoint,
    set_seed,
    train_loop,
)


def make_loss_fn(reg_weight: float):
    def loss_fn(model: AffectCore, batch, device):
        s_before = batch["s_before"].to(device)
        s_after_true = batch["s_after"].to(device)
        s_pred, _delta = model(s_before, batch["texts"])
        mse = F.mse_loss(s_pred, s_after_true)
        if reg_weight > 0:
            return mse + reg_weight * homeostasis_reg(s_pred)
        return mse

    return loss_fn


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2: pre-train the Affect Core.")
    ap.add_argument("--config", default="tiny", help="config name or path")
    ap.add_argument("--steps", type=int, default=0, help="override train.max_steps")
    ap.add_argument("--data", default="", help="optional JSONL of AffectRecords")
    ap.add_argument("--out", default="", help="checkpoint path (default <out_dir>/affect_core.pt)")
    args = ap.parse_args()

    cfg = load_run_config(args.config, steps=args.steps)
    set_seed(cfg.train.seed)
    device = pick_device(cfg.train)
    print(f"[phase2] config={cfg.name} device={device}")

    records = read_jsonl(args.data) if args.data else build_records(cfg.data)
    print(f"[phase2] {len(records)} records")

    ds = AffectCoreDataset(records)
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=affect_collate)

    model = AffectCore(cfg.model)
    loss_fn = make_loss_fn(cfg.train.homeostasis_reg)
    losses = train_loop(model, dl, loss_fn, cfg.train, device, desc="phase2")

    out = args.out or os.path.join(cfg.train.out_dir, "affect_core.pt")
    save_checkpoint(model, out, extra={"config_name": cfg.name})
    if losses:
        print(f"[phase2] first/last loss: {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
