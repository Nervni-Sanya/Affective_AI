#!/usr/bin/env python3
"""End-to-end driver: Phase 1 (annotate) -> Phase 2 (Affect Core) -> Phase 3 (SFT).

    python scripts/run_pipeline.py --config tiny --steps 10

Runs the whole reference pipeline on CPU with tiny models and synthetic data,
writing the annotated dataset and both checkpoints, then runs a short demo
dialogue from the trained checkpoints.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from emotive_llm.affect.core import AffectCore  # noqa: E402
from emotive_llm.backbone.conditioned_lm import ConditionedLM  # noqa: E402
from emotive_llm.data.dataset import (  # noqa: E402
    AffectCoreDataset,
    SFTDataset,
    affect_collate,
    build_records,
    make_sft_collate,
)
from emotive_llm.data.schema import write_jsonl  # noqa: E402
from emotive_llm.eval.metrics import evaluate_affect_core  # noqa: E402
from emotive_llm.inference.runtime import EmotiveDialogue  # noqa: E402
from emotive_llm.training.loop import (  # noqa: E402
    load_run_config,
    pick_device,
    save_checkpoint,
    set_seed,
    train_loop,
)
from emotive_llm.training.train_affect_core import make_loss_fn  # noqa: E402
from emotive_llm.training.train_sft import sft_loss_fn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full EmotiveLLM pipeline.")
    ap.add_argument("--config", default="tiny")
    ap.add_argument("--steps", type=int, default=0, help="override train.max_steps")
    args = ap.parse_args()

    cfg = load_run_config(args.config, steps=args.steps)
    set_seed(cfg.train.seed)
    device = pick_device(cfg.train)
    out_dir = cfg.train.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== EmotiveLLM pipeline (config={cfg.name}, device={device}) ===")

    # --- Phase 1: data + annotation -------------------------------------
    print("\n[Phase 1] building/annotating records ...")
    records = build_records(cfg.data)
    data_path = os.path.join(cfg.data.data_dir, "records.jsonl")
    write_jsonl(data_path, records)
    print(f"  {len(records)} records -> {data_path}")
    print(f"  sample: {records[0].user_text!r} -> {records[0].assistant_text!r}")

    # --- Phase 2: Affect Core (ΔS, MSE) ---------------------------------
    print("\n[Phase 2] pre-training Affect Core ...")
    core = AffectCore(cfg.model)
    dl2 = DataLoader(
        AffectCoreDataset(records), batch_size=cfg.train.batch_size,
        shuffle=True, collate_fn=affect_collate,
    )
    losses2 = train_loop(core, dl2, make_loss_fn(cfg.train.homeostasis_reg), cfg.train, device, "phase2")
    save_checkpoint(core, os.path.join(out_dir, "affect_core.pt"), {"config_name": cfg.name})
    metrics = evaluate_affect_core(core, records[: min(64, len(records))])
    print(f"  consistency: { {k: round(v,4) for k,v in metrics.items()} }")
    if losses2:
        print(f"  loss: {losses2[0]:.4f} -> {losses2[-1]:.4f}")

    # --- Phase 3: SFT (conditioned cross-entropy) -----------------------
    print("\n[Phase 3] SFT of conditioned backbone ...")
    lm = ConditionedLM(cfg.model)
    dl3 = DataLoader(
        SFTDataset(records, lm.tokenizer, max_seq_len=cfg.train.max_seq_len),
        batch_size=cfg.train.batch_size, shuffle=True,
        collate_fn=make_sft_collate(lm.tokenizer.pad_token_id),
    )
    losses3 = train_loop(lm, dl3, sft_loss_fn, cfg.train, device, "phase3")
    save_checkpoint(lm, os.path.join(out_dir, "sft.pt"), {"config_name": cfg.name})
    if losses3:
        print(f"  loss: {losses3[0]:.4f} -> {losses3[-1]:.4f}")

    # --- Demo from the trained checkpoints ------------------------------
    print("\n[Demo] sampling a short dialogue from trained checkpoints ...")
    dialogue = EmotiveDialogue(cfg, affect_core=core, lm=lm, device=device)
    for user in ["You're amazing, thank you!", "You're useless. Fix it now.", "How does this work?"]:
        turn = dialogue.step(user, max_new_tokens=12)
        print(f"  USER: {user}\n    -> [{turn.label}] valence={turn.state[0]:+.2f} arousal={turn.state[1]:+.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
