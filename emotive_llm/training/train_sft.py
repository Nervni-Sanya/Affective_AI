"""Phase 3: SFT of the affect-conditioned backbone (spec §5 Phase 3).

Standard cross-entropy on the reply tokens only. The affect state enters purely
as input conditioning (a soft prompt) — **no** loss coefficient is multiplied by
emotion (spec §8). Correct behaviour at different states is taught by the
training examples themselves (state -> reply style).
"""

from __future__ import annotations

import argparse
import os

from torch.utils.data import DataLoader

from emotive_llm.backbone.conditioned_lm import ConditionedLM
from emotive_llm.data.dataset import SFTDataset, build_records, make_sft_collate
from emotive_llm.data.schema import read_jsonl
from emotive_llm.training.loop import (
    load_run_config,
    pick_device,
    save_checkpoint,
    set_seed,
    train_loop,
)


def sft_loss_fn(model: ConditionedLM, batch, device):
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        affect_state=batch["affect_state"].to(device),
        labels=batch["labels"].to(device),
    )
    return out.loss


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3: SFT the conditioned backbone.")
    ap.add_argument("--config", default="tiny", help="config name or path")
    ap.add_argument("--steps", type=int, default=0, help="override train.max_steps")
    ap.add_argument("--data", default="", help="optional JSONL of AffectRecords")
    ap.add_argument("--out", default="", help="checkpoint path (default <out_dir>/sft.pt)")
    args = ap.parse_args()

    cfg = load_run_config(args.config, steps=args.steps)
    set_seed(cfg.train.seed)
    device = pick_device(cfg.train)
    print(f"[phase3] config={cfg.name} device={device}")

    model = ConditionedLM(cfg.model)

    records = read_jsonl(args.data) if args.data else build_records(cfg.data)
    print(f"[phase3] {len(records)} records")

    ds = SFTDataset(records, model.tokenizer, max_seq_len=cfg.train.max_seq_len)
    dl = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=make_sft_collate(model.tokenizer.pad_token_id),
    )

    losses = train_loop(model, dl, sft_loss_fn, cfg.train, device, desc="phase3")

    out = args.out or os.path.join(cfg.train.out_dir, "sft.pt")
    save_checkpoint(model, out, extra={"config_name": cfg.name})
    if losses:
        print(f"[phase3] first/last loss: {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
