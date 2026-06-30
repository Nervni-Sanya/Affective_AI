#!/usr/bin/env python3
"""Scripted demo: drive EmotiveLLM through a few turns and print the affect
trajectory (with prototype labels) alongside the generated reply.

    python scripts/demo_chat.py --config tiny

With the tiny (random-init) models the generated text is gibberish — the point
is to show the *state dynamics* and the end-to-end loop running on CPU. Load
trained checkpoints (or use --config base with --pretrained weights) for fluent
replies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from the repo without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotive_llm.affect import space  # noqa: E402
from emotive_llm.inference.runtime import EmotiveDialogue  # noqa: E402
from emotive_llm.training.loop import load_run_config  # noqa: E402

SCRIPT = [
    "Hi! You did a wonderful job earlier, thank you so much.",
    "Wait, how does the affect core actually work?",
    "Ugh, you're being useless and slow. Fix it now or else.",
    "...sorry. I'm just having a really hard day.",
    "Anyway. whatever. tell me the time.",
]


def _fmt_state(vec):
    return "  ".join(f"{name[:4]}={v:+.2f}" for name, v in zip(space.AXES, vec))


def main() -> None:
    ap = argparse.ArgumentParser(description="EmotiveLLM scripted demo.")
    ap.add_argument("--config", default="tiny")
    ap.add_argument("--affect-core", default="")
    ap.add_argument("--sft", default="")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--no-empathic", action="store_true")
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    dialogue = EmotiveDialogue.from_checkpoints(
        cfg,
        affect_core_path=args.affect_core or None,
        sft_path=args.sft or None,
        empathic=not args.no_empathic,
    )

    print(f"=== EmotiveLLM demo (config={cfg.name}, empathic={not args.no_empathic}) ===")
    print(f"S_0  [{dialogue.label()}]  {_fmt_state(dialogue.state_vector())}\n")

    for user in SCRIPT:
        turn = dialogue.step(user, max_new_tokens=args.max_new_tokens)
        print(f"USER : {user}")
        print(f"BOT  : {turn.reply!r}")
        print(f"STATE: [{turn.label}]  {_fmt_state(turn.state)}\n")


if __name__ == "__main__":
    main()
