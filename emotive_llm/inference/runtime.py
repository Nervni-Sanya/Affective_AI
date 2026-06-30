"""The dialogue runtime: state management + conditioned generation (spec §6).

``EmotiveDialogue`` wires the Affect Core and the conditioned backbone into the
inference loop:

1. start from the neutral state ``S_0``;
2. on each user turn, the Affect Core updates the state ``S_t``;
3. ``S_t`` conditions the backbone, which generates the reply;
4. (optional) the reply is fed back into the Affect Core — the empathic loop,
   where the model "hears itself";
5. a homeostasis pull + extreme-dwell guard keep the state from getting stuck
   in an extreme (spec §7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from emotive_llm.affect import space
from emotive_llm.affect.core import AffectCore
from emotive_llm.affect.homeostasis import ExtremeDwellGuard, homeostasis_pull
from emotive_llm.backbone.conditioned_lm import ConditionedLM
from emotive_llm.config import Config, resolve_device


@dataclass
class DialogueTurn:
    user_text: str
    reply: str
    state: List[float]
    label: str


@dataclass
class EmotiveDialogue:
    config: Config
    affect_core: Optional[AffectCore] = None
    lm: Optional[ConditionedLM] = None
    empathic: bool = True
    homeostasis_strength: float = 0.05
    device: Optional[str] = None
    dwell_max_steps: int = 3

    history: List[DialogueTurn] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.device = self.device or resolve_device(self.config.train.device)
        if self.affect_core is None:
            self.affect_core = AffectCore(self.config.model)
        if self.lm is None:
            self.lm = ConditionedLM(self.config.model)
        self.affect_core.to(self.device).eval()
        self.lm.to(self.device).eval()
        self._guard = ExtremeDwellGuard(max_steps=self.dwell_max_steps)
        self.state = space.default_state(1, device=self.device)

    # -- state helpers -----------------------------------------------------
    def reset(self) -> None:
        self.state = space.default_state(1, device=self.device)
        self._guard.reset()
        self.history.clear()

    def state_vector(self) -> List[float]:
        return [round(float(x), 4) for x in self.state[0].tolist()]

    def label(self) -> str:
        return space.nearest_prototype(self.state[0])

    # -- the loop ----------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        user_text: str,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> DialogueTurn:
        # 1-2: Affect Core reacts to the user message.
        self.state = self.affect_core.step(self.state, user_text)

        # 3: condition the backbone on S_t and generate.
        reply = self.lm.generate_text(
            user_text,
            self.state,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
        )

        # 4: optional empathic feedback — the model hears its own reply.
        if self.empathic and reply.strip():
            self.state = self.affect_core.step(self.state, reply)

        # 5: homeostasis pull + extreme-dwell safety guard.
        self.state = homeostasis_pull(self.state, strength=self.homeostasis_strength)
        self.state = self._guard.apply(self.state)

        turn = DialogueTurn(
            user_text=user_text,
            reply=reply,
            state=self.state_vector(),
            label=self.label(),
        )
        self.history.append(turn)
        return turn

    # -- checkpoint loading ------------------------------------------------
    @classmethod
    def from_checkpoints(
        cls,
        config: Config,
        affect_core_path: Optional[str] = None,
        sft_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> "EmotiveDialogue":
        dev = device or resolve_device(config.train.device)
        affect_core = AffectCore(config.model)
        lm = ConditionedLM(config.model)
        if affect_core_path:
            affect_core.load_state_dict(torch.load(affect_core_path, map_location=dev)["state_dict"])
        if sft_path:
            lm.load_state_dict(torch.load(sft_path, map_location=dev)["state_dict"])
        return cls(config=config, affect_core=affect_core, lm=lm, device=dev, **kwargs)


def _cli() -> None:
    import argparse

    from emotive_llm.training.loop import load_run_config

    ap = argparse.ArgumentParser(description="Chat with EmotiveLLM (prints affect state).")
    ap.add_argument("--config", default="tiny")
    ap.add_argument("--affect-core", default="")
    ap.add_argument("--sft", default="")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--no-empathic", action="store_true")
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    dialogue = EmotiveDialogue.from_checkpoints(
        cfg,
        affect_core_path=args.affect_core or None,
        sft_path=args.sft or None,
        empathic=not args.no_empathic,
    )
    print("EmotiveLLM chat. Ctrl-D / empty line to exit.\n")
    try:
        while True:
            user = input("you> ").strip()
            if not user:
                break
            turn = dialogue.step(user, max_new_tokens=args.max_new_tokens)
            print(f"bot> {turn.reply}")
            print(f"     [{turn.label}] {turn.state}\n")
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    _cli()
