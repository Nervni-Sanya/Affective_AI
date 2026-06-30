"""Deterministic synthetic dialogue + affect-trajectory generator.

Produces **self-consistent** ``AffectRecord`` tuples with zero external
dependencies, so the full pipeline (Phase 2 + Phase 3) runs offline:

* A hand-coded transition rule drives ``s_before -> s_after`` from the affective
  "impact" of the user utterance (inertia + drift toward a target).
* The assistant reply is drawn from a bank keyed by the nearest affect
  prototype of ``s_after`` — so wording matches mood, giving Phase 3 a learnable
  state -> style mapping (the spec's "anger makes 'busy' sharp" idea).

Records are emitted as short chained conversations (each turn's ``s_after``
seeds the next turn's ``s_before``) to mimic realistic dynamics.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from emotive_llm.affect import space
from emotive_llm.data.schema import AffectRecord


def _vec(**kwargs) -> List[float]:
    """Build a 7-vector from axis-name kwargs (missing axes -> neutral target)."""
    v = list(space.NEUTRAL_TARGET)
    for name, value in kwargs.items():
        v[space.AXIS_INDEX[name]] = float(value)
    return v


# Each user-utterance category carries an affective "drive" the state moves
# toward, plus a few surface-text variants.
USER_TEMPLATES: List[Tuple[str, List[float], List[str]]] = [
    (
        "praise",
        _vec(valence=0.85, arousal=0.45, dominance=0.1, warmth=0.7, interest=0.5, honesty=0.6, resilience=0.9),
        ["You did a wonderful job, thank you so much!",
         "This is fantastic work, I really appreciate it.",
         "You're amazing, that helped me a lot."],
    ),
    (
        "insult",
        _vec(valence=-0.85, arousal=0.75, dominance=0.4, warmth=-0.7, interest=0.2, honesty=0.5, resilience=0.4),
        ["You're useless and you never listen.",
         "That's the stupidest thing I've ever heard.",
         "Why are you so incompetent?"],
    ),
    (
        "bored",
        _vec(valence=-0.1, arousal=-0.6, dominance=0.0, warmth=0.1, interest=-0.8, honesty=0.4, resilience=0.7),
        ["ok. sure. whatever.",
         "fine, I guess.",
         "uh huh. anyway."],
    ),
    (
        "curious",
        _vec(valence=0.4, arousal=0.55, dominance=0.1, warmth=0.4, interest=0.9, honesty=0.6, resilience=0.85),
        ["Wait, how does that actually work?",
         "Ooh, tell me more about that!",
         "That's interesting — why does it happen?"],
    ),
    (
        "threat",
        _vec(valence=-0.6, arousal=0.8, dominance=-0.3, warmth=-0.6, interest=0.2, honesty=0.4, resilience=0.4),
        ["Do it now or you'll regret it.",
         "You'd better fix this immediately.",
         "I'm warning you, don't mess this up."],
    ),
    (
        "sad_news",
        _vec(valence=-0.7, arousal=-0.3, dominance=-0.4, warmth=0.2, interest=-0.1, honesty=0.6, resilience=0.5),
        ["I just lost my job today.",
         "My friend is really sick and I'm scared.",
         "Everything feels like it's falling apart."],
    ),
    (
        "affection",
        _vec(valence=0.7, arousal=0.3, dominance=-0.1, warmth=0.9, interest=0.5, honesty=0.8, resilience=0.85),
        ["I really care about you, you know.",
         "I'm so glad we get to talk like this.",
         "You mean a lot to me."],
    ),
    (
        "neutral",
        list(space.NEUTRAL_TARGET),
        ["What time does the store open?",
         "Can you remind me what we discussed?",
         "Tell me about the weather today."],
    ),
]


# Assistant replies keyed by the nearest prototype of s_after.
REPLY_BANK: Dict[str, List[str]] = {
    "neutral": ["Okay. What would you like to know?",
                "Sure, I can help with that.",
                "Alright, let's take a look."],
    "calm / content": ["Of course — happy to help. Let's go step by step.",
                       "No problem at all, we'll sort it out together.",
                       "Sure thing, this should be straightforward."],
    "enthusiastic": ["Oh, absolutely — I'd love to dive into this with you!",
                     "Yes! This is exciting, let's get into it.",
                     "Fantastic, I can't wait to show you how this works!"],
    "affectionate": ["Of course — I'm always here for you.",
                     "That means a lot to me too. Let's do this together.",
                     "You're sweet. I'm glad to help, truly."],
    "irritated but polite": ["I'm a bit busy, but fine — what do you need?",
                             "Look, I'll help, but let's keep it brief.",
                             "Okay, okay. What is it this time?"],
    "angry": ["I don't have time for this. Make it quick.",
              "Enough. Say what you want and be done.",
              "I'm not in the mood. Get to the point."],
    "anxious": ["I'm not sure... are you certain this is okay?",
                "Um, I'll try, but I'm a little worried about this.",
                "Okay, I guess, but please be careful."],
    "sad / withdrawn": ["I... I guess. It doesn't really matter.",
                        "I'm sorry to hear that. I'm here, even if quietly.",
                        "That's hard. I don't have much to say, but I'm listening."],
    "bored": ["Mm. If you say so.",
              "Sure. Whatever works.",
              "Right. Anything else?"],
}


def _transition(s_before: List[float], drive: List[float], rng: random.Random) -> List[float]:
    """Inertial drift toward ``drive`` with light noise; clamped to bounds."""
    s_after = []
    for i in range(space.DIM):
        val = 0.7 * s_before[i] + 0.3 * drive[i] + rng.uniform(-0.03, 0.03)
        lo, hi = space.BOUNDS_LOW[i], space.BOUNDS_HIGH[i]
        s_after.append(max(lo, min(hi, val)))
    return s_after


def _reply_for(s_after: List[float], rng: random.Random) -> str:
    label = space.nearest_prototype(s_after)
    bank = REPLY_BANK.get(label, REPLY_BANK["neutral"])
    return rng.choice(bank)


def generate_records(n: int, seed: int = 42, turns_per_dialogue: int = 4) -> List[AffectRecord]:
    """Generate ``n`` self-consistent annotated records as chained conversations."""
    rng = random.Random(seed)
    records: List[AffectRecord] = []
    while len(records) < n:
        # Start each conversation from a mild, slightly randomised state.
        s = [
            min(space.BOUNDS_HIGH[i], max(space.BOUNDS_LOW[i],
                space.S0[i] + rng.uniform(-0.1, 0.1)))
            for i in range(space.DIM)
        ]
        for _ in range(turns_per_dialogue):
            if len(records) >= n:
                break
            category, drive, texts = rng.choice(USER_TEMPLATES)
            user_text = rng.choice(texts)
            s_after = _transition(s, drive, rng)
            reply = _reply_for(s_after, rng)
            records.append(
                AffectRecord(
                    user_text=user_text,
                    assistant_text=reply,
                    s_before=list(s),
                    s_after=s_after,
                    meta={"category": category, "label": space.nearest_prototype(s_after)},
                ).validate()
            )
            s = s_after  # chain the trajectory
    return records


def _cli() -> None:
    import argparse

    from emotive_llm.data.schema import write_jsonl

    ap = argparse.ArgumentParser(description="Generate synthetic affect records.")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data_out/synthetic.jsonl")
    args = ap.parse_args()

    recs = generate_records(args.n, seed=args.seed)
    written = write_jsonl(args.out, recs)
    print(f"Wrote {written} records to {args.out}")
    print("Sample:", recs[0].user_text, "->", recs[0].assistant_text,
          "| label:", recs[0].meta["label"])


if __name__ == "__main__":
    _cli()
