"""Phase 1 affect annotation (spec §5 Phase 1).

Turns raw (user, assistant) turn pairs into ``AffectRecord`` tuples by labelling
the assistant's affect state *before* and *after* each reply.

Two backends:

* **LLM judge** — calls Claude with a strict JSON-schema structured output
  (one float per axis). Uses the official ``anthropic`` SDK; model defaults to
  ``claude-opus-4-8``.
* **Synthetic heuristic** — a deterministic, dependency-free keyword scorer used
  as a fallback so the pipeline runs with no API key.

``mode='auto'`` uses the LLM judge when the SDK and credentials are available,
otherwise falls back to the heuristic.
"""

from __future__ import annotations

import json
import random
from typing import List, Sequence, Tuple

from emotive_llm.affect import space
from emotive_llm.data.schema import AffectRecord
from emotive_llm.data.synthetic import USER_TEMPLATES, _transition

TurnPair = Tuple[str, str]

# Keyword -> user-template category, for the heuristic fallback.
_KEYWORDS = {
    "praise": ["thank", "great", "wonderful", "awesome", "amazing", "love it", "appreciate", "good job"],
    "insult": ["stupid", "useless", "idiot", "hate you", "incompetent", "terrible", "worst"],
    "sad_news": ["lost", "sick", "died", "scared", "depressed", "falling apart", "alone", "cry"],
    "threat": ["or else", "warning", "regret", "now or", "better fix", "do it now"],
    "curious": ["how", "why", "what if", "tell me more", "interesting", "explain"],
    "bored": ["whatever", "boring", "meh", "don't care", "uh huh"],
    "affection": ["care about you", "love you", "mean a lot", "glad we"],
}
_DRIVE_BY_CATEGORY = {cat: drive for cat, drive, _ in USER_TEMPLATES}


def _heuristic_drive(user_text: str) -> List[float]:
    text = (user_text or "").lower()
    for category, words in _KEYWORDS.items():
        if any(w in text for w in words):
            return _DRIVE_BY_CATEGORY[category]
    return _DRIVE_BY_CATEGORY["neutral"]


def annotate_heuristic(pairs: Sequence[TurnPair], seed: int = 42) -> List[AffectRecord]:
    """Deterministic keyword-based annotation (no external dependencies)."""
    rng = random.Random(seed)
    records: List[AffectRecord] = []
    for user_text, assistant_text in pairs:
        s_before = list(space.S0)
        drive = _heuristic_drive(user_text)
        s_after = _transition(s_before, drive, rng)
        records.append(
            AffectRecord(
                user_text=user_text,
                assistant_text=assistant_text,
                s_before=s_before,
                s_after=s_after,
                meta={"annotator": "heuristic"},
            ).validate()
        )
    return records


# --- LLM judge -----------------------------------------------------------

def _axis_doc() -> str:
    lines = []
    for i, name in enumerate(space.AXES):
        lo, hi = space.BOUNDS_LOW[i], space.BOUNDS_HIGH[i]
        lines.append(f"- {name} [{lo}, {hi}]")
    return "\n".join(lines)


_JUDGE_SYSTEM = (
    "You are an expert affect annotator for a dialogue assistant with an "
    "internal 7-dimensional emotional state (VAD + social drivers). For each "
    "exchange, estimate two states: s_before = the assistant's affect just "
    "BEFORE reading the user's latest message, and s_after = its affect AFTER "
    "reading the message — the mood that colours the reply it then writes. "
    "Infer s_after from the tone of the assistant's reply.\n\n"
    "Axes and ranges:\n" + _axis_doc() + "\n\n"
    "valence: unpleasant->pleasant; arousal: calm->activated; dominance: "
    "submissive->in-control; warmth: hostile->affectionate; interest: "
    "bored->curious; honesty: guarded->candid; resilience: depleted->fresh "
    "(0..1). Stay within the stated ranges."
)


def _axis_schema() -> dict:
    props = {name: {"type": "number"} for name in space.AXES}
    return {
        "type": "object",
        "properties": props,
        "required": list(space.AXES),
        "additionalProperties": False,
    }


def _to_vec(obj: dict) -> List[float]:
    vec = [float(obj.get(name, space.NEUTRAL_TARGET[i])) for i, name in enumerate(space.AXES)]
    # Clamp client-side (structured outputs can't enforce numeric ranges).
    return [max(space.BOUNDS_LOW[i], min(space.BOUNDS_HIGH[i], v)) for i, v in enumerate(vec)]


def annotate_llm(
    pairs: Sequence[TurnPair],
    judge_model: str = "claude-opus-4-8",
    max_pairs: int = 0,
) -> List[AffectRecord]:
    """Annotate with a Claude LLM judge using structured JSON outputs."""
    import anthropic  # lazy import; only needed for this backend

    client = anthropic.Anthropic()  # resolves credentials from the environment
    schema = {
        "type": "object",
        "properties": {"s_before": _axis_schema(), "s_after": _axis_schema()},
        "required": ["s_before", "s_after"],
        "additionalProperties": False,
    }

    items = list(pairs)
    if max_pairs:
        items = items[:max_pairs]

    records: List[AffectRecord] = []
    for user_text, assistant_text in items:
        prompt = (
            f"User said:\n{user_text}\n\n"
            f"Assistant replied:\n{assistant_text}\n\n"
            "Return s_before and s_after as JSON objects of the 7 axis values."
        )
        resp = client.messages.create(
            model=judge_model,
            max_tokens=1024,
            system=_JUDGE_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "{}")
        data = json.loads(text)
        records.append(
            AffectRecord(
                user_text=user_text,
                assistant_text=assistant_text,
                s_before=_to_vec(data["s_before"]),
                s_after=_to_vec(data["s_after"]),
                meta={"annotator": "llm", "judge_model": judge_model},
            ).validate()
        )
    return records


def _llm_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    import os

    # SDK can also resolve an `ant auth login` profile, but an explicit key is
    # the reliable signal here; treat its absence as "fall back to heuristic".
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def annotate_pairs(
    pairs: Sequence[TurnPair],
    mode: str = "auto",
    judge_model: str = "claude-opus-4-8",
    seed: int = 42,
) -> List[AffectRecord]:
    """Annotate turn pairs into ``AffectRecord``s using the selected backend."""
    if mode == "synthetic":
        return annotate_heuristic(pairs, seed=seed)
    if mode == "llm":
        return annotate_llm(pairs, judge_model=judge_model)
    if mode == "auto":
        if _llm_available():
            try:
                return annotate_llm(pairs, judge_model=judge_model)
            except Exception as e:  # noqa: BLE001 - robust fallback for the pipeline
                print(f"[annotate] LLM judge failed ({e}); falling back to heuristic.")
        return annotate_heuristic(pairs, seed=seed)
    raise ValueError(f"Unknown annotate mode: {mode}")
