"""The annotated-dialogue record and JSONL (de)serialisation (spec §5 Phase 1).

A record is the tuple the specification calls for: the user utterance, the
assistant reply, and the affect state *before* and *after* the exchange.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Iterable, List

from emotive_llm.affect import space


@dataclass
class AffectRecord:
    user_text: str
    assistant_text: str
    s_before: List[float]
    s_after: List[float]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.s_before = [float(x) for x in self.s_before]
        self.s_after = [float(x) for x in self.s_after]

    def validate(self) -> "AffectRecord":
        """Assert both vectors have the right length and lie within bounds."""
        for name, vec in (("s_before", self.s_before), ("s_after", self.s_after)):
            if len(vec) != space.DIM:
                raise ValueError(f"{name} must have {space.DIM} dims, got {len(vec)}")
            for i, v in enumerate(vec):
                lo, hi = space.BOUNDS_LOW[i], space.BOUNDS_HIGH[i]
                if not (lo - 1e-6 <= v <= hi + 1e-6):
                    raise ValueError(
                        f"{name}[{space.AXES[i]}]={v} out of bounds [{lo}, {hi}]"
                    )
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AffectRecord":
        return cls(
            user_text=d["user_text"],
            assistant_text=d["assistant_text"],
            s_before=d["s_before"],
            s_after=d["s_after"],
            meta=d.get("meta", {}),
        )


def write_jsonl(path: str, records: Iterable[AffectRecord]) -> int:
    """Write records as JSON Lines; returns the number written."""
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> List[AffectRecord]:
    """Read records from a JSON Lines file."""
    out: List[AffectRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(AffectRecord.from_dict(json.loads(line)))
    return out
