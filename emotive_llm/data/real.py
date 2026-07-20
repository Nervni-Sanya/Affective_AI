"""Real corpus loader: DailyDialog -> raw (user, assistant) turn pairs.

DailyDialog is a multi-turn open-domain dialogue dataset. We flatten each
conversation into consecutive (user, assistant) pairs, ready to be annotated
with affect states in Phase 1. Network access is required to download it; tiny
tests use the synthetic generator instead.
"""

from __future__ import annotations

from typing import List, Tuple

TurnPair = Tuple[str, str]


def load_dailydialog_pairs(max_dialogues: int = 1000, split: str = "train") -> List[TurnPair]:
    """Load DailyDialog and return (user, assistant) consecutive turn pairs.

    Raises a clear error if the ``datasets`` library is missing or the download
    fails, so callers can fall back to synthetic data.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The 'datasets' package is required for DailyDialog. "
            "Install with: pip install datasets"
        ) from e

    # datasets>=3 removed script-based loading, so use the parquet mirror of
    # DailyDialog; fall back to the legacy script id on older versions.
    try:
        ds = load_dataset("li2017dailydialog/daily_dialog", split=split)
    except Exception:  # noqa: BLE001 - older datasets or renamed repo
        ds = load_dataset("daily_dialog", split=split, trust_remote_code=True)

    pairs: List[TurnPair] = []
    for i, ex in enumerate(ds):
        if i >= max_dialogues:
            break
        turns = [t.strip() for t in ex["dialog"] if t and t.strip()]
        for u, a in zip(turns[::2], turns[1::2]):
            pairs.append((u, a))
    return pairs


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Inspect DailyDialog turn pairs.")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--split", type=str, default="train")
    args = ap.parse_args()

    try:
        pairs = load_dailydialog_pairs(max_dialogues=args.limit, split=args.split)
    except Exception as e:  # noqa: BLE001 - surface the reason to the user
        print(f"Could not load DailyDialog: {e}")
        return
    for u, a in pairs[: args.limit]:
        print(f"USER: {u}\nASSISTANT: {a}\n---")


if __name__ == "__main__":
    _cli()
