import torch

from emotive_llm.affect import space
from emotive_llm.data.annotate import annotate_heuristic
from emotive_llm.data.dataset import (
    AffectCoreDataset,
    SFTDataset,
    affect_collate,
    make_sft_collate,
)
from emotive_llm.data.schema import AffectRecord, read_jsonl, write_jsonl
from emotive_llm.data.synthetic import generate_records


def test_synthetic_records_valid(records):
    assert len(records) == 24
    for r in records:
        r.validate()  # raises if out of bounds
        assert "label" in r.meta


def test_synthetic_is_deterministic():
    a = generate_records(10, seed=5)
    b = generate_records(10, seed=5)
    assert [r.to_dict() for r in a] == [r.to_dict() for r in b]


def test_jsonl_roundtrip(tmp_path, records):
    p = tmp_path / "r.jsonl"
    n = write_jsonl(str(p), records)
    assert n == len(records)
    back = read_jsonl(str(p))
    assert back[0].to_dict() == records[0].to_dict()


def test_record_validate_rejects_out_of_bounds():
    bad = AffectRecord("u", "a", [2.0, 0, 0, 0, 0, 0, 0.5], list(space.S0))
    try:
        bad.validate()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_heuristic_annotation_polarity():
    recs = annotate_heuristic([
        ("Thank you, that was wonderful!", "yw"),
        ("You are useless and stupid", "..."),
    ])
    assert recs[0].s_after[0] > 0  # praise -> positive valence
    assert recs[1].s_after[0] < 0  # insult -> negative valence
    assert all(r.meta["annotator"] == "heuristic" for r in recs)


def test_affect_collate(records):
    ds = AffectCoreDataset(records)
    batch = affect_collate([ds[i] for i in range(4)])
    assert len(batch["texts"]) == 4
    assert tuple(batch["s_before"].shape) == (4, 7)


def test_sft_collate_and_label_masking(model_cfg, records):
    from emotive_llm.tokenization import SimpleByteTokenizer

    tok = SimpleByteTokenizer()
    ds = SFTDataset(records, tok, max_seq_len=48)
    collate = make_sft_collate(tok.pad_token_id)
    batch = collate([ds[i] for i in range(4)])
    assert tuple(batch["affect_state"].shape) == (4, 7)
    assert batch["input_ids"].shape == batch["labels"].shape
    # Some positions are masked (-100) and some are real reply tokens.
    assert (batch["labels"] == -100).any()
    assert (batch["labels"] != -100).any()
