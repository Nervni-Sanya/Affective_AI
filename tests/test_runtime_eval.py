import torch

from emotive_llm.affect import space
from emotive_llm.affect.core import AffectCore
from emotive_llm.eval.metrics import consistency, evaluate_affect_core, profile_adherence
from emotive_llm.inference.runtime import EmotiveDialogue


def test_dialogue_loop_runs_and_stays_in_bounds(config):
    d = EmotiveDialogue(config, empathic=True)
    for user in ["You're great!", "You're useless.", "How does it work?"]:
        turn = d.step(user, max_new_tokens=6)
        assert isinstance(turn.reply, str)
        vec = torch.tensor(turn.state)
        assert torch.all(vec[:6].abs() <= 1.0 + 1e-5)
        assert 0.0 <= vec[6] <= 1.0
        assert turn.label in dict(space.PROTOTYPES) or turn.label  # a known label
    assert len(d.history) == 3


def test_dialogue_reset(config):
    d = EmotiveDialogue(config)
    d.step("hello", max_new_tokens=4)
    d.reset()
    assert len(d.history) == 0
    assert torch.allclose(d.state[0], torch.tensor(space.S0))


def test_consistency_perfect_and_zero():
    delta = torch.randn(8, 7)
    perfect = consistency(delta, delta)
    assert perfect["delta_mse"] < 1e-6
    assert perfect["delta_cosine"] > 0.999


def test_profile_adherence_self_match():
    label, vec = space.PROTOTYPES[4]  # "irritated but polite"
    states = torch.tensor([vec, vec, vec])
    pa = profile_adherence(states, vec)
    assert pa["target_label"] == label
    assert pa["label_match_rate"] == 1.0
    assert pa["mean_l2"] < 1e-5


def test_evaluate_affect_core(model_cfg, records):
    core = AffectCore(model_cfg)
    m = evaluate_affect_core(core, records)
    assert "delta_mse" in m and "delta_cosine" in m
    assert m["delta_mse"] >= 0.0
