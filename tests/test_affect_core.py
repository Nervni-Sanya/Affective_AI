import torch

from emotive_llm.affect import space
from emotive_llm.affect.core import AffectCore


def test_forward_shapes_and_bounds(model_cfg):
    core = AffectCore(model_cfg)
    s0 = space.default_state(4)
    s_new, delta = core(s0, ["hello", "I am furious", "wow nice", "meh"])
    assert tuple(s_new.shape) == (4, 7)
    assert tuple(delta.shape) == (4, 7)
    res = s_new[:, space.RESILIENCE_IDX]
    assert torch.all((res >= 0.0) & (res <= 1.0))
    assert torch.all(s_new[:, :6].abs() <= 1.0 + 1e-5)


def test_alpha_initialised_to_config(model_cfg):
    core = AffectCore(model_cfg)
    alpha = core.alpha_base().detach()
    assert torch.allclose(alpha, torch.tensor(model_cfg.alpha_init), atol=1e-4)


def test_fatigue_lowers_inertia(model_cfg):
    """Low resilience -> smaller alpha_eff -> larger move toward the drive."""
    core = AffectCore(model_cfg)
    core.eval()
    text = ["You are absolutely wonderful!"]
    fresh = space.to_tensor((0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0))
    tired = space.to_tensor((0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0))
    with torch.no_grad():
        s_fresh, _ = core(fresh, text)
        s_tired, _ = core(tired, text)
    # Compare movement on the six expressive axes (resilience has its own rule).
    move_fresh = (s_fresh[0, :6] - fresh[0, :6]).abs().sum().item()
    move_tired = (s_tired[0, :6] - tired[0, :6]).abs().sum().item()
    assert move_tired >= move_fresh - 1e-6


def test_gradients_flow(model_cfg):
    core = AffectCore(model_cfg)
    s0 = space.default_state(2)
    s_new, _ = core(s0, ["a", "b"])
    loss = (s_new ** 2).mean()
    loss.backward()
    assert core.alpha_raw.grad is not None
    assert core.proj.weight.grad is not None
