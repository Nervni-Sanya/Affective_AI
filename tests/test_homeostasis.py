import torch

from emotive_llm.affect import space
from emotive_llm.affect.homeostasis import (
    ExtremeDwellGuard,
    homeostasis_pull,
    homeostasis_reg,
)


def test_pull_moves_toward_target():
    far = space.to_tensor((-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 0.0))
    target = space.neutral_target()
    before = torch.linalg.vector_norm(far - target)
    after = torch.linalg.vector_norm(homeostasis_pull(far, 0.5) - target)
    assert after < before


def test_pull_zero_strength_is_identity():
    s = space.to_tensor((-0.5, 0.3, 0.0, 0.2, 0.1, 0.0, 0.8))
    assert torch.allclose(homeostasis_pull(s, 0.0), space.clamp_to_bounds(s))


def test_reg_is_nonnegative_and_zero_at_target():
    assert homeostasis_reg(space.neutral_target()).item() < 1e-6
    assert homeostasis_reg(space.to_tensor((-1.0, 1.0, 0, 0, 0, 0, 0))).item() > 0


def test_dwell_guard_triggers_after_n_steps():
    guard = ExtremeDwellGuard(max_steps=2, relax_strength=0.5)
    extreme = space.to_tensor((-0.95, 0.95, 0, 0, 0, 0, 0.2))
    # Steps 1 and 2: dwell accrues, no relaxation yet.
    out1 = guard.apply(extreme.clone())
    assert guard.dwell == 1 and out1[0, 0].item() < -0.8
    guard.apply(extreme.clone())
    assert guard.dwell == 2
    # Step 3 exceeds max_steps -> relaxation, dwell resets.
    out3 = guard.apply(extreme.clone())
    assert guard.dwell == 0
    assert out3[0, 0].item() > -0.8  # valence pulled back toward neutral
