import torch

from emotive_llm.affect import space


def test_dim_and_axes():
    assert space.DIM == 7
    assert space.AXES[0] == "valence"
    assert space.AXES[-1] == "resilience"


def test_default_state_shape_and_value():
    s = space.default_state(3)
    assert tuple(s.shape) == (3, 7)
    assert torch.allclose(s[0], torch.tensor(space.S0))


def test_bounded_activation_in_bounds():
    raw = torch.randn(16, 7) * 8.0
    out = space.bounded_activation(raw)
    assert torch.all(out[:, :6].abs() <= 1.0 + 1e-5)
    res = out[:, space.RESILIENCE_IDX]
    assert torch.all((res >= 0.0) & (res <= 1.0))


def test_clamp_to_bounds():
    s = torch.tensor([[5.0, -5.0, 0.0, 0.0, 0.0, 0.0, 2.0]])
    c = space.clamp_to_bounds(s)
    assert c[0, 0].item() == 1.0
    assert c[0, 1].item() == -1.0
    assert c[0, 6].item() == 1.0


def test_is_extreme():
    angry = space.to_tensor((-0.9, 0.9, 0, 0, 0, 0, 1.0))
    calm = space.to_tensor((0.4, -0.2, 0, 0, 0, 0, 1.0))
    assert bool(space.is_extreme(angry).item()) is True
    assert bool(space.is_extreme(calm).item()) is False


def test_nearest_prototype_recovers_labels():
    for label, vec in space.PROTOTYPES:
        assert space.nearest_prototype(vec) == label
