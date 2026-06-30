import torch

from emotive_llm.affect import space
from emotive_llm.backbone.conditioned_lm import ConditionedLM
from emotive_llm.conditioning.projector import AffectProjector


def test_projector_output_shape():
    proj = AffectProjector(d_model=32, hidden=64, n_prefix=3)
    out = proj(space.default_state(5))
    assert tuple(out.shape) == (5, 3, 32)


def test_projector_accepts_1d_state():
    proj = AffectProjector(d_model=16, hidden=32, n_prefix=1)
    out = proj(torch.tensor(space.S0))
    assert tuple(out.shape) == (1, 1, 16)


def test_backbone_forward_loss_and_grad(model_cfg):
    lm = ConditionedLM(model_cfg)
    tok = lm.tokenizer
    prompt = tok.encode("hello there\n", add_special_tokens=False)
    reply = tok.encode("hi friend", add_special_tokens=False) + [tok.eos_token_id]
    input_ids = torch.tensor([prompt + reply])
    labels = torch.tensor([[-100] * len(prompt) + reply])
    out = lm(input_ids=input_ids, affect_state=space.default_state(1), labels=labels)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    # Conditioning path must be differentiable.
    assert lm.projector.net[0].weight.grad is not None


def test_backbone_generates_tokens(model_cfg):
    lm = ConditionedLM(model_cfg)
    txt = lm.generate_text("hello", space.default_state(1), max_new_tokens=8)
    assert isinstance(txt, str)
    gen = lm.generate(
        torch.tensor([[1, 2, 3]]), space.default_state(1), max_new_tokens=6, do_sample=True, top_k=5
    )
    assert gen.shape[0] == 1 and gen.shape[1] >= 1
