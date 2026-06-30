# EmotiveLLM — Design

This document maps the implementation to the project specification (the ТЗ):
a dialogue LLM with an internal multi-dimensional emotional state that evolves
with context and **conditions** generation rather than scaling the loss.

## 1. Affect Space (spec §3) — `emotive_llm/affect/space.py`

A 7-dimensional, interpretable state (VAD + social drivers + a service axis):

| idx | axis | range | meaning |
|----|------|-------|---------|
| 0 | valence | [-1, 1] | unpleasant → pleasant |
| 1 | arousal | [-1, 1] | calm → activated |
| 2 | dominance | [-1, 1] | submissive → in control |
| 3 | warmth | [-1, 1] | hostile → affectionate |
| 4 | interest | [-1, 1] | bored → curious |
| 5 | honesty | [-1, 1] | guarded → candid |
| 6 | resilience | [0, 1] | depleted → fresh (service: inertia/fatigue) |

The first six axes are **bipolar** (per §8, real emotions are bipolar, so clear
poles help annotators and generalisation). `resilience` is unipolar and drives
the fatigue mechanism. Start state `S_0 = [0,0,0,0.5,0.5,0.5,1.0]` (§6). Bounds
are enforced by construction via `bounded_activation` (tanh on axes 0–5, sigmoid
on resilience). `nearest_prototype` maps a state to a human label (e.g.
"irritated but polite") for metrics and the demo.

## 2. Affect Core (spec §4.1) — `emotive_llm/affect/core.py`

A recurrent decay cell over the state:

```
S_t = bounded( α_eff ⊙ S_{t-1} + (1 - α_eff) ⊙ (W · h_t + b) )
```

* `h_t` — mean-pooled embedding of the latest utterance from a light encoder
  (DistilBERT; tiny random-init in offline mode).
* `α` — learnable per-axis inertia, initialised to `(0.9, 0.6, …)` (Valence
  sticky, Arousal cools fast), kept in `(0,1)` via a sigmoid parameterisation.
* **Fatigue** (§4.1): `α_eff = α_base · (β + (1-β)·resilience)`. Low resilience
  lowers effective inertia, making the state more volatile.
* `resilience` has its own hand-designed dynamics: spent by exertion (high
  `|arousal|`, negative valence), recovered when calm.

Phase 2 trains this module to predict `ΔS` (MSE).

## 3. Conditioning + Backbone (spec §4.2)

`emotive_llm/conditioning/projector.py` — `S_t (7) → MLP(7→128→n_prefix·d_model)`
→ prefix embeddings (soft prompt).

`emotive_llm/backbone/conditioned_lm.py` — wraps a GPT-like causal LM (GPT-2;
tiny random-init offline). The prefix embeddings are **prepended** to the token
embeddings (`inputs_embeds`); the attention mask is extended and the prefix
positions are masked out of the loss (`-100`). Generation uses a self-contained
decode loop, robust across transformers versions and CPU.

## 4. Training (spec §5)

* **Phase 1 — annotation** (`data/annotate.py`): a Claude LLM judge labels real
  dialogue pairs into `{user, reply, s_before, s_after}` with strict JSON output;
  a deterministic heuristic is the offline fallback. Semantics: `s_before` =
  state as the user message arrives; `s_after` = state after reading it (the mood
  that colours the reply, i.e. `S_t`).
* **Phase 2 — Affect Core** (`training/train_affect_core.py`): MSE on `ΔS`, plus
  an optional homeostasis regulariser.
* **Phase 3 — SFT** (`training/train_sft.py`): plain cross-entropy on the reply
  tokens, conditioned on `s_after`. **No loss coefficient is multiplied by
  emotion** (§5/§8) — the affect enters purely as input. The dataset teaches the
  state → reply-style mapping directly.

## 5. Inference (spec §6) — `emotive_llm/inference/runtime.py`

`EmotiveDialogue` runs the loop: `S_0` → Affect Core reacts to the user turn →
project → generate → optional empathic feedback (the model hears its own reply)
→ homeostasis pull + extreme-dwell guard.

## 6. Safety & Metrics (spec §7)

* **Homeostasis** (`affect/homeostasis.py`): a gentle per-step pull toward a
  light-positive target, plus `ExtremeDwellGuard` which forces a stronger
  relaxation once the state has dwelt in the high-distress corner for > N steps.
* **Metrics** (`eval/metrics.py`): `consistency` (predicted vs annotated ΔS, MSE
  + cosine) and `profile_adherence` (distance + prototype-label match) as
  automatable proxies for the human Likert evaluation, which remains a manual
  protocol.

## 7. Why conditioning, not loss weighting (spec §8)

Multiplying the loss by an emotion scalar destabilises training (a weight of 0
silences neurons, a large weight explodes gradients). Treating the affect vector
as a **conditioning input** gives the same "context shift" effect while keeping
the optimisation a standard, stable cross-entropy.
