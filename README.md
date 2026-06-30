# Affective_AI — EmotiveLLM (MoonCore-Emotive)

A dialogue LLM with an internal **7-dimensional affective state** (Valence,
Arousal, Dominance, Warmth, Interest, Honesty, Resilience) that evolves with
conversation context and **conditions** generation via soft prompting. A
recurrent **Affect Core** models realistic emotion dynamics; the affect vector
is a *conditioning signal*, never a loss coefficient — giving the "context
shift" effect without destabilising training.

See [`docs/design.md`](docs/design.md) for the full mapping to the specification.

## Architecture

```
user turn ──► Affect Core ──► S_t (7-D state) ──► Projector ──► soft prompt
   ▲          (DistilBERT +        │                 (MLP 7→128→d_model)  │
   │           decay cell)         │                                     ▼
   └── empathic feedback ◄─────────┘                         Conditioned GPT-2 ──► reply
                                          homeostasis + extreme-dwell guard
```

* **Affect Core** — `S_t = bounded(α·S_{t-1} + (1-α)·(W·h_t + b))`, per-axis
  inertia with a resilience-driven fatigue mechanism.
* **Backbone** — GPT-like causal LM; the projected affect vector is prepended as
  prefix embeddings (soft prompt).
* **Two model sizes** — tiny random-init configs run offline on CPU (for tests
  and the demo); set `*_pretrained: true` (config `base`/`prod`) to load real
  `distilbert-base` + `gpt2` weights.

## Install

```bash
pip install -r requirements.txt          # core: torch, transformers, numpy, pyyaml
pip install datasets anthropic pytest     # real data (DailyDialog) + LLM judge + tests
```

## Quickstart (offline, CPU, tiny models)

Run the whole pipeline — Phase 1 (annotate) → Phase 2 (Affect Core) → Phase 3
(SFT) — on synthetic data:

```bash
python scripts/run_pipeline.py --config tiny --steps 20
```

Watch the affect trajectory across a scripted dialogue:

```bash
python scripts/demo_chat.py --config tiny
```

Train a phase on its own:

```bash
python -m emotive_llm.training.train_affect_core --config tiny --steps 20   # Phase 2
python -m emotive_llm.training.train_sft         --config tiny --steps 20   # Phase 3
```

Interactive chat (prints the affect state each turn):

```bash
python -m emotive_llm.inference.runtime --config tiny \
    --affect-core checkpoints/affect_core.pt --sft checkpoints/sft.pt
```

> With tiny random-init models the generated text is gibberish — the demo shows
> the **state dynamics** and the loop. Use `--config base` (pretrained weights,
> GPU recommended) for fluent replies.

## Real data + LLM-judge annotation

`configs/base.yaml` and `configs/prod.yaml` load real weights and use
**DailyDialog** (via `datasets`). Phase 1 annotation calls a Claude judge
(`anthropic` SDK, model `claude-opus-4-8`) to label dialogues into
`{user, reply, s_before, s_after}`; with no API key it falls back to a
deterministic heuristic. Inspect the corpus:

```bash
python -m emotive_llm.data.real --limit 5
```

## Tests

```bash
pytest -q
```

## Configs

| config | models | data | use |
|--------|--------|------|-----|
| `tiny` | random-init mini | synthetic | offline CPU tests/demo |
| `base` | distilbert-base + gpt2 | DailyDialog | single GPU |
| `prod` | distilbert-base + gpt2-medium | full DailyDialog, AMP + grad-accum | production-scale (GPU) |

## Evaluation

`emotive_llm/eval/metrics.py` provides **consistency** (predicted vs annotated
ΔS) and **profile adherence** (prototype-label match) as automatable proxies.
The human Likert assessment of "does the reply match the stated affective
profile" remains a manual protocol.

## Layout

```
emotive_llm/
  affect/        space, core (dynamics), homeostasis
  conditioning/  affect → soft-prompt projector
  backbone/      affect-conditioned causal LM
  data/          schema, synthetic, real (DailyDialog), annotate (LLM judge), datasets
  training/      shared loop, Phase 2, Phase 3
  inference/     EmotiveDialogue runtime
  eval/          metrics
configs/         tiny | base | prod
scripts/         demo_chat.py, run_pipeline.py
tests/           pytest suite
docs/design.md   spec mapping
```
