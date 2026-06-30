"""EmotiveLLM (MoonCore-Emotive).

A dialogue language model with an internal 7-dimensional affective state that
evolves with conversation context and *conditions* generation (soft prompting)
rather than scaling the loss. See ``docs/design.md`` for the mapping to the
original specification.

Public surface is intentionally small; import submodules directly for the rest:

    from emotive_llm.affect.space import AffectSpace, default_state
    from emotive_llm.affect.core import AffectCore
    from emotive_llm.conditioning.projector import AffectProjector
    from emotive_llm.backbone.conditioned_lm import ConditionedLM
    from emotive_llm.inference.runtime import EmotiveDialogue
"""

from emotive_llm.config import (
    AFFECT_DIM,
    Config,
    DataConfig,
    ModelConfig,
    TrainConfig,
    load_config,
)

__all__ = [
    "AFFECT_DIM",
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "load_config",
]

__version__ = "0.1.0"
