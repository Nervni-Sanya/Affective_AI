"""Shared pytest fixtures. Everything uses tiny random-init models on CPU."""

import pytest
import torch

from emotive_llm.config import Config, ModelConfig
from emotive_llm.data.synthetic import generate_records


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def model_cfg():
    # Defaults are already the tiny, offline, random-init configuration.
    return ModelConfig()


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def records():
    return generate_records(24, seed=3)
