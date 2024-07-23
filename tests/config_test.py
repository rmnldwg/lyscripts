"""Test the dataclass configs."""

import pytest
import dacite

from lyscripts.utils import load_yaml_params, flatten
from lyscripts.configs import (
    DistributionConfig,
    ModalityConfig,
    ModelConfig,
    add_dists,
    add_modalities,
    construct_model,
)


@pytest.fixture
def path() -> str:
    """Fixture returning the path to the `test_params_v1.yaml` file."""
    return "./tests/test_params_v1.yaml"

@pytest.fixture
def params_v1(path) -> dict:
    """Fixture loading the `test_params_v1.yaml` file."""
    return load_yaml_params(path)

@pytest.fixture
def graph_dict(params_v1) -> dict:
    """Fixture loading the graph dictionary."""
    return params_v1["graph"]

@pytest.fixture
def model_config(params_v1) -> ModelConfig:
    """Fixture loading the model configuration."""
    return dacite.from_dict(ModelConfig, params_v1["model"])

@pytest.fixture
def dists(params_v1) -> dict[str, DistributionConfig]:
    """Fixture loading the distribution dictionaries."""
    return {
        key: dacite.from_dict(DistributionConfig, value)
        for key, value in params_v1["distributions"].items()
    }

@pytest.fixture
def modalities(params_v1) -> dict[str, ModalityConfig]:
    """Fixture loading the modality dictionaries."""
    return {
        key: dacite.from_dict(ModalityConfig, value)
        for key, value in params_v1["modalities"].items()
    }


def test_model_config(path, model_config):
    """Test creating the model config from a YAML file."""
    assert ModelConfig.from_params(path) == model_config, "Model config not loaded properly."

def test_dists(path, dists):
    """Test creating the distribution config from a YAML file."""
    assert DistributionConfig.dict_from_params(path) == dists, "Distributions not loaded properly."

def test_modalities(path, modalities):
    """Test creating the modality config from a YAML file."""
    assert ModalityConfig.dict_from_params(path) == modalities, "Modalities not loaded properly."


def test_construct_model(model_config, graph_dict, dists, modalities):
    """Test the model construction."""
    model = construct_model(model_config, flatten(graph_dict))
    model = add_dists(model, dists)
    model = add_modalities(model, modalities)

    assert model.max_time == 10, "Max time was not set properly."
