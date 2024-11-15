"""Test the configs module."""

from pathlib import Path

import pytest

from lyscripts.configs import GraphConfig, ModelConfig, construct_model


@pytest.fixture
def external_model_config() -> ModelConfig:
    return ModelConfig(external=Path("tests/_dummy_model.py"))


@pytest.fixture
def graph_config() -> GraphConfig:
    return GraphConfig(
        tumor={"T": ["II", "III"]},
        lnl={"II": ["III"], "III": []},
    )


def test_model_from_external(
    external_model_config: ModelConfig,
    graph_config: GraphConfig,
):
    """Check if loading model from external file works."""
    model = construct_model(external_model_config, graph_config)
    assert model.was_externally_loaded
