"""Test the configs module."""

from pathlib import Path

import pytest
import yaml
from lydata.utils import ModalityConfig
from pydantic import ValidationError

from lyscripts.configs import (
    DistributionConfig,
    GraphConfig,
    ModelConfig,
    add_distributions,
    add_modalities,
    construct_model,
    modalityconfig_from_model,
)


@pytest.fixture
def yaml_config_dict() -> dict:
    """Load the YAML params file."""
    with open("tests/test_params_v1.yaml") as file:
        return yaml.safe_load(file)


@pytest.fixture
def external_model_config() -> ModelConfig:
    return ModelConfig(external_file=Path("tests/_dummy_model.py"))


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        class_name="Bilateral",
        constructor="trinary",
        named_params=["spread", "TtoII_spread", "late_p"],
    )


@pytest.fixture
def graph_config() -> GraphConfig:
    return GraphConfig(
        tumor={"T": ["II", "III"]},
        lnl={"II": ["III"], "III": []},
    )


@pytest.fixture
def distribution_configs() -> dict[str, DistributionConfig]:
    return {
        "early": DistributionConfig(kind="frozen", params={"p": 0.3}),
        "late": DistributionConfig(kind="parametric", params={"p": 0.7}),
    }


def test_model_from_external(
    external_model_config: ModelConfig,
    graph_config: GraphConfig,
):
    """Check if loading model from external file works."""
    model = construct_model(external_model_config, graph_config)
    assert model.was_externally_loaded


def test_no_model_from_external() -> None:
    """Ensure a `ValidationError` is raised when no model is provided."""
    with pytest.raises(ValidationError):
        ModelConfig(external_file=Path("tests/_dummy_no_model.py"))


def test_model_from_no_file() -> None:
    """Ensure a `ValidationError` is raised when the file does not exist."""
    with pytest.raises(ValidationError):
        ModelConfig(external_file=Path("tests/_no_file.py"))


def test_model_from_config(
    model_config: ModelConfig,
    graph_config: GraphConfig,
    distribution_configs: dict[str, DistributionConfig],
):
    """Check that loading the model works correctly. Especially the named params."""
    model = construct_model(
        model_config=model_config,
        graph_config=graph_config,
    )
    model = add_distributions(
        model=model,
        configs=distribution_configs,
    )
    assert model.ipsi.get_distribution(t_stage="late") == model.contra.get_distribution(
        t_stage="late"
    )
    assert model.get_num_dims() == len(model_config.named_params)


def test_config_recovery(yaml_config_dict: dict) -> None:
    """Ensure the round trip config -> mode -> config works."""
    model = construct_model(
        model_config=ModelConfig(**yaml_config_dict["model"]),
        graph_config=GraphConfig(**yaml_config_dict["graph"]),
    )
    model = add_modalities(
        model=model,
        modalities={
            name: ModalityConfig(**config)
            for name, config in yaml_config_dict["modalities"].items()
        },
    )

    recovered_config = {}
    recovered_config["model"] = ModelConfig.from_model(model).model_dump(
        exclude_none=True
    )
    recovered_config["graph"] = GraphConfig.from_model(model).model_dump()
    recovered_config["modalities"] = {
        name: modalityconfig_from_model(model, name).model_dump()
        for name in model.get_all_modalities()
    }

    yaml_config_dict.pop("distributions")
    yaml_config_dict.pop("version")
    assert recovered_config == yaml_config_dict
