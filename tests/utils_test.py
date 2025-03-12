"""Test the core utility functions of the package."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import TypeAdapter

from lyscripts.configs import DeprecatedModelConfig, DistributionConfig, ModelConfig
from lyscripts.utils import (
    flatten,
    get_modalities_subset,
)


def test_flatten():
    """Check if the dictionary flattening works."""
    nested = {
        "A": {"a": 1, "b": 2, "c": 3},
        "B": {"a": 4, "b": 5, "c": 6},
        "C": {"a": {"i": 7, "ii": 8}},
    }
    exp_flattened = {
        ("A", "a"): 1,
        ("A", "b"): 2,
        ("A", "c"): 3,
        ("B", "a"): 4,
        ("B", "b"): 5,
        ("B", "c"): 6,
        ("C", "a", "i"): 7,
        ("C", "a", "ii"): 8,
    }

    actual_flattened = flatten(nested)
    assert actual_flattened == exp_flattened, "Dictionary was not flattened properly."


def test_get_modalities_subset():
    """Test the extraction of a modality subset."""
    modalities = {
        "CT": [0.76, 0.81],
        "MRI": [0.63, 0.86],
        "PET": [0.79, 0.83],
        "path": [1.0, 1.0],
    }
    selected = ["CT", "path"]
    exp_subset = {
        "CT": [0.76, 0.81],
        "path": [1.0, 1.0],
    }

    actual_subset = get_modalities_subset(modalities, selected)
    assert actual_subset == exp_subset, "Extraction of modalities did not work."


@pytest.fixture
def v0_config() -> dict[str, Any]:
    """Return a deprecated model configuration."""
    config_path = Path("tests/test_params_v0.yaml")
    with open(config_path) as config_file:
        return yaml.safe_load(config_file)


@pytest.fixture
def v1_config() -> dict[str, Any]:
    """Return a deprecated model configuration."""
    config_path = Path("tests/test_params_v1.yaml")
    with open(config_path) as config_file:
        return yaml.safe_load(config_file)


def test_translate_deprecated_model_config(
    v0_config: dict[str, Any],
    v1_config: dict[str, Any],
):
    """Test the translation of the deprecated model configuration."""
    adapter = TypeAdapter(dict[str | int, DistributionConfig])

    old_model_config = DeprecatedModelConfig(**v0_config["model"])
    exp_model_config = ModelConfig(**v1_config["model"])
    exp_dist_configs = adapter.validate_python(v1_config["distributions"])

    trans_model_config, trans_dist_configs = old_model_config.translate()

    assert (  # noqa
        exp_model_config.model_dump(exclude="kwargs")
        == trans_model_config.model_dump(exclude="kwargs")
    )
    assert exp_dist_configs == trans_dist_configs
