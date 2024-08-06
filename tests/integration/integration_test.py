"""Test the ``generate`` CLI."""

from pathlib import Path
import subprocess

import emcee
import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from lyscripts.configs import ScenarioConfig
from lyscripts.utils import load_patient_data, load_yaml_params


@pytest.fixture(scope="module")
def data_file() -> Path:
    """Return the path to the generated data."""
    res = Path("tests/integration/generated.csv")
    res.parent.mkdir(exist_ok=True)
    return res


@pytest.fixture(scope="module")
def sample_file() -> Path:
    """Return the path to the generated samples."""
    res = Path("tests/integration/samples.hdf5")
    res.parent.mkdir(exist_ok=True)
    return res


@pytest.fixture(scope="module")
def generated_data(data_file: Path) -> pd.DataFrame:
    """Generate a dataset and return it."""
    subprocess.run([
        "lyscripts",
        "data",
        "generate",
        "--configs",
        "tests/integration/model.ly.yaml",
        "tests/integration/graph.ly.yaml",
        "tests/integration/distributions.ly.yaml",
        "tests/integration/modalities.ly.yaml",
        "--num_patients", "200",
        "--output_file", str(data_file),
        "--seed", "42",
    ])
    return load_patient_data(data_file)


@pytest.fixture(scope="module")
def drawn_samples(
    generated_data: pd.DataFrame,
    data_file: Path,
    sample_file: Path,
) -> np.ndarray:
    """Draw samples from the defined model."""
    subprocess.run([
        "lyscripts",
        "sample",
        "--configs",
        "tests/integration/model.ly.yaml",
        "tests/integration/graph.ly.yaml",
        "tests/integration/distributions.ly.yaml",
        "tests/integration/modalities.ly.yaml",
        "tests/integration/sample.ly.yaml",
        "--data.input_file", str(data_file),
        "--sampling.output_file", str(sample_file),
    ])
    backend = emcee.backends.HDFBackend(sample_file, read_only=True)
    return backend.get_chain(flat=True)


@pytest.fixture(scope="module")
def scenarios() -> list[ScenarioConfig]:
    """Return a list of defined scenarios."""
    yaml_config = load_yaml_params("tests/integration/scenarios.ly.yaml")
    type_adapter = TypeAdapter(list[ScenarioConfig])
    return type_adapter.validate_python(yaml_config["scenarios"])


def test_generated_data(generated_data: pd.DataFrame) -> None:
    """Test the generated data."""
    assert generated_data.shape == (200, 3)


def test_scenarios(scenarios: list[ScenarioConfig]) -> None:
    """Check the loaded scenarios."""
    for scenario in scenarios:
        assert np.isclose(np.sum(scenario.t_stages_dist), 1.0)


def test_drawn_samples(drawn_samples: np.ndarray) -> None:
    """Test the drawn samples."""
    assert drawn_samples.shape[-1] == 4
