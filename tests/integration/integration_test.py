"""Test the ``generate`` CLI."""

import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from lydata.utils import ModalityConfig
from pydantic import TypeAdapter

from lyscripts.compute.priors import compute_priors
from lyscripts.compute.utils import get_cached
from lyscripts.configs import (
    DistributionConfig,
    GraphConfig,
    ModelConfig,
    SamplingConfig,
    ScenarioConfig,
)
from lyscripts.utils import load_patient_data, load_yaml_params


@pytest.fixture(scope="session")
def data_file() -> Path:
    """Return the path to the generated data."""
    res = Path("tests/integration/generated.csv")
    res.parent.mkdir(exist_ok=True)
    if res.exists():
        res.unlink()
    return res


@pytest.fixture(scope="session")
def samples_file() -> Path:
    """Return the path to the generated samples."""
    res = Path("tests/integration/samples.hdf5")
    res.parent.mkdir(exist_ok=True)
    if res.exists():
        res.unlink()
    return res


@pytest.fixture(scope="session")
def model_config_file() -> Path:
    """Return the path to the model configuration file."""
    return Path("tests/integration/model.ly.yaml")


@pytest.fixture(scope="session")
def graph_config_file() -> Path:
    """Return the path to the graph configuration file."""
    return Path("tests/integration/graph.ly.yaml")


@pytest.fixture(scope="session")
def distributions_config_file() -> Path:
    """Return the path to the distributions configuration file."""
    return Path("tests/integration/distributions.ly.yaml")


@pytest.fixture(scope="session")
def modalities_config_file() -> Path:
    """Return the path to the modalities configuration file."""
    return Path("tests/integration/modalities.ly.yaml")


@pytest.fixture(scope="session")
def scenarios_config_file() -> Path:
    """Return the path to the scenarios configuration file."""
    return Path("tests/integration/scenarios.ly.yaml")


@pytest.fixture(scope="session")
def sampling_config_file() -> Path:
    """Return the path to the sampling configuration file."""
    return Path("tests/integration/sampling.ly.yaml")


@pytest.fixture(scope="session")
def model_config(model_config_file: Path) -> ModelConfig:
    """Return the model configuration."""
    yaml_config = load_yaml_params(model_config_file)
    return ModelConfig(**yaml_config["model"])


@pytest.fixture(scope="session")
def graph_config(graph_config_file: Path) -> GraphConfig:
    """Return the graph configuration."""
    yaml_config = load_yaml_params(graph_config_file)
    return GraphConfig(**yaml_config["graph"])


@pytest.fixture(scope="session")
def distributions_config(
    distributions_config_file: Path,
) -> dict[str, DistributionConfig]:
    """Return the distributions configuration."""
    yaml_config = load_yaml_params(distributions_config_file)
    type_adapter = TypeAdapter(dict[str, DistributionConfig])
    return type_adapter.validate_python(yaml_config["distributions"])


@pytest.fixture(scope="session")
def modalities_config(modalities_config_file: Path) -> dict[str, ModalityConfig]:
    """Return the modalities configuration."""
    yaml_config = load_yaml_params(modalities_config_file)
    type_adapter = TypeAdapter(dict[str, ModalityConfig])
    return type_adapter.validate_python(yaml_config["modalities"])


@pytest.fixture(scope="session")
def scenarios_config(scenarios_config_file: Path) -> list[ScenarioConfig]:
    """Return a list of defined scenarios."""
    yaml_config = load_yaml_params(scenarios_config_file)
    type_adapter = TypeAdapter(list[ScenarioConfig])
    return type_adapter.validate_python(yaml_config["scenarios"])


@pytest.fixture(scope="session")
def sampling_config(sampling_config_file: Path) -> SamplingConfig:
    """Return the sampling configuration."""
    yaml_config = load_yaml_params(sampling_config_file)
    return SamplingConfig(**yaml_config["sampling"])


@pytest.fixture(scope="session")
def generated_data(
    data_file: Path,
    model_config_file: Path,
    graph_config_file: Path,
    distributions_config_file: Path,
    modalities_config_file: Path,
) -> pd.DataFrame:
    """Generate a dataset and return it."""
    subprocess.run(
        [
            "lyscripts",
            "--log-level",
            "DEBUG",
            "data",
            "generate",
            "--configs",
            str(model_config_file.resolve()),
            "--configs",
            str(graph_config_file.resolve()),
            "--configs",
            str(distributions_config_file.resolve()),
            "--configs",
            str(modalities_config_file.resolve()),
            "--num-patients",
            "200",
            "--output-file",
            str(data_file),
            "--seed",
            "42",
        ],
        check=True,
    )
    return load_patient_data(data_file)


@pytest.fixture(scope="session")
def drawn_samples(
    generated_data: pd.DataFrame,
    data_file: Path,
    model_config_file: Path,
    graph_config_file: Path,
    distributions_config_file: Path,
    modalities_config_file: Path,
    sampling_config_file: Path,
    samples_file: Path,
) -> np.ndarray:
    """Draw samples from the defined model."""
    subprocess.run(
        [
            "lyscripts",
            "--log-level",
            "DEBUG",
            "sample",
            "--configs",
            str(model_config_file.resolve()),
            "--configs",
            str(graph_config_file.resolve()),
            "--configs",
            str(distributions_config_file.resolve()),
            "--configs",
            str(modalities_config_file.resolve()),
            "--configs",
            str(sampling_config_file.resolve()),
            "--sampling.file",
            str(samples_file.resolve()),
            "--data.source",
            str(data_file),
        ],
        check=True,
    )
    _yaml_params = load_yaml_params(sampling_config_file)
    _sampling_config = SamplingConfig(file=samples_file, **_yaml_params["sampling"])
    return _sampling_config.load()


@pytest.fixture(scope="session")
def cache_dir() -> Path:
    """Return the path to the empty cache directory."""
    res = Path("tests/integration/.cache")
    if res.exists():
        shutil.rmtree(res)
    return res


@pytest.fixture(scope="session")
def priors_file() -> Path:
    """Return the path to the computed priors."""
    res = Path("tests/integration/priors.hdf5")
    res.parent.mkdir(exist_ok=True)
    if res.exists():
        res.unlink()
    return res


@pytest.fixture(scope="session")
def computed_priors(
    cache_dir: Path,
    model_config_file: Path,
    graph_config_file: Path,
    distributions_config_file: Path,
    scenarios_config_file: Path,
    sampling_config_file: Path,
    samples_file: Path,
    priors_file: Path,
    dataset: str = "000",
) -> np.ndarray:
    """Compute the priors for the drawn samples."""
    subprocess.run(
        [
            "lyscripts",
            "--log-level",
            "DEBUG",
            "compute",
            "priors",
            "--cache-dir",
            str(cache_dir.resolve()),
            "--configs",
            str(model_config_file.resolve()),
            "--configs",
            str(graph_config_file.resolve()),
            "--configs",
            str(distributions_config_file.resolve()),
            "--configs",
            str(scenarios_config_file.resolve()),
            "--configs",
            str(sampling_config_file.resolve()),
            "--sampling.file",
            str(samples_file.resolve()),
            "--priors.file",
            str(priors_file),
        ],
        check=True,
    )
    with h5py.File(priors_file, "r") as h5file:
        return h5file[dataset][:]


def test_generated_data(generated_data: pd.DataFrame) -> None:
    """Test the generated data."""
    assert generated_data.shape == (200, 3)


def test_scenarios(scenarios_config: list[ScenarioConfig]) -> None:
    """Check the loaded scenarios."""
    for scenario in scenarios_config:
        assert np.isclose(np.sum(scenario.t_stages_dist), 1.0)


def test_drawn_samples(drawn_samples: np.ndarray) -> None:
    """Test the drawn samples."""
    assert drawn_samples.shape[-1] == 4


def test_computed_priors(
    cache_dir: Path,
    model_config: ModelConfig,
    graph_config: GraphConfig,
    distributions_config: dict[str, DistributionConfig],
    drawn_samples: np.ndarray,
    scenarios_config: list[ScenarioConfig],
    computed_priors: np.ndarray,
) -> None:
    """Test the computed priors."""
    scenario = scenarios_config[0]
    kwargs = scenario.model_dump(include={"t_stages", "t_stages_dist", "mode"})
    cached_compute_priors = get_cached(compute_priors, cache_dir)

    kwargs.update(
        {
            "model_config": model_config,
            "graph_config": graph_config,
            "dist_configs": distributions_config,
            "samples": drawn_samples,
        }
    )
    assert cached_compute_priors._cached_func.check_call_in_cache(**kwargs)
    cached_output = cached_compute_priors(**kwargs)
    assert np.allclose(computed_priors, cached_output)

    assert computed_priors.shape[-1] == 4
    assert np.all(computed_priors >= 0.0)
    assert np.all(computed_priors <= 1.0)
