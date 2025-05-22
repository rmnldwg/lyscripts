"""Test the ``generate`` CLI."""

import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from lydata import C
from lydata.utils import ModalityConfig
from pydantic import TypeAdapter

from lyscripts.cli import assemble_main
from lyscripts.compute.prevalences import PrevalencesCLI
from lyscripts.compute.priors import PriorsCLI, compute_priors
from lyscripts.compute.utils import get_cached
from lyscripts.configs import (
    DistributionConfig,
    GraphConfig,
    ModelConfig,
    SamplingConfig,
    ScenarioConfig,
)
from lyscripts.data.generate import GenerateCLI
from lyscripts.sample import SampleCLI
from lyscripts.utils import load_patient_data, load_yaml_params


@pytest.fixture(scope="session")
def monkeymodule():
    """Create a session scoped monkeypatch fixture.

    This can be used to e.g. mock the command line arguments by setting the
    ``sys.argv`` variable.
    """
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="session")
def data_file() -> Path:
    """Provide the path to the generated data.

    Delete any file at the beginning of a session if it exists.
    """
    res = Path("tests/integration/generated.csv")
    res.parent.mkdir(exist_ok=True)
    if res.exists():
        res.unlink()
    return res


@pytest.fixture(scope="session")
def samples_file() -> Path:
    """Provide the path to the generated samples.

    Delete any file at the beginning of a session if it exists.
    """
    res = Path("tests/integration/samples.hdf5")
    res.parent.mkdir(exist_ok=True)
    if res.exists():
        res.unlink()
    return res


def _get_config_file(name: str) -> Path:
    return Path(f"tests/integration/config/{name}.ly.yaml")


@pytest.fixture(scope="session")
def model_config_file() -> Path:
    """Provide the path to the model configuration file."""
    return _get_config_file("model")


@pytest.fixture(scope="session")
def graph_config_file() -> Path:
    """Provide the path to the graph configuration file."""
    return _get_config_file("graph")


@pytest.fixture(scope="session")
def distributions_config_file() -> Path:
    """Provide the path to the distributions configuration file."""
    return _get_config_file("distributions")


@pytest.fixture(scope="session")
def modalities_config_file() -> Path:
    """Provide the path to the modalities configuration file."""
    return _get_config_file("modalities")


@pytest.fixture(scope="session")
def scenarios_config_file() -> Path:
    """Provide the path to the scenarios configuration file."""
    return _get_config_file("scenarios")


@pytest.fixture(scope="session")
def sampling_config_file() -> Path:
    """Provide the path to the sampling configuration file."""
    return _get_config_file("sampling")


@pytest.fixture(scope="session")
def data_config_file() -> Path:
    """Provide the path to the data configuration file."""
    return _get_config_file("data")


@pytest.fixture(scope="session")
def model_config(model_config_file: Path) -> ModelConfig:
    """Provide the model configuration."""
    yaml_config = load_yaml_params(model_config_file)
    return ModelConfig(**yaml_config["model"])


@pytest.fixture(scope="session")
def graph_config(graph_config_file: Path) -> GraphConfig:
    """Provide the graph configuration."""
    yaml_config = load_yaml_params(graph_config_file)
    return GraphConfig(**yaml_config["graph"])


@pytest.fixture(scope="session")
def distributions_config(
    distributions_config_file: Path,
) -> dict[str, DistributionConfig]:
    """Provide the distributions configuration."""
    yaml_config = load_yaml_params(distributions_config_file)
    type_adapter = TypeAdapter(dict[str, DistributionConfig])
    return type_adapter.validate_python(yaml_config["distributions"])


@pytest.fixture(scope="session")
def modalities_config(modalities_config_file: Path) -> dict[str, ModalityConfig]:
    """Provide the modalities configuration."""
    yaml_config = load_yaml_params(modalities_config_file)
    type_adapter = TypeAdapter(dict[str, ModalityConfig])
    return type_adapter.validate_python(yaml_config["modalities"])


@pytest.fixture(scope="session")
def scenarios_config(scenarios_config_file: Path) -> list[ScenarioConfig]:
    """Provide a list of defined scenarios."""
    yaml_config = load_yaml_params(scenarios_config_file)
    type_adapter = TypeAdapter(list[ScenarioConfig])
    return type_adapter.validate_python(yaml_config["scenarios"])


@pytest.fixture(scope="session")
def sampling_config(sampling_config_file: Path) -> SamplingConfig:
    """Provide the sampling configuration."""
    yaml_config = load_yaml_params(sampling_config_file)
    return SamplingConfig(**yaml_config["sampling"])


@pytest.fixture(scope="session")
def generated_data(
    monkeymodule,
    data_file: Path,
    model_config_file: Path,
    graph_config_file: Path,
    distributions_config_file: Path,
    modalities_config_file: Path,
) -> pd.DataFrame:
    """Execute the generate CLI and provide the generated data as a fixture."""
    monkeymodule.setattr(
        sys,
        "argv",
        [
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
    )
    main = assemble_main(settings_cls=GenerateCLI, prog_name="generate")
    main()
    return load_patient_data(data_file)


@pytest.fixture(scope="session")
def drawn_samples(
    monkeymodule,
    generated_data: pd.DataFrame,
    data_file: Path,
    model_config_file: Path,
    graph_config_file: Path,
    distributions_config_file: Path,
    modalities_config_file: Path,
    sampling_config_file: Path,
    samples_file: Path,
) -> np.ndarray:
    """Execute the sampling CLI and provide the samples as a fixture."""
    monkeymodule.setattr(
        sys,
        "argv",
        [
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
            "--sampling.storage-file",
            str(samples_file.resolve()),
            # mapping because generated data already has the correct T-stage column
            '--data.mapping={"early": "early", "late": "late"}',
            "--data.source",
            str(data_file),
        ],
    )
    main = assemble_main(settings_cls=SampleCLI, prog_name="sample")
    main()
    _yaml_params = load_yaml_params(sampling_config_file)
    _sampling_config = SamplingConfig(
        storage_file=samples_file, **_yaml_params["sampling"]
    )
    return _sampling_config.load()


@pytest.fixture(scope="session")
def cache_dir() -> Path:
    """Provide the path to the cache directory as a fixture.

    Delete any directory at the beginning of a session if it exists.
    """
    res = Path("tests/integration/.cache")
    if res.exists():
        shutil.rmtree(res)
    return res


@pytest.fixture(scope="session")
def priors_file() -> Path:
    """Provide the path to the computed priors as a fixture.

    Delete any file at the beginning of a session if it exists.
    """
    res = Path("tests/integration/priors.hdf5")
    res.parent.mkdir(exist_ok=True)
    if res.exists():
        res.unlink()
    return res


@pytest.fixture(scope="session")
def prevalences_file() -> Path:
    """Provide the path to the computed prevalences as a fixture.

    Delete any file at the beginning of a session if it exists.
    """
    res = Path("tests/integration/prevalences.hdf5")
    res.parent.mkdir(exist_ok=True)
    if res.exists():
        res.unlink()
    return res


@pytest.fixture(scope="session")
def computed_priors(
    monkeymodule,
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
    """Execute the ``priors`` CLI and provide the computed arrays as a fixture."""
    monkeymodule.setattr(
        sys,
        "argv",
        [
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
            "--sampling.storage-file",
            str(samples_file.resolve()),
            "--priors.file",
            str(priors_file),
        ],
    )
    main = assemble_main(settings_cls=PriorsCLI, prog_name="priors")
    main()
    with h5py.File(priors_file, "r") as h5file:
        return h5file[dataset][:]


@pytest.fixture(scope="session")
def computed_prevalences(
    monkeymodule,
    cache_dir: Path,
    model_config_file: Path,
    graph_config_file: Path,
    distributions_config_file: Path,
    scenarios_config_file: Path,
    modalities_config_file: Path,
    sampling_config_file: Path,
    samples_file: Path,
    prevalences_file: Path,
    data_file: Path,
    # to ensure the correct execution order, also require data and samples
    generated_data: pd.DataFrame,
    drawn_samples: np.ndarray,
    dataset: str = "000",
) -> tuple[np.ndarray, int, int]:
    """Provide the computed prevalences as a fixture."""
    monkeymodule.setattr(
        sys,
        "argv",
        [
            "prevalences",
            "--cache-dir",
            str(cache_dir.resolve()),
            "--configs",
            str(graph_config_file.resolve()),
            "--configs",
            str(model_config_file.resolve()),
            "--configs",
            str(distributions_config_file.resolve()),
            "--configs",
            str(scenarios_config_file.resolve()),
            "--configs",
            str(modalities_config_file.resolve()),
            "--configs",
            str(sampling_config_file.resolve()),
            "--sampling.storage-file",
            str(samples_file.resolve()),
            "--prevalences.file",
            str(prevalences_file),
            "--data.source",
            str(data_file.resolve()),
            "--data.mapping",
            '{"early": "early", "late": "late"}',
        ],
    )
    main = assemble_main(settings_cls=PrevalencesCLI, prog_name="prevalences")
    main()
    with h5py.File(prevalences_file, "r") as h5file:
        return (
            h5file[dataset][:],
            h5file[dataset].attrs["num_match"],
            h5file[dataset].attrs["num_total"],
        )


def test_generated_data(generated_data: pd.DataFrame) -> None:
    """Test the generated data."""
    assert generated_data.shape == (200, 3)
    assert (
        generated_data["imaging", "ipsi", "II"].sum()
        > generated_data["imaging", "ipsi", "III"].sum()
    )
    assert generated_data.ly.t_stage.isin(["early", "late"]).all()
    assert all(
        generated_data.ly.query(C("t_stage") == "early")["imaging", "ipsi"].mean()
        < generated_data.ly.query(C("t_stage") == "late")["imaging", "ipsi"].mean()
    )


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


def test_computed_prevalences(
    computed_prevalences: tuple[np.ndarray, int, int],
) -> None:
    """Test the computed prevalences."""
    prevalences, num_match, num_total = computed_prevalences
    num_match, num_total = int(num_match), int(num_total)
    assert num_match == 64
    assert num_total == 123
