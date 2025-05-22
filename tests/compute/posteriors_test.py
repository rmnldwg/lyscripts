"""Test utilities of the predict submodule."""

import numpy as np
import pytest
from lydata.utils import ModalityConfig

from lyscripts.compute.posteriors import compute_posteriors
from lyscripts.compute.priors import compute_priors
from lyscripts.compute.utils import complete_pattern
from lyscripts.configs import (
    DiagnosisConfig,
    DistributionConfig,
    GraphConfig,
    ModelConfig,
    add_distributions,
    construct_model,
)

RNG = np.random.default_rng(42)


@pytest.fixture(params=["Unilateral", "Bilateral"])
def model_config(request) -> ModelConfig:
    """Create unilateral model config."""
    return ModelConfig(class_name=request.param)


@pytest.fixture
def graph_config() -> GraphConfig:
    """Create simple graph."""
    return GraphConfig(
        tumor={"T": ["I", "II", "III"]},
        lnl={"I": ["II"], "II": ["III"], "III": []},
    )


@pytest.fixture
def dist_configs() -> dict[str, DistributionConfig]:
    """Provide early and late distributions."""
    return {
        "early": DistributionConfig(kind="frozen", func="binomial"),
        "late": DistributionConfig(kind="parametric", func="binomial"),
    }


@pytest.fixture
def modality_config() -> ModalityConfig:
    """Create modality config."""
    return ModalityConfig(spec=0.9, sens=0.8)


@pytest.fixture
def diagnosis_config() -> DiagnosisConfig:
    """Create a simple diagnosis config."""
    return DiagnosisConfig(
        ipsi={"D": {"I": True, "II": True, "III": False}},
        contra={"D": {"I": False, "II": True, "III": False}},
    )


@pytest.fixture
def samples(
    model_config: ModelConfig,
    graph_config: GraphConfig,
    dist_configs: dict[str, DistributionConfig],
) -> np.ndarray:
    """Generate some samples."""
    model = construct_model(model_config, graph_config)
    model = add_distributions(model, dist_configs)
    return RNG.uniform(size=(100, model.get_num_dims()))


@pytest.fixture
def priors(
    model_config: ModelConfig,
    graph_config: GraphConfig,
    dist_configs: dict[str, DistributionConfig],
    samples: np.ndarray,
) -> np.ndarray:
    """Provide some priors."""
    return compute_priors(
        model_config=model_config,
        graph_config=graph_config,
        dist_configs=dist_configs,
        samples=samples,
        t_stages=["late"],
        t_stages_dist=[1.0],
    )


def test_compute_posterior(
    model_config: ModelConfig,
    graph_config: GraphConfig,
    dist_configs: dict[str, DistributionConfig],
    modality_config: ModalityConfig,
    diagnosis_config: DiagnosisConfig,
    priors: np.ndarray,
) -> None:
    """Ensure that the diagnosis is correctly treated."""
    posteriors = compute_posteriors(
        model_config=model_config,
        graph_config=graph_config,
        dist_configs=dist_configs,
        modality_configs={"D": modality_config},
        priors=priors,
        diagnosis=diagnosis_config.model_dump(),
    )

    assert np.all(posteriors >= 0), "Negative probabilities in posterior."
    assert np.all(posteriors <= 1), "Probabilities above 1 in posterior."


def test_clean_pattern():
    """Test outdated utility function."""
    empty_pattern = {}
    one_pos_pattern = {"ipsi": {"II": True}}
    nums_pattern = {"ipsi": {"I": 1}, "contra": {"III": 0}}
    lnls = ["I", "II", "III"]

    empty_cleaned = complete_pattern(empty_pattern, lnls)
    one_pos_cleaned = complete_pattern(one_pos_pattern, lnls)
    nums_cleaned = complete_pattern(nums_pattern, lnls)

    assert empty_cleaned == {
        "ipsi": {"I": None, "II": None, "III": None},
        "contra": {"I": None, "II": None, "III": None},
    }, "Empty pattern does not get filled correctly."
    assert one_pos_cleaned == {
        "ipsi": {"I": None, "II": True, "III": None},
        "contra": {"I": None, "II": None, "III": None},
    }, "Pattern with one positive LNL not cleaned properly."
    assert nums_cleaned == {
        "ipsi": {"I": True, "II": None, "III": None},
        "contra": {"I": None, "II": None, "III": False},
    }, "Number pattern cleaned wrongly."
