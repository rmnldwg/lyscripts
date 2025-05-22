"""Test the computation of the prevalences."""

import pandas as pd
import pytest
from lydata import infer_and_combine_levels, load_datasets

from lyscripts.compute.prevalences import observe_prevalence
from lyscripts.configs import DiagnosisConfig, ScenarioConfig


@pytest.fixture
def scenario_config() -> ScenarioConfig:
    """Create a simple scenario config."""
    return ScenarioConfig(
        t_stages=["early"],
        diagnosis=DiagnosisConfig(
            ipsi={"max_llh": {"II": "involved", "III": False}},
            contra={"max_llh": {"II": 0}},
        ),
    )


@pytest.fixture
def data() -> pd.DataFrame:
    """Load one of the lyDATA datasets."""
    data = next(load_datasets())
    return infer_and_combine_levels(data)


def test_observe_prevalence(
    data: pd.DataFrame,
    scenario_config: ScenarioConfig,
) -> None:
    """Ensure that observing the prevalence works."""
    portion = observe_prevalence(
        data=data,
        scenario_config=scenario_config,
    )

    assert portion.match == 67
    assert portion.total == 150
