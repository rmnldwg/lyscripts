"""Test the ``generate`` CLI."""

import subprocess

import pandas as pd
import pytest

from lyscripts.utils import load_patient_data


@pytest.fixture
def generated_data() -> pd.DataFrame:
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
        "--output_file", "tests/integration/generated.csv",
        "--seed", "42",
    ])
    return load_patient_data("tests/integration/generated.csv")


def test_generated_data(generated_data: pd.DataFrame):
    """Test the generated data."""
    assert generated_data.shape == (200, 3)
