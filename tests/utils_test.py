"""
Test the core utility functions of the package.
"""
import pytest
from lymph import models

from lyscripts.utils import (
    create_model,
    flatten,
    get_modalities_subset,
    load_yaml_params,
)


@pytest.fixture
def params_v1() -> dict:
    """Fixture loading the `test_parmams_v1.yaml` file."""
    return load_yaml_params("./tests/test_params_v1.yaml")


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


def test_create_model(params_v1):
    """Test the creation of a model."""
    model = create_model(params_v1)
    assert isinstance(model, models.Midline), "Model has wrong type"
    assert model.use_mixing, "Model should use mixing"
    assert not model.use_central, "Model should not use central"
    assert model.use_midext_evo, "Model should use midext evolution"
    assert "early" in model.get_all_distributions(), "Early distribution is missing"
    assert not model.get_distribution("early").is_updateable, "Early distribution should not be updateable"
    assert "late" in model.get_all_distributions(), "Late distribution is missing"
    assert model.get_distribution("late").is_updateable, "Late distribution should be updateable"
    assert "CT" in model.get_all_modalities(), "CT modality is missing"
    assert model.get_modality("CT").spec == 0.76, "CT modality has wrong specificity"
    assert model.get_modality("CT").sens == 0.81, "CT modality has wrong sensitivity"
    assert "FNA" in model.get_all_modalities(), "MRI modality is missing"
    assert model.get_modality("FNA").spec == 0.98, "MRI modality has wrong specificity"
    assert model.get_modality("FNA").sens == 0.80, "MRI modality has wrong sensitivity"
