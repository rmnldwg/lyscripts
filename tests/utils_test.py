"""
Test the core utility functions of the package.
"""
from lyscripts.utils import flatten, get_modalities_subset


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
