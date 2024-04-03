"""
Test the functions of the prevalence prediction submodule.
"""
import pandas as pd
import pytest

from lyscripts.compute.prevalences import does_midext_match
from lyscripts.data.accessor import get_match_idx


def test_get_match_idx():
    """Test if the pattern dictionaries & pandas data are compared correctly."""
    oneside_pattern = {"I": False, "II": True, "III": None}
    ignorant_pattern = {"I": None, "II": None, "III": None}
    three_patients = pd.DataFrame.from_dict({
        "I":   [False, False, True],
        "II":  [True , True , True],
        "III": [True , False, True],
    })
    lnls = list(oneside_pattern.keys())

    matching_idxs = get_match_idx(
        True, oneside_pattern, three_patients, invert=False
    )
    inverted_matching_idxs = get_match_idx(
        False, oneside_pattern, three_patients, invert=True
    )
    ignorant_matching_idxs = get_match_idx(
        True, ignorant_pattern, three_patients, invert=False
    )

    assert all(
        matching_idxs == pd.Series([True, True, False])
    ), "Patients were incorrectly matched with pattern."
    assert all(
        inverted_matching_idxs == pd.Series([False, False, True])
    ), "Inverse matching did not work correclty."
    assert all(
        ignorant_matching_idxs == pd.Series([True, True, True])
    ), "Ignorant matching did not always return `True`."


def test_does_midline_ext_match():
    """
    Test the function that returns indices of a `DataFrame` where the midline
    extension matches.
    """
    midline_data = pd.DataFrame({
        ("tumor", "1", "extension"): [True, False, None]
    })
    keyerr_data = pd.DataFrame({("way", "too", "many", "levels"): [True, False, None]})

    midline_match = does_midext_match(midline_data, midext=False)

    assert all(midline_match == pd.Series([False, True, False])), (
        "Matching midline extension with correct data does not work."
    )
    with pytest.raises(KeyError):
        _ = does_midext_match(keyerr_data, midext=False)
