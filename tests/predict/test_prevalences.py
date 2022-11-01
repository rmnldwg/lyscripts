"""
Test the functions of the prevalence prediction submodule.
"""
import lymph
import pandas as pd

from lyscripts.predict.prevalences import get_lnls, get_match_idx


def test_get_lnls():
    """Check if LNLs are correctly extracted from lymph model."""
    graph = {
        ("tumor", "primary"): ["I", "II", "III"],
        ("lnl", "I"): ["II"],
        ("lnl", "II"): ["III"],
        ("lnl", "III"): [],
    }
    lnls = graph[("tumor", "primary")]
    uni_model = lymph.Unilateral(graph)
    bi_model = lymph.Bilateral(graph)
    mid_model = lymph.MidlineBilateral(graph)

    uni_lnls = get_lnls(uni_model)
    bi_lnls = get_lnls(bi_model)
    mid_lnls = get_lnls(mid_model)

    assert uni_lnls == lnls, "Did not extract LNLs correctly from unilateral model"
    assert bi_lnls == lnls, "Did not extract LNLs correctly from bilateral model"
    assert mid_lnls == lnls, "Did not extract LNLs correctly from midline model"


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
        True, oneside_pattern, three_patients, lnls, invert=False
    )
    inverted_matching_idxs = get_match_idx(
        False, oneside_pattern, three_patients, lnls, invert=True
    )
    ignorant_matching_idxs = get_match_idx(
        True, ignorant_pattern, three_patients, lnls, invert=False
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
