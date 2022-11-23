"""
Test utilities of the predict submodule.
"""
from lyscripts.predict.utils import complete_pattern


def test_clean_pattern():
    """
    Test the utility function that cleans the involvement patterns from the
    `params.yaml` file
    """
    empty_pattern = {}
    one_pos_pattern = {"ipsi": {"II": True}}
    nums_pattern = {"ipsi": {"I": 1}, "contra": {"III": 0}}
    lnls = ["I", "II", "III"]

    empty_cleaned = complete_pattern(empty_pattern, lnls)
    one_pos_cleaned = complete_pattern(one_pos_pattern, lnls)
    nums_cleaned = complete_pattern(nums_pattern, lnls)

    assert empty_cleaned == {
        "ipsi": {"I": None, "II": None, "III": None},
        "contra": {"I": None, "II": None, "III": None}
    }, "Empty pattern does not get filled correctly."
    assert one_pos_cleaned == {
        "ipsi": {"I": None, "II": True, "III": None},
        "contra": {"I": None, "II": None, "III": None}
    }, "Pattern with one positive LNL not cleaned properly."
    assert nums_cleaned == {
        "ipsi": {"I": True, "II": None, "III": None},
        "contra": {"I": None, "II": None, "III": False}
    }, "Number pattern cleaned wrongly."
