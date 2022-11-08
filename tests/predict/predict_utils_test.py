"""
Test utilities of the predict submodule.
"""
from lyscripts.predict.utils import clean_pattern, rich_enumerate


def test_clean_pattern():
    """
    Test the utility function that cleans the involvement patterns from the
    `params.yaml` file
    """
    empty_pattern = {}
    one_pos_pattern = {"ipsi": {"II": True}}
    nums_pattern = {"ipsi": {"I": 1}, "contra": {"III": 0}}
    lnls = ["I", "II", "III"]

    empty_cleaned = clean_pattern(empty_pattern, lnls)
    one_pos_cleaned = clean_pattern(one_pos_pattern, lnls)
    nums_cleaned = clean_pattern(nums_pattern, lnls)

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


def test_rich_enumerate():
    """Make sure the enumeration with possible rich progress bar works."""
    samples = [1, 2, 3, 4]

    enum_with_desc = rich_enumerate(
        samples,
        description="A dummy description",
    )
    enum_without_desc = rich_enumerate(samples)

    for item in enum_with_desc:
        assert len(item) == 2, "Rich enumeration does not work."
    for item in enum_without_desc:
        assert len(item) == 2, "Non-rich enumeration does not work."
