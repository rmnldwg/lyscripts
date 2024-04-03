"""
Test the correct joining of datasets.
"""
from pathlib import Path

from lyscripts.data.join import load_and_join_tables


def test_load_and_join_tables():
    """Test the correct joining of datasets."""

    input_paths = [Path("./tests/data/b.csv"), Path("./tests/data/a.csv")]
    joined = load_and_join_tables(input_paths)

    assert joined.shape == (13,3), "Wrong concatenation shape."
    assert joined['x','z','b'].isna().sum() == 6, "Wrong number of NaNs."
    assert (joined['x','z','b'] == True).sum() == 4, "Wrong number of True values."
    assert (joined['x','z','b'] == False).sum() == 3, "Wrong number of False values."
    assert ('x', 'z', 'c') in joined.columns, "Column 'c' not found in joined table."
