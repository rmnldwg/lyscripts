"""
Utilities related to the commands for data cleaning and processing.
"""
from pathlib import Path

import pandas as pd

from lyscripts.decorators import check_output_dir_exists, log_state


@log_state()
@check_output_dir_exists
def save_table_to_csv(output_path: Path, table: pd.DataFrame):
    """Save a ``table`` to ``output_path``."""
    table.to_csv(output_path, index=None)
