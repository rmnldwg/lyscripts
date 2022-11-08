"""
Utilities related to the commands for data cleaning and processing.
"""
from pathlib import Path
from typing import Union

import pandas as pd

from lyscripts.utils import check_output_dir_exists, report_func_state


@report_func_state(
    status_msg="Save processed CSV file...",
    success_msg="Saved processed CSV file.",
)
@check_output_dir_exists
def save_table_to_csv(
    output_path: Union[str, Path],
    table: pd.DataFrame,
):
    """Save a `pd.DataFrame` to `output_path`."""
    table.to_csv(output_path, index=None)
