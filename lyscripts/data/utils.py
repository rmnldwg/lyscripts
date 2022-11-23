"""
Utilities related to the commands for data cleaning and processing.
"""
from pathlib import Path
from typing import List

import pandas as pd

from lyscripts.utils import (
    check_input_file_exists,
    check_output_dir_exists,
    raise_if_args_none,
    report_state,
)


@report_state(
    status_msg="Save processed CSV file...",
    success_msg="Saved processed CSV file.",
)
@check_output_dir_exists
@raise_if_args_none(message="Specify table to save", level="warning")
def save_table_to_csv(output_path: Path, table: pd.DataFrame):
    """Save a `pd.DataFrame` to `output_path`."""
    table.to_csv(output_path, index=None)


@report_state(
    status_msg="Load input CSV file...",
    success_msg="Loaded input CSV file.",
)
@check_input_file_exists
@raise_if_args_none(message="Header rows must be specified", level="warning")
def load_csv_table(input_path: Path, header_row: List[int]) -> pd.DataFrame:
    """
    Load a CSV table from `input_path` into a `pd.DataFrame` where the list `header`
    defines which rows make up the column names.
    """
    return pd.read_csv(input_path, header=header_row)
