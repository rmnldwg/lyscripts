"""
Utilities related to the commands for data cleaning and processing.
"""
from pathlib import Path
from typing import List

import pandas as pd
from pandas.errors import EmptyDataError

from lyscripts.utils import (
    check_input_file_exists,
    check_output_dir_exists,
    report,
    report_func_state,
)


@report_func_state(
    status_msg="Save processed CSV file...",
    success_msg="Saved processed CSV file.",
)
@check_output_dir_exists
def save_table_to_csv(output_path: Path, table: pd.DataFrame):
    """Save a `pd.DataFrame` to `output_path`."""
    table.to_csv(output_path, index=None)


@report_func_state(
    status_msg="Load input CSV file...",
    success_msg="Loaded input CSV file.",
    actions={
        FileNotFoundError: (True, report.exception, "Input CSV file not found, stopping."),
        UnicodeDecodeError: (True, report.exception, "Input is not a CSV file, stopping."),
        EmptyDataError: (True, report.exception, "CSV input seems to be empty, stopping."),
    }
)
@check_input_file_exists
def load_csv_table(input_path: Path, header=List[int]) -> pd.DataFrame:
    """
    Load a CSV table from `input_path` into a `pd.DataFrame` where the list `header`
    defines which rows make up the column names.
    """
    return pd.read_csv(input_path, header=header)
