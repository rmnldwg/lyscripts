"""Utilities related to the commands for data cleaning and processing."""

from pathlib import Path

import pandas as pd
from loguru import logger

from lyscripts.decorators import check_output_dir_exists


@check_output_dir_exists
def save_table_to_csv(file_path: Path, table: pd.DataFrame):
    """Save a ``table`` to ``output_path``."""
    logger.info(f"Saving table ({table.shape = }) to {file_path}")
    table.to_csv(file_path, index=None)
