"""
Join datasets from different sources (but of the same format) into one.
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import logging
import warnings
from pathlib import Path

import pandas as pd

from lyscripts.data.utils import save_table_to_csv
from lyscripts.utils import load_patient_data

warnings.simplefilter(action="ignore", category=FutureWarning)


logger = logging.getLogger(__name__)


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """Add an ``ArgumentParser`` to the subparsers action."""
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """Add arguments to the parser."""
    parser.add_argument(
        "-i", "--inputs", nargs='+', type=Path, required=True,
        help="List of paths to inference-ready CSV datasets to concatenate."
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Location to store the concatenated CSV file."
    )

    parser.set_defaults(run_main=main)


def load_and_join_tables(input_paths: list[Path]):
    """Load and concatenate CSV files from the given ``input_paths``."""
    concatenated_table = pd.DataFrame()
    for path in input_paths:
        input_table = load_patient_data(path).convert_dtypes()
        concatenated_table = pd.concat(
            [concatenated_table, input_table],
            axis="index",
            ignore_index=True
        )
        logger.info(f"+ concatenated data from {path}")
    return concatenated_table


def main(args: argparse.Namespace):
    """Join datasets from different sources into one."""
    concatenated_table = load_and_join_tables(args.inputs)
    logger.info(f"Read & concatenated all {len(args.inputs)} CSV files")
    save_table_to_csv(args.output, concatenated_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
