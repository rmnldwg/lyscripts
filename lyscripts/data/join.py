"""
Join datasets from different sources (but of the same format) into one.
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd

from lyscripts.data.utils import load_csv_table, save_table_to_csv
from lyscripts.utils import report

warnings.simplefilter(action="ignore", category=FutureWarning)



def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """
    Add an `ArgumentParser` to the subparsers action.
    """
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """
    Add arguments needed to run this script to a `subparsers` instance
    and run the respective main function when chosen.
    """
    parser.add_argument(
        "-i", "--inputs", nargs='+', type=Path, required=True,
        help="List of paths to inference-ready CSV datasets to concatenate."
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Location to store the concatenated CSV file."
    )

    parser.set_defaults(run_main=main)


def main(args: argparse.Namespace):
    """
    This program simply loops over the provided CSV files, reading in and appending
    them to a concatenated `pd.DataFrame` one by one, hoping that they are all provided
    in the same format.

    In the end, the joined `pd.DataFrame` is stored at the desired location.

    It's command help when running `lyscripts join --help` shows

    ```
    USAGE: lyscripts data join [-h] -i INPUTS [INPUTS ...] -o OUTPUT

    Join datasets from different sources (but of the same format) into one.

    OPTIONAL ARGUMENTS:
      -h, --help            show this help message and exit
      -i, --inputs INPUTS [INPUTS ...]
                            List of paths to inference-ready CSV datasets to
                            concatenate. (default: None)
      -o, --output OUTPUT   Location to store the concatenated CSV file. (default:
                            None)
    ```
    """
    with report.status("Reading & concatenating CSV files..."):
        concatenated_table = pd.DataFrame()
        for input_path in args.inputs:
            input_table = load_csv_table(input_path, header_row=[0,1,2])
            concatenated_table = pd.concat(
                [concatenated_table, input_table],
                ignore_index=True
            )
            report.print(f"+ concatenated data from {input_path}")
        report.success(f"Read & concatenated all {len(args.inputs)} CSV files")

    save_table_to_csv(args.output, concatenated_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
