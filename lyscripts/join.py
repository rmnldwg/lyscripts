"""
Join datasets from different sources (but of the same format) into one.
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd

from .helpers import clean_docstring, report

warnings.simplefilter(action="ignore", category=FutureWarning)



def add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """
    Add an `ArgumentParser` to the subparsers action.
    """
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=clean_docstring(__doc__),
        help=clean_docstring(__doc__),
        formatter_class=help_formatter,
    )
    add_arguments(parser)


def add_arguments(parser: argparse.ArgumentParser):
    """
    Add arguments needed to run this script to a `subparsers` instance
    and run the respective main function when chosen.
    """
    parser.add_argument(
        "inputs", nargs='+', type=Path,
        help="List of paths to inference-ready CSV datasets to concatente."
    )
    parser.add_argument(
        "output", type=Path,
        help="Location to store the concatenated CSV file."
    )

    parser.set_defaults(run_main=main)


def main(args: argparse.Namespace):
    """
    Run main program with `args` parsed by argparse.
    """
    with report.status("Reading & concatenating CSV files..."):
        concatenated_df = pd.DataFrame()
        for input in args.inputs:
            df = pd.read_csv(input, header=[0,1,2])
            concatenated_df = pd.concat(
                [concatenated_df, df],
                ignore_index=True
            )
            report.print(f"+ concatenated data from {input}")
        report.success(f"Read & concatenated all {len(args.inputs)} CSV files")

    with report.status("Saving concatenated dataset..."):
        # Make sure the output directory exists
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Write the concatenated dataset to disk
        concatenated_df.to_csv(args.output, index=None)
        report.success(f"Saved concatenated dataset to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
