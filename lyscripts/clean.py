"""
Transform the enhanced lyDATA CSV files into a format that can be used by the
lymph model using this package's utilities.
"""
import argparse
import warnings
from pathlib import Path

import lymph
import pandas as pd
import yaml

from .helpers import clean_docstring, report

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
        description=clean_docstring(__doc__),
        help=clean_docstring(__doc__),
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """
    Add arguments needed to run this script to a `subparsers` instance
    and run the respective main function when chosen.
    """
    parser.add_argument(
        "input", type=Path,
        help="Path to the enhanced lyDATA CSV file to transform."
    )
    parser.add_argument(
        "output", type=Path,
        help="Path to the cleand CSV file ready for inference."
    )

    parser.add_argument(
        "-p", "--params", default="./params.yaml", type=Path,
        help="Path to the params file to use for the transformation."
    )

    parser.set_defaults(run_main=main)


def main(args: argparse.Namespace):
    """
    When running `python -m lyscripts clean --help` the output is the following:

    ```
    usage: lyscripts clean [-h] [-p PARAMS] input output

    Transform the enhanced lyDATA CSV files into a format that can be used by the lymph
    model using this package's utilities.


    POSITIONAL ARGUMENTS
    input                Path to the enhanced lyDATA CSV file to transform.
    output               Path to the cleand CSV file ready for inference.

    OPTIONAL ARGUMENTS
    -h, --help           show this help message and exit
    -p, --params PARAMS  Path to the params file to use for the transformation.
                        (default: ./params.yaml)
    ```
    """
    with report.status("Read in parameters..."):
        with open(args.params, mode='r') as params_file:
            params = yaml.safe_load(params_file)
        report.success(f"Read in params from {args.params}")

    with report.status("Reading in CSV file..."):
        enhanced_df = pd.read_csv(args.input, header=[0,1,2])
        method = {
            "Unilateral": "unilateral",
            "Bilateral": "midline",
            "MidlineBilateral": "midline",
        }[params["model"]["class"]]
        cleaned_df = lymph.utils.lyprox_to_lymph(enhanced_df, method=method)
        report.success(f"Read in CSV file from {args.input}")

    with report.status("Saving cleaned dataset..."):
        cleaned_df.to_csv(args.output, index=None)
        report.success(f"Saved cleaned dataset to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
