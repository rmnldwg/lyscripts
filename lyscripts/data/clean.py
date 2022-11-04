"""
Transform the enhanced lyDATA CSV files into a format that can be used by the
lymph model using this package's utilities.
"""
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from lyscripts.utils import cli_load_yaml_params, report

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


def lyprox_to_lymph(
    data: pd.DataFrame,
    method: str = "unilateral",
    convert_t_stage: Optional[Dict[int, Any]] = None
) -> pd.DataFrame:
    """
    Convert [LyProX](https://lyprox.org) `data` into `pd.DataFrame` that the
    [lymph](https://github.com/rmnldwg/lymph) package can use for sampling.

    This conversion can be done according to a given `method` out of three that
    specifies `"unilateral"`, `"bilateral"` or `"midline"`, depending on the class
    that is later supposed to load the data.

    `convert_t_stage` is a dictionary that maps from the range of T-stages in the
    LyProX `data` (keys) to T-stages that the lymph library is supposed to work with
    (values). It could look like this (which is also the default):

    ```python
    convert_t_stage = {
        0: 'early',
        1: 'early',
        2: 'early',
        3: 'late',
        4: 'late'
    }
    ```
    """
    t_stage_data = data[("tumor", "1", "t_stage")]
    midline_extension_data = data[("tumor", "1", "extension")]

    # Extract modalities
    top_lvl_headers = set(data.columns.get_level_values(0))
    modalities = [h for h in top_lvl_headers if h not in ["tumor", "patient"]]
    diagnostic_data = data[modalities].drop(columns=["date"], level=2)

    if convert_t_stage is None:
        convert_t_stage = {
            0: "early",
            1: "early",
            2: "early",
            3: "late",
            4: "late"
        }
    diagnostic_data[("info", "tumor", "t_stage")] = [
        convert_t_stage[t] for t in t_stage_data.values
    ]

    if method == "midline":
        diagnostic_data[("info", "tumor", "midline_extension")] = midline_extension_data
    elif method == "unilateral":
        diagnostic_data = diagnostic_data.drop(columns=["contra"], level=1)
        diagnostic_data.columns = diagnostic_data.columns.droplevel(1)

    return diagnostic_data


def main(args: argparse.Namespace):
    """
    When running `lyscripts clean --help` the output is the following:

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
    params = cli_load_yaml_params(args.params)

    with report.status("Reading in CSV file..."):
        enhanced_df = pd.read_csv(args.input, header=[0,1,2])
        method = {
            "Unilateral": "unilateral",
            "Bilateral": "midline",
            "MidlineBilateral": "midline",
        }[params["model"]["class"]]
        cleaned_df = lyprox_to_lymph(enhanced_df, method=method)
        report.success(f"Read in CSV file from {args.input}")

    with report.status("Saving cleaned dataset..."):
        cleaned_df.to_csv(args.output, index=None)
        report.success(f"Saved cleaned dataset to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
