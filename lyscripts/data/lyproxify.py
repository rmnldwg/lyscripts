"""
Consumes raw data and transforms it into a CSV of the format that
LyProX can understand.

To do so, it needs a dictionary that defines a mapping from raw columns to the LyProX
style data format. See the documentation of the `transform_to_lyprox` function for
more information.
"""
import argparse
import importlib.util
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from lyscripts.data.utils import load_csv_table, save_table_to_csv
from lyscripts.utils import raise_if_args_none, report, report_state

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
        "-i", "--input", type=Path, required=True,
        help="Location of raw CSV data."
    )
    parser.add_argument(
        "-r", "--header-rows", nargs="+", default=[0], type=int,
        help="List with header row indices of raw file."
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Location to store the lyproxified CSV file."
    )
    parser.add_argument(
        "-m", "--mapping", type=Path, required=True,
        help=(
            "Location of the Python file that contains column mapping instructions. "
            "This must contain a dictionary with the name 'column_map'."
        )
    )
    parser.add_argument(
        "--drop-rows", nargs="+", type=int, default=[],
        help=(
            "Delete rows of specified indices. Counting of rows start at 0 _after_ "
            "the `header-rows`."
        )
    )
    parser.add_argument(
        "--drop-cols", nargs="+", type=int, default=[],
        help="Delete columns of specified indices.",
    )

    parser.set_defaults(run_main=main)


class ParsingError(Exception):
    """Error while parsing the CSV file."""


@report_state(
    status_msg="Transform raw data to LyProX style table...",
    success_msg="Transformed raw data to LyProX style table.",
    stop_on_exc=True
)
@raise_if_args_none(
    message="Must provide raw data and mapping instruction module",
    level="warning",
)
def transform_to_lyprox(
    raw: pd.DataFrame,
    column_map: Dict[Tuple, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Transform any `raw` data frame into a table that can be uploaded directly to
    [LyProX](https://lyprox.org). To do so, it uses instructions in the `colum_map`
    dictionary, that needs to have a particular structure:

    For each column in the final 'lyproxified' `pd.DataFrame`, one entry must exist in
    the `column_map` dctionary. E.g., for the column corresponding to a patient's age,
    the dictionary should contain a key-value pari of this shape:

    ```python
    column_map = {
        ("patient", "#", "age"): {
            "func": compute_age_from_raw,
            "kwargs": {"randomize": False},
            "columns": ["birthday", "date of diagnosis"]
        },
    }
    ```
    """
    multi_idx = pd.MultiIndex.from_tuples(column_map.keys())
    processed = pd.DataFrame(columns=multi_idx)

    for multi_idx_col, instruction in column_map.items():
        if instruction != "":
            if "default" in instruction:
                processed[multi_idx_col] = [instruction["default"]] * len(raw)
            else:
                cols = instruction.get("columns", [])
                kwargs = instruction.get("kwargs", {})
                func = instruction.get("func", lambda x, *_a, **_kw: x)

                try:
                    processed[multi_idx_col] = [
                        func(*vals, **kwargs) for vals in raw[cols].values
                    ]
                except Exception as exc:
                    raise ParsingError(
                        f"Exception encountered while parsing column {multi_idx_col}"
                    ) from exc
    return processed


@report_state(
    status_msg="Transform absolute side reporting to tumor-relative...",
    success_msg="Transformed absolute side reporting to tumor-relative.",
    stop_on_exc=True,
)
@raise_if_args_none(message="Missing data table", level="warning")
def leftright_to_ipsicontra(data: pd.DataFrame):
    """
    Transform reporting of LNL involvement by absolute side (right & left) to a
    reporting relative to the tumor (ipsi- & contralateral). The table `data` should
    already be in the format LyProX requires, except for the side-reporting of LNL
    involvement.
    """
    len_before = len(data)
    left_data = data.loc[
                data["tumor", "1", "side"] != "right"
            ]
    right_data = data.loc[
                data["tumor", "1", "side"] == "right"
            ]

    left_data = left_data.rename(columns={"left": "ipsi"}, level=1)
    left_data = left_data.rename(columns={"right": "contra"}, level=1)
    right_data = right_data.rename(columns={"left": "contra"}, level=1)
    right_data = right_data.rename(columns={"right": "ipsi"}, level=1)

    data = pd.concat(
                [left_data, right_data], ignore_index=True
            )
    assert len_before == len(data), "Number of patients changed"
    return data


@report_state(
    status_msg="Exclude patients based on provided mapping module...",
    success_msg="Excluded patients based on provided mapping module.",
    stop_on_exc=True,
)
@raise_if_args_none(message="Raw data and mapping module needed", level="warning")
def exclude_patients(raw: pd.DataFrame, exclude: List[Tuple[str, Any]]):
    """
    Exclude patients in the `raw` data based on a list of what to `exclude`. This
    list contains tuples `(column: str, condition: Any)`. This function will then
    axclude any patients from the cohort where `raw[column] == condition`.
    """
    for column, condition in exclude:
        exclude = raw[column] == condition
        raw = raw.loc[~exclude]
    return raw


def main(args: argparse.Namespace):
    """
    The main entry point for the CLI of this command. Upon requesting `lyscripts
    data lyproxify --help`, this is the help output:

    ```
    USAGE: lyscripts data lyproxify [-h] -i INPUT [-r HEADER_ROWS [HEADER_ROWS ...]]
                                    -m MAPPING -o OUTPUT

    Consumes raw data and transforms it into a CSV of the format that LyProX can
    understand.

    To do so, it needs a dictionary that defines a mapping from raw columns to the
    LyProX style data format. See the documentation of the `transform_to_lyprox`
    function for more information.

    OPTIONAL ARGUMENTS:
      -h, --help            show this help message and exit
      -i, --input INPUT     Location of raw CSV data. (default: None)
      -r, --header-rows HEADER_ROWS [HEADER_ROWS ...]
                            List with header row indices of raw file. (default: [0])
      -m, --mapping MAPPING
                            Location of the Python file that contains column mapping
                            instructions. This must contain a dictionary with the name
                            'column_map'. (default: None)
      -o, --output OUTPUT   Location to store the lyproxified CSV file. (default:
                            None)
    ```
    """
    raw = load_csv_table(args.input, header_row=args.header_rows)
    cols_to_drop = raw.columns[args.drop_cols]
    trimmed = raw.drop(cols_to_drop, axis="columns")
    trimmed = trimmed.drop(index=args.drop_rows)

    with report.status("Import mapping instructions..."):
        spec = importlib.util.spec_from_file_location("map_module", args.mapping)
        mapping = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mapping)
        report.success(f"Imported mapping instructions from {args.mapping}")

    reduced = exclude_patients(trimmed, mapping.exclude)
    processed = transform_to_lyprox(reduced, mapping.column_map)

    if ("tumor", "1", "side") in processed.columns:
        processed = leftright_to_ipsicontra(processed)

    save_table_to_csv(args.output, processed)
