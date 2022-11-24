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
from types import ModuleType

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
        "-i", "--input", type=Path, required=True,
        help="Location of raw CSV data."
    )
    parser.add_argument(
        "-r", "--header-rows", nargs="+", default=[0],
        help="List with header row indices of raw file."
    )
    parser.add_argument(
        "-m", "--mapping", type=Path, required=True,
        help=(
            "Location of the Python file that contains column mapping instructions. "
            "This must contain a dictionary with the name 'column_map'."
        )
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Location to store the lyproxified CSV file."
    )

    parser.set_defaults(run_main=main)


def transform_to_lyprox(raw: pd.DataFrame, mapping: ModuleType) -> pd.DataFrame:
    """
    Transform any `raw` data frame into a table that can be uploaded directly to
    [LyProX](https://lyprox.org). To do so, it imports instructions form the `mapping` module. In it, a
    dictionary `column_map` must be provided with a particular structure:

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

    The example function `compute_age_from_raw` should also be defined in the
    `mapping` module.
    """
    multi_idx = pd.MultiIndex.from_tuples(mapping.column_map.keys())
    processed = pd.DataFrame(columns=multi_idx)

    for multi_idx_col, instruction in mapping.column_map.items():
        if instruction != "":
            if "default" in instruction:
                processed[multi_idx_col] = [instruction["default"]] * len(raw)
            else:
                cols = instruction.get("columns", [])
                kwargs = instruction.get("kwrags", {})
                func = instruction.get("func", lambda x, *_a, **_kw: x)
                processed[multi_idx_col] = [
                        func(*values, **kwargs) for values in raw[cols].values
                    ]
    return processed


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

    with report.status("Import mapping instructions..."):
        spec = importlib.util.spec_from_file_location("map_module", args.mapping)
        mapping = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mapping)
        report.success(f"Imported mapping instructions from {args.mapping}")

    with report.status("Transform columns..."):
        processed = transform_to_lyprox(raw, mapping)
        report.success(f"Transformed {len(mapping.column_map)} columns.")

    if ("tumor", "1", "side") in processed.columns:
        with report.status("Tranform 'left'/'right' to 'ipsi'/'contra'..."):
            len_before = len(processed)
            left_data = processed.loc[
                processed["tumor", "1", "side"] != "right"
            ]
            right_data = processed.loc[
                processed["tumor", "1", "side"] == "right"
            ]

            left_data = left_data.rename(columns={"left": "ipsi"}, level=1)
            left_data = left_data.rename(columns={"right": "contra"}, level=1)
            right_data = right_data.rename(columns={"left": "contra"}, level=1)
            right_data = right_data.rename(columns={"right": "ipsi"}, level=1)

            processed = pd.concat(
                [left_data, right_data], ignore_index=True
            )
            assert len_before == len(processed)
            report.success("Transformed 'left' & 'right' to 'ipsi' & 'contra'")

    save_table_to_csv(args.output, processed)
