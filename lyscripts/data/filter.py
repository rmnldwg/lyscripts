"""
Filter a datset according to some common criteria, like tumor location, subsite,
T-category, etc.
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import logging
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from lyscripts.data.utils import save_table_to_csv
from lyscripts.decorators import log_state
from lyscripts.utils import load_patient_data

warnings.simplefilter(action="ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


FILTER_TO_COLUMN = {
    "locations": ("tumor", "1", "location"),
    "subsites": ("tumor", "1", "subsite"),
    "t_categories": ("tumor", "1", "t_stage"),
}


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
        "input", type=Path,
        help="The path to the full dataset to split."
    )
    parser.add_argument(
        "output", type=Path,
        help="Folder to store the split CSV files in."
    )

    for prefix in ["include", "exclude"]:
        for filter_by in ["locations", "subsites", "t_categories"]:
            parser.add_argument(
                f"--{prefix}-{filter_by}", default=None, type=str, nargs="+",
                help=f"If provided, {prefix} patients with the given tumor {filter_by}."
            )

    parser.set_defaults(run_main=main)


@log_state()
def filter_patients(
    data: pd.DataFrame,
    by: tuple[str, str, str],
    values: Iterable[Any],
    method: str,
    match: str = "isin",
) -> pd.DataFrame:
    """Filter patient data.

    Filter ``by`` the given column. Rows are in- or excluded (depending on the chosen
    ``method``) if their corresponding column value is in the ``values`` iterable.

    The ``match`` string may be any method of a pandas ``Series`` object that returns a
    boolean series, e.g. "isin" or "contains".
    """
    try:
        match_func = getattr(data[by], match)
        match_idx = match_func(values)
    except AttributeError:
        match_func = getattr(data[by].str, match)
        match_idx = False
        for value in values:
            match_idx = match_func(value) | match_idx

    return_idx = match_idx if method == "include" else ~match_idx
    return data[return_idx]


def sanitize(value: str) -> int | str:
    """Sanitize a value for use in a filter."""
    try:
        return int(value)
    except ValueError:
        return value


def main(args: argparse.Namespace):
    """Filter a dataset according to the given criteria."""
    table = load_patient_data(args.input)

    for prefix in ["include", "exclude"]:
        for filter_by in ["locations", "subsites", "t_categories"]:
            filter_values = getattr(args, f"{prefix}_{filter_by}")

            if filter_values is None:
                continue

            sanitized_filter_values = [sanitize(value) for value in filter_values]

            table = filter_patients(
                data=table,
                by=FILTER_TO_COLUMN[filter_by],
                values=sanitized_filter_values,
                method=prefix,
                match="contains" if filter_by == "subsites" else "isin",
            )

    save_table_to_csv(args.output, table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
