"""
Provide a range of commands related to datasets on patterns of lymphatic progression.
Currently, the following modules provide additional commands:

1. `lyscripts.data.enhance`, a module for computing consensus diagnoses and to ensure
that super- and sublevels are consistently reported.
2. The module `lyscripts.data.generate` for creating synthetic datasets with certain
characteristics.
3. Submodule `lyscripts.data.join` to concatenate two datasets, e.g. from different
institutions.
4. `lyscripts.data.split`, a module with which datasets may be split into random sets
of patient data. The split data may then be used e.g. for cross-validation.
"""
import argparse
from pathlib import Path

from . import enhance, generate, join, lyproxify, split


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """
    Add an `ArgumentParser` to the subparsers action and then add more subparsers.
    """
    parser = subparsers.add_parser(
        Path(__file__).parent.name,
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    subparsers = parser.add_subparsers()
    enhance._add_parser(subparsers, help_formatter=parser.formatter_class)
    generate._add_parser(subparsers, help_formatter=parser.formatter_class)
    join._add_parser(subparsers, help_formatter=parser.formatter_class)
    lyproxify._add_parser(subparsers, help_formatter=parser.formatter_class)
    split._add_parser(subparsers, help_formatter=parser.formatter_class)
