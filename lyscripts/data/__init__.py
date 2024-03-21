"""
Provide a range of commands and functions related to managing CSV datasets on patterns
of lymphatic progression.

It helps transform raw CSV data of any form to be converted into our `LyProX`_ format,
which can then be uploaded to the `LyProX`_ online tool for others to inspect the data.

.. _LyProX: https://lyprox.org
"""
import argparse
from pathlib import Path

from lyscripts.data import enhance, filter, generate, join, lyproxify, split


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """Add parser to the ``subparsers`` and then add this module's subcommands."""
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
    filter._add_parser(subparsers, help_formatter=parser.formatter_class)
