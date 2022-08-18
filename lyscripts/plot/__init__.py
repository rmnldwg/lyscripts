"""
Provide various plotting utilities for displaying results of e.g. the inference
or prediction process.
"""
import argparse
from pathlib import Path

from ..helpers import clean_docstring
from . import __doc__, corner, histograms


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """
    Add an `ArgumentParser` to the subparsers action and then add more subparsers.
    """
    parser = subparsers.add_parser(
        Path(__file__).parent.name,
        description=clean_docstring(__doc__),
        help=clean_docstring(__doc__),
        formatter_class=help_formatter,
    )
    subparsers = parser.add_subparsers()
    corner._add_parser(subparsers, help_formatter=parser.formatter_class)
    histograms._add_parser(subparsers, help_formatter=parser.formatter_class)