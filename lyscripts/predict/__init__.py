"""
This module provides functions and scripts to predict the risk of hidden involvement,
given observed diagnoses, and prevalences of patterns for diagnostic modalities.
"""
import argparse
from pathlib import Path

from ..helpers import clean_docstring
from . import __doc__, prevalences, risks


def add_parser(
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
    prevalences.add_parser(subparsers, help_formatter=parser.formatter_class)
    risks.add_parser(subparsers, help_formatter=parser.formatter_class)
