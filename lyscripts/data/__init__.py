"""
Provide a range of commands related to datasets on patterns of lymphatic progression.
"""
import argparse
from pathlib import Path

from . import clean, enhance, generate, join, split


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
    clean._add_parser(subparsers, help_formatter=parser.formatter_class)
    enhance._add_parser(subparsers, help_formatter=parser.formatter_class)
    generate._add_parser(subparsers, help_formatter=parser.formatter_class)
    join._add_parser(subparsers, help_formatter=parser.formatter_class)
    split._add_parser(subparsers, help_formatter=parser.formatter_class)
