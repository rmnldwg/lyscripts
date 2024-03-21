"""
Provide various plotting utilities for displaying results of e.g. the inference
or prediction process. At the moment, three subcommands are grouped under
:py:mod:`.plot`.
"""
import argparse
from pathlib import Path

from lyscripts.plot import corner, histograms, thermo_int


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """Add an ``ArgumentParser`` to the subparsers and then add more subparsers."""
    parser = subparsers.add_parser(
        Path(__file__).parent.name,
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    subparsers = parser.add_subparsers()
    corner._add_parser(subparsers, help_formatter=parser.formatter_class)
    histograms._add_parser(subparsers, help_formatter=parser.formatter_class)
    thermo_int._add_parser(subparsers, help_formatter=parser.formatter_class)
