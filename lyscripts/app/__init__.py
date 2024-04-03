"""
Module containing scripts to run different `streamlit`_ applications.

.. _streamlit: https://streamlit.io/
"""
import argparse
from pathlib import Path

from lyscripts.app import prevalence


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """Add parser for the :py:mod:`.app` module to the given ``subparsers``.

    This will also add any subcommands within this module to the given
    ``subparsers``.
    """
    parser = subparsers.add_parser(
        Path(__file__).parent.name,
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    subparsers = parser.add_subparsers()
    prevalence._add_parser(subparsers, help_formatter=parser.formatter_class)
