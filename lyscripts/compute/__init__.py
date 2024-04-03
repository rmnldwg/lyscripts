"""
With the commands of this module, a user may compute prior and posterior state
distributions from drawn samples of a model. This can in turn speed up the computation
of risks and prevalences.
"""
import argparse
from pathlib import Path

from lyscripts.compute import posteriors, prevalences, priors, risks


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
    priors._add_parser(subparsers, help_formatter=parser.formatter_class)
    posteriors._add_parser(subparsers, help_formatter=parser.formatter_class)
    prevalences._add_parser(subparsers, help_formatter=parser.formatter_class)
    risks._add_parser(subparsers, help_formatter=parser.formatter_class)
