"""
Given samples drawn during an MCMC round, precompute the state distribution for each
sample. This may then later on be used to compute risks and prevalences more quickly.
"""
import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """Add an `ArgumentParser` to the subparsers action."""
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """Add arguments needed to run this script to a `subparsers` instance."""
    parser.set_defaults(run_main=main)


def main() -> None:
    """Compute posteriors from priors or drawn samples."""
    pass
