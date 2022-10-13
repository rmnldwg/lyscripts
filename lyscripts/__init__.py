"""
.. include:: ../README.md
"""
import argparse

from . import (
    clean,
    enhance,
    evaluate,
    generate,
    join,
    plot,
    predict,
    sample,
    split,
    temp_schedule,
)
from ._version import version
from .helpers import clean_docstring, report
from .rich_argparse import RichHelpFormatter

__version__ = version
__description__ = "Package containing scripts used in lynference pipelines"
__author__ = "Roman Ludwig"
__email__ = "roman.ludwig@usz.ch"
__uri__ = "https://github.com/rmnldwg/lyscripts"

# nopycln: file


class RichDefaultHelpFormatter(
    RichHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter
):
    """
    Empty class that combines the functionality of displaying the default value with
    the beauty of the `rich` formatter
    """

def exit(args: argparse.Namespace):
    """Exit the cmd line tool"""
    if args.version:
        report.print("lyscripts ", __version__)
    else:
        report.print("No command chosen. Exiting...")

def main():
    """The main entry point of the CLI."""
    parser = argparse.ArgumentParser(
        prog="lyscripts",
        description=clean_docstring(__doc__),
        formatter_class=RichDefaultHelpFormatter,
    )
    parser.set_defaults(run_main=exit)
    parser.add_argument(
        "-v", "--version", action="store_true",
        help="Display the version of lyscripts"
    )

    subparsers = parser.add_subparsers()

    # the individual scripts add `ArgumentParser` instances and their arguments to
    # this `subparsers` object
    generate._add_parser(subparsers, help_formatter=parser.formatter_class)
    join._add_parser(subparsers, help_formatter=parser.formatter_class)
    enhance._add_parser(subparsers, help_formatter=parser.formatter_class)
    clean._add_parser(subparsers, help_formatter=parser.formatter_class)
    split._add_parser(subparsers, help_formatter=parser.formatter_class)
    sample._add_parser(subparsers, help_formatter=parser.formatter_class)
    evaluate._add_parser(subparsers, help_formatter=parser.formatter_class)
    predict._add_parser(subparsers, help_formatter=parser.formatter_class)
    plot._add_parser(subparsers, help_formatter=parser.formatter_class)
    temp_schedule._add_parser(subparsers, help_formatter=parser.formatter_class)

    args = parser.parse_args()
    args.run_main(args)
