"""
Utility for performing common tasks w.r.t. the inference and prediction tasks one
can use the `lymph` package for.
"""
import argparse

from . import (
    _exit,
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
from .helpers import clean_docstring
from .rich_argparse import RichHelpFormatter


class RichDefaultHelpFormatter(
    RichHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter
):
    """
    Empty class that combines the functionality of displaying the default value with
    the beauty of the `rich` formatter
    """

# I need another __main__ guard here, because otherwise pdoc tries to run this
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="lyscripts",
        description=clean_docstring(__doc__),
        formatter_class=RichDefaultHelpFormatter,
    )
    parser.set_defaults(run_main=_exit)
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
