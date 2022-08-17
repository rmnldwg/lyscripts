"""
Utility for performing common tasks w.r.t. the inference and prediction tasks one
can use the `lymph` package for.
"""
import argparse

from . import (
    clean,
    evaluate,
    exit_lyscripts,
    generate,
    join,
    predict,
    sample,
    split,
    temp_schedule,
)
from .helpers import clean_docstring
from .rich_argparse import RichHelpFormatter

parser = argparse.ArgumentParser(
    prog="lyscripts",
    description=clean_docstring(__doc__),
    formatter_class=RichHelpFormatter,
)
parser.set_defaults(run_main=exit_lyscripts)
parser.add_argument(
    "-v", "--version", action="store_true",
    help="Display the version of lyscripts"
)

subparsers = parser.add_subparsers()

# the individual scripts add `ArgumentParser` instances and their arguments to
# this `subparsers` object
generate.add_parser(subparsers, help_formatter=parser.formatter_class)
join.add_parser(subparsers, help_formatter=parser.formatter_class)
clean.add_parser(subparsers, help_formatter=parser.formatter_class)
split.add_parser(subparsers, help_formatter=parser.formatter_class)
sample.add_parser(subparsers, help_formatter=parser.formatter_class)
evaluate.add_parser(subparsers, help_formatter=parser.formatter_class)
predict.add_parser(subparsers, help_formatter=parser.formatter_class)
temp_schedule.add_parser(subparsers, help_formatter=parser.formatter_class)

args = parser.parse_args()
args.run_main(args)
