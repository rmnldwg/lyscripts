import argparse

from .. import exit_lyscripts
from ..helpers import clean_docstring
from ..rich_argparse import RichHelpFormatter
from . import __doc__, prevalences, risks

parser = argparse.ArgumentParser(
    prog="lyscripts",
    description=clean_docstring(__doc__),
    formatter_class=RichHelpFormatter,
)
parser.set_defaults(run_main=exit_lyscripts)
subparsers = parser.add_subparsers()

# the individual scripts add `ArgumentParser` instances and their arguments to
# this `subparsers` object
risks.add_parser(subparsers, help_formatter=parser.formatter_class)
prevalences.add_parser(subparsers, help_formatter=parser.formatter_class)

args = parser.parse_args()
args.run_main(args)
