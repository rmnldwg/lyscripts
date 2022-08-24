import argparse

from .. import _exit
from ..helpers import clean_docstring
from ..rich_argparse import RichHelpFormatter
from . import __doc__, corner, histograms, thermo_int

# I need another __main__ guard here, because otherwise pdoc tries to run this
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="lyscripts",
        description=clean_docstring(__doc__),
        formatter_class=RichHelpFormatter,
    )
    parser.set_defaults(run_main=_exit)
    subparsers = parser.add_subparsers()

    # the individual scripts add `ArgumentParser` instances and their arguments to
    # this `subparsers` object
    corner._add_parser(subparsers, help_formatter=parser.formatter_class)
    histograms._add_parser(subparsers, help_formatter=parser.formatter_class)
    thermo_int._add_parser(subparsers, help_formatter=parser.formatter_class)

    args = parser.parse_args()
    args.run_main(args)
