"""Run the plot module as a script."""

import argparse

from lyscripts import RichDefaultHelpFormatter, exit_cli
from lyscripts.plot import corner, histograms, thermo_int


def main(args: argparse.Namespace) -> None:
    """Run the main function of the selected subcommand."""
    parser = argparse.ArgumentParser(
        prog="lyscripts plot",
        description=__doc__,
        formatter_class=RichDefaultHelpFormatter,
    )
    parser.set_defaults(run_main=exit_cli)
    subparsers = parser.add_subparsers()

    # the individual scripts add `ArgumentParser` instances and their arguments to
    # this `subparsers` object
    corner._add_parser(subparsers, help_formatter=parser.formatter_class)
    histograms._add_parser(subparsers, help_formatter=parser.formatter_class)
    thermo_int._add_parser(subparsers, help_formatter=parser.formatter_class)

    args = parser.parse_args()
    args.run_main(args, parser)


if __name__ == "__main__":
    main()
