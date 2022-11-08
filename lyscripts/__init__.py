"""
.. include:: ../README.md
"""
import argparse

from rich.containers import Lines
from rich.text import Text
from rich_argparse import RichHelpFormatter

from lyscripts.utils import report

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
    utils,
)
from ._version import version

__version__ = version
__description__ = "Package containing scripts used in lynference pipelines"
__author__ = "Roman Ludwig"
__email__ = "roman.ludwig@usz.ch"
__uri__ = "https://github.com/rmnldwg/lyscripts"

# nopycln: file


class RichDefaultHelpFormatter(
    RichHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    """
    Empty class that combines the functionality of displaying the default value with
    the beauty of the `rich` formatter
    """
    def _rich_fill_text(self, text: Text, width: int, indent: Text) -> Text:
        text_cls = type(text)
        if text[0] == text_cls("\n"):
            text = text[1:]

        paragraphs = text.split(separator="\n\n")
        text_lines = Lines()
        for par in paragraphs:
            no_newline_par = text_cls(" ").join(line for line in par.split())
            wrapped_par = no_newline_par.wrap(self.console, width)

            for line in wrapped_par:
                text_lines.append(line)

            text_lines.append(text_cls("\n"))

        return text_cls("\n").join(indent + line for line in text_lines) + "\n\n"

RichDefaultHelpFormatter.styles["argparse.syntax"] = "red"
RichDefaultHelpFormatter.styles["argparse.formula"] = "green"
RichDefaultHelpFormatter.highlights.append(
    r"\$(?P<formula>[^$]*)\$"
)
RichDefaultHelpFormatter.styles["argparse.bold"] = "bold"
RichDefaultHelpFormatter.highlights.append(
    r"\*(?P<bold>[^*]*)\*"
)
RichDefaultHelpFormatter.styles["argparse.italic"] = "italic"
RichDefaultHelpFormatter.highlights.append(
    r"_(?P<italic>[^_]*)_"
)

def exit_cli(args: argparse.Namespace):
    """Exit the cmd line tool"""
    if args.version:
        report.print("lyscripts ", __version__)
    else:
        report.print("No command chosen. Exiting...")

def main():
    """The main entry point of the CLI."""
    parser = argparse.ArgumentParser(
        prog="lyscripts",
        description=__doc__,
        formatter_class=RichDefaultHelpFormatter,
    )
    parser.set_defaults(run_main=exit_cli)
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
