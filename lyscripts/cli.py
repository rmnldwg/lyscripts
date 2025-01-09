"""Utilities for configuring and running the CLI app."""

import argparse
from collections.abc import Callable

import rich
import rich.text
from pydantic_settings import BaseSettings, CliApp, CliSettingsSource
from rich_argparse import RichHelpFormatter


class RichDefaultHelpFormatter(
    RichHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    """Combine formatter that shows defaults with `rich`_ formatting.

    .. _rich: https://rich.readthedocs.io/en/stable/introduction.html
    """

    def _rich_fill_text(
        self,
        text: rich.text.Text,
        width: int,
        indent: rich.text.Text,
    ) -> rich.text.Text:
        text_cls = type(text)
        if text[0] == text_cls("\n"):
            text = text[1:]

        paragraphs = text.split(separator="\n\n")
        text_lines = rich.containers.Lines()
        for par in paragraphs:
            no_newline_par = text_cls(" ").join(line for line in par.split())
            wrapped_par = no_newline_par.wrap(self.console, width)

            for line in wrapped_par:
                text_lines.append(line)

            text_lines.append(text_cls("\n"))

        return text_cls("\n").join(indent + line for line in text_lines) + "\n\n"


RichDefaultHelpFormatter.styles["argparse.syntax"] = "red"
RichDefaultHelpFormatter.styles["argparse.formula"] = "green"
RichDefaultHelpFormatter.highlights.append(r"\$(?P<formula>[^$]*)\$")
RichDefaultHelpFormatter.styles["argparse.bold"] = "bold"
RichDefaultHelpFormatter.highlights.append(r"\*(?P<bold>[^*]*)\*")
RichDefaultHelpFormatter.styles["argparse.italic"] = "italic"
RichDefaultHelpFormatter.highlights.append(r"_(?P<italic>[^_]*)_")


def _assemble_main(
    settings_cls: type[BaseSettings],
    prog_name: str,
) -> Callable[[], None]:
    """Assemble the main function for a CLI app.

    This is mostly to save some boilerplate code when creating the main functions of
    all subcommands. And those, in turn, are only implemented to allow running the
    subcommands as scripts.
    """

    def main() -> None:
        """Start the main CLI app."""
        cli_settings_source = CliSettingsSource(
            settings_cls=settings_cls,
            cli_prog_name=prog_name,
            cli_kebab_case=True,
            cli_use_class_docs_for_groups=True,
            formatter_class=RichDefaultHelpFormatter,
        )
        CliApp.run(settings_cls, cli_settings_source=cli_settings_source)

    return main
