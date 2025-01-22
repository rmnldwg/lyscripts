"""Utilities for configuring and running CLIs app.

In this module, we define and configure a :py:class:`RichDefaultHelpFormatter` that
nicely displays the CLI's ``--help`` text. We also provide a function to
:py:func:`assemble a main function <assemble_main>` for the different CLI apps to save
some boilerplate code. Lastly, we have two functions related to the `loguru`_ setup.

.. _loguru: https://loguru.readthedocs.io/en/stable
"""

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
    """Assemble a ``main()`` function for a CLI app.

    It creates a :py:class:`~pydantic_settings.CliSettingsSource` object with the
    provided ``settings_cls`` and ``prog_name``. Then, it fills in some default
    settings for the CLI configuration and runs the CLI app.

    Assembling a ``main()`` function for all subcommands like this saves some
    boilerplate code.
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
