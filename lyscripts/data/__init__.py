"""Commands and functions for managing CSV data on patterns of lymphatic progression.

This contains helpful CLI commands that allow building quick and reproducible workflows
even when using language-agnostic tools like `Make`_ or `DVC`_.

Most of these commands can load `LyProX`_ style data from CSV files, but also from
the installed datasets provided by the `lydata`_ package and directly from the
associated GitHub repository.

.. _Make: https://www.gnu.org/software/make/
.. _DVC: https://dvc.org
.. _LyProX: https://lyprox.org
.. _lydata: https://lydata.readthedocs.io
"""

from pydantic_settings import BaseSettings, CliApp, CliSubCommand

from lyscripts.data import (  # noqa: F401
    enhance,
    filter,
    generate,
    join,
    lyproxify,
    split,
)


class DataCLI(BaseSettings):
    """Work with lymphatic progression data through this CLI."""

    enhance: CliSubCommand[enhance.EnhanceCLI]
    filter: CliSubCommand[filter.FilterCLI]
    generate: CliSubCommand[generate.GenerateCLI]
    lyproxify: CliSubCommand[lyproxify.LyproxifyCLI]
    join: CliSubCommand[join.JoinCLI]
    split: CliSubCommand[split.SplitCLI]

    def cli_cmd(self) -> None:
        """Start the ``data`` subcommand."""
        CliApp.run_subcommand(self)
