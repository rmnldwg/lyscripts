"""Commands and functions for managing CSV data on patterns of lymphatic progression.

This contains helpful CLI commands that allow building quick and reproducible workflows
even when using language-agnostic tools like `Make`_ or `DVC`_.

Most of these commands can load `LyProX`_ style data from CSV files, but also from
the installed datasets provided by the `lydata`_ package and directly from the
associated `GitHub repository`_.

.. _Make: https://www.gnu.org/software/make/
.. _DVC: https://dvc.org
.. _LyProX: https://lyprox.org
.. _lydata: https://lydata.readthedocs.io
.. _GitHub repository: https://github.com/rmnldwg/lydata
"""

from pydantic_settings import BaseSettings, CliApp, CliSubCommand

from lyscripts.data import (  # noqa: F401
    enhance,
    fetch,
    generate,
    join,
    lyproxify,
    split,
)

# Avoid conflict with built-in `filter` function
from lyscripts.data import filter as filter_


class DataCLI(BaseSettings):
    """Work with lymphatic progression data through this CLI."""

    lyproxify: CliSubCommand[lyproxify.LyproxifyCLI]
    join: CliSubCommand[join.JoinCLI]
    split: CliSubCommand[split.SplitCLI]
    fetch: CliSubCommand[fetch.FetchCLI]
    filter: CliSubCommand[filter_.FilterCLI]
    enhance: CliSubCommand[enhance.EnhanceCLI]
    generate: CliSubCommand[generate.GenerateCLI]

    def cli_cmd(self) -> None:
        """Run one of the ``data`` subcommands."""
        CliApp.run_subcommand(self)
