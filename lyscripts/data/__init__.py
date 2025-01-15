"""Commands and functions for managing CSV data on patterns of lymphatic progression.

It helps transform raw CSV data of any form to be converted into our `LyProX`_ format,
which can then be uploaded to the `LyProX`_ online tool for others to inspect the data.

.. _LyProX: https://lyprox.org
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
    join: CliSubCommand[join.JoinCLI]
    split: CliSubCommand[split.SplitCLI]

    def cli_cmd(self) -> None:
        """Start the ``data`` subcommand."""
        CliApp.run_subcommand(self)
