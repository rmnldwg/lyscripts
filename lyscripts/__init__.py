"""Initial entry point for the lyscripts package and CLIs.

This top-level module configures and provides the top-level CLI through which all
subcommands can be accessed.
"""

import sys
from typing import Literal

import pandas as pd
from loguru import logger
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    CliApp,
    CliImplicitFlag,
    CliSubCommand,
)

from lyscripts import compute, data, sample, schedule  # noqa: F401
from lyscripts._version import version
from lyscripts.cli import assemble_main, configure_logging
from lyscripts.utils import console

__version__ = version
__description__ = "Package to interact with lymphatic progression data and models."
__author__ = "Roman Ludwig"
__email__ = "roman.ludwig@usz.ch"
__uri__ = "https://github.com/rmnldwg/lyscripts"

# activate copy on write in pandas.
# See https://pandas.pydata.org/docs/user_guide/copy_on_write.html
pd.options.mode.copy_on_write = True

logger.disable("lyscripts")


class LyscriptsCLI(BaseSettings):
    """A CLI to interact with lymphatic progression data and models."""

    version: CliImplicitFlag[bool] = Field(
        default=False,
        description="Display the version of lyscripts and exit.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Set the log level of the lyscripts CLI.",
    )

    data: CliSubCommand[data.DataCLI]
    sample: CliSubCommand[sample.SampleCLI]
    compute: CliSubCommand[compute.ComputeCLI]
    schedule: CliSubCommand[schedule.ScheduleCLI]

    def __init__(self, **kwargs):
        """Add logging configuration to the lyscripts CLI."""
        configure_logging(argv=sys.argv, console=console)
        super().__init__(**kwargs)

    def cli_cmd(self) -> None:
        """Start the main lyscripts CLI.

        If the ``version`` flag is set, the version of lyscripts is displayed and the
        program exits. Otherwise, the lyscripts CLI runs one of the subcommands.
        """
        logger.debug("Starting lyscripts CLI.")

        if self.version:
            logger.info(f"lyscripts {__version__}")
            return

        CliApp.run_subcommand(self)


main = assemble_main(settings_cls=LyscriptsCLI, prog_name="lyscripts")
