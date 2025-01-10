"""Top-level module of the `lyscripts` package.

It contains the :py:func:`.main` function that is used to start the command line
interface (CLI) for the package.

Also, it configures the logging system and sets the metadata of the package.
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

from lyscripts import compute, data, evaluate, plot, sample, temp_schedule  # noqa: F401
from lyscripts._version import version
from lyscripts.cli import _assemble_main

__version__ = version
__description__ = "Package containing scripts used in lynference pipelines"
__author__ = "Roman Ludwig"
__email__ = "roman.ludwig@usz.ch"
__uri__ = "https://github.com/rmnldwg/lyscripts"

# activate copy on write in pandas.
# See https://pandas.pydata.org/docs/user_guide/copy_on_write.html
pd.options.mode.copy_on_write = True

logger.disable("lyscripts")


def somewhat_safely_get_loglevel(
    argv: list[str],
) -> Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
    """Set the log level of the lyscripts CLI.

    This is a bit of a hack, since the :py:class:`.LyscriptsCLI` class is not yet
    initialized when we need to set the log level. In case the provided log-level is
    not valid, :py:class:`.LyscriptsCLI` will raise an exception at a later point.
    """
    args_str = " ".join(argv)
    if "--log-level" in args_str:
        for log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            if log_level in args_str:
                return log_level

    return "INFO"


def configure_logging(argv: list[str]) -> None:
    """Configure the logging system of the lyscripts CLI.

    This function sets the log level and format of the lyscripts CLI. Notably, for
    a log-level of `DEBUG` the output will contain more information.
    """
    logger.enable("lyscripts")
    log_level = somewhat_safely_get_loglevel(argv=argv)
    logger.remove()
    kwargs = {"format": "<lvl>{message}</>"} if log_level != "DEBUG" else {}
    logger.add(
        sink=sys.stderr,
        level=log_level,
        **kwargs,
    )


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

    compute: CliSubCommand[compute.ComputeCLI]
    data: CliSubCommand[data.DataCLI]
    sample: CliSubCommand[sample.SampleCLI]

    def __init__(self, **kwargs):
        """Add logging configuration to the lyscripts CLI."""
        configure_logging(argv=sys.argv)
        super().__init__(**kwargs)

    def cli_cmd(self) -> None:
        """Start the main lyscripts CLI."""
        logger.debug("Starting lyscripts CLI.")

        if self.version:
            logger.info(f"lyscripts {__version__}")
            return

        CliApp.run_subcommand(self)


main = _assemble_main(settings_cls=LyscriptsCLI, prog_name="lyscripts")
