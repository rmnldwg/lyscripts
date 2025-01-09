"""Top-level module of the `lyscripts` package.

It contains the :py:func:`.main` function that is used to start the command line
interface (CLI) for the package.

Also, it configures the logging system and sets the metadata of the package.
"""

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

    def cli_cmd(self) -> None:
        """Start the main lyscripts CLI."""
        logger.remove()
        # logger.add(sys.stderr, level=self.log_level)

        if self.version:
            logger.info(f"lyscripts {__version__}")
            return

        CliApp.run_subcommand(self)


main = _assemble_main(settings_cls=LyscriptsCLI, prog_name="lyscripts")
