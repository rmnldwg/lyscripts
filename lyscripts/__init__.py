"""An API and multiple CLIs to interact with lymphatic progression data and models.

The lyscripts package provides a set of tools to perform common tasks w.r.t. handling
and analyzing `lymphatic progression data`_ or perform inference on that data using
our `lymphatic progression models`_.

The package is structured hierarchically into submodules. At the top level, we provide
some :py:mod:`utilities <lyscripts.utils>` and helpful `pydantic`_
:py:mod:`configurations <lyscripts.configs>`.

Under the :py:mod:`~lyscripts.data` module, functions and subcommands are defined
that perform common tasks that one might face when building data and modelling
pipelines based on `lymphatic progression data`_.

Back in the top level we also find a very important CLI: The :py:mod:`~lyscripts.sample`
CLI. With it, one may specify a dataset to infer lymphatic progression model parameters
from it. These learned parameters may subsequently be used by one of the
:py:mod:`~lyscripts.compute` subcommands to predict the personalized risk of occult
disease in the lymph drainage system of head and neck cancer patients.

.. _lymphatic progression data: https://github.com/rmnldwg/lydata
.. _lymphatic progression models: https://github.com/rmnldwg/lymph
.. _pydantic: https://docs.pydantic.dev/latest/
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

    compute: CliSubCommand[compute.ComputeCLI]
    data: CliSubCommand[data.DataCLI]
    sample: CliSubCommand[sample.SampleCLI]
    schedule: CliSubCommand[schedule.ScheduleCLI]

    def __init__(self, **kwargs):
        """Add logging configuration to the lyscripts CLI."""
        configure_logging(argv=sys.argv, console=console)
        super().__init__(**kwargs)

    def cli_cmd(self) -> None:
        """Start the main lyscripts CLI."""
        logger.debug("Starting lyscripts CLI.")

        if self.version:
            logger.info(f"lyscripts {__version__}")
            return

        CliApp.run_subcommand(self)


main = assemble_main(settings_cls=LyscriptsCLI, prog_name="lyscripts")
