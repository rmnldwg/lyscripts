"""Predict risks of involvements for scenarios using drawn MCMC samples.

As the priors and posteriors, this computation, too, uses caching and may skip the
computation of these two initial steps if the cache directory is the same as during
their computation.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings, CliSettingsSource

from lyscripts.compute.utils import HDF5FileStorage
from lyscripts.configs import (
    DistributionConfig,
    GraphConfig,
    InvolvementConfig,
    ModalityConfig,
    ModelConfig,
    SamplingConfig,
)
from lyscripts.scenario import Scenario

logger = logging.getLogger(__name__)


class CmdSettings(BaseSettings):
    """Command line settings for the computation of posterior state distributions."""

    model_config = ConfigDict(extra="allow")

    cache_dir: Path = Field(
        default=Path.cwd() / ".cache",
        description="Cache directory for storing function calls.",
    )
    sampling: SamplingConfig
    risks: HDF5FileStorage = Field(description="Storage for the computed risks.")
    graph: GraphConfig
    model: ModelConfig = ModelConfig()
    distributions: dict[str, DistributionConfig] = Field(
        default={},
        description=(
            "Mapping of model T-categories to predefined distributions over "
            "diagnose times."
        ),
    )
    modalities: dict[str, ModalityConfig] = Field(
        default={},
        description=(
            "Maps names of diagnostic modalities to their specificity/sensitivity."
        ),
    )
    scenarios: list[Scenario] = Field(
        default=[],
        description="List of scenarios to compute risks for.",
    )


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """Add an ``ArgumentParser`` to the subparsers action."""
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """Add arguments to a ``subparsers`` instance and run its main function when chosen.

    This is called by the parent module that is called via the command line.
    """
    parser.add_argument(
        "--configs",
        default=[],
        nargs="*",
        help=(
            "Path(s) to YAML configuration file(s). Subsequent files overwrite "
            "previous ones. Command line arguments take precedence over all files."
        ),
    )
    parser.set_defaults(
        run_main=main,
        cli_settings_source=CliSettingsSource(
            settings_cls=CmdSettings,
            cli_use_class_docs_for_groups=True,
            root_parser=parser,
        ),
    )


def compute_risks(
    model_config: ModelConfig,
    graph_config: GraphConfig,
    dist_configs: dict[str, DistributionConfig],
    modality_configs: dict[str, ModalityConfig],
    posteriors: np.ndarray,
    involvement: InvolvementConfig,
    progress_desc: str = "Computing risks from posteriors",
) -> np.ndarray:
    """Compute the risk of ``involvement`` from each of the ``posteriors``.

    Essentially, this only calls the model's :py:meth:`lymph.models.Model.marginalize`
    method, as nothing more is necessary than to marginalize the full posterior state
    distribution over the states that correspond to the involvement of interest.
    """


def main(args: argparse.Namespace):
    """Run the main risk prediction routine."""
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
