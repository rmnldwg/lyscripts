"""Given samples drawn during an MCMC round, compute the (prior) state distributions.

This is done for each sample. This may then later on be used to compute risks and
prevalences more quickly.

The computed priors are stored in an HDF5 file under a hash key of the scenario they
were computed for. This scenario consists of the T-stages it was computed for and the
distribution that was used to marginalize over them, as well as the model's computation
mode (hidden Markov model or Bayesian network).
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel, DirectoryPath, Field, FilePath
from pydantic._internal._utils import deep_update
from pydantic_settings import BaseSettings, CliSettingsSource
from rich import progress

from lyscripts import utils
from lyscripts.configs import (
    DistributionConfig,
    GraphConfig,
    ModalityConfig,
    ModelConfig,
    ScenarioConfig,
    construct_model,
)

logger = logging.getLogger(__name__)


class SamplesConfig(BaseModel):
    """Configuration for the samples file."""

    input_file: FilePath = Field(description="Path to the drawn samples (HDF5 file).")
    dset_name: str = Field(
        default="mcmc",
        description="Name of the dataset in the HDF5 file.",
    )
    thin: int = Field(
        gt=0,
        default=1,
        description="Only use every `thin`-th sample from the input file.",
    )


class PriorsConfig(BaseModel):
    """Configure how the priors are computed."""

    output_file: Path = Field(
        description="Path to file for storing the computed prior distributions."
    )
    dset_name: str | None = Field(
        default=None,
        description=(
            "Name of the dataset in the HDF5 file. If `None`, this will be "
            "dynamically computed from the scenario."
        ),
    )


class CmdSettings(BaseSettings):
    """Settings required to compute priors from model configs and samples."""

    cache_dir: DirectoryPath = Field(
        default=Path.cwd() / ".lyscripts_cache",
        description="Cache directory for storing function calls.",
    )
    samples: SamplesConfig
    priors: PriorsConfig
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
    scenarios: list[ScenarioConfig] = Field(
        default=[],
        description="List of scenarios to compute priors for.",
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
        "--params",
        default=[],
        nargs="*",
        help=(
            "Path(s) to parameter file(s). Subsequent files overwrite previous ones. "
            "Command line arguments take precedence over all files."
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


def compute_priors_using_cache(
    model_config: ModelConfig,
    graph_config: GraphConfig,
    samples: np.ndarray,
    scenario: ScenarioConfig,
    progress_desc: str = "Computing priors from samples",
) -> np.ndarray:
    """Compute prior state distributions from the ``samples`` for the ``model``.

    This will call the ``model`` method :py:meth:`~lymph.types.Model.state_dist`
    for each of the ``samples``. If ``t_stage`` is not provided, the priors will be
    computed by marginalizing over the provided ``t_stage_dist``. Otherwise, the
    priors will be computed for the given ``t_stage``.
    """
    model = construct_model(model_config=model_config, graph_config=graph_config)
    priors = []

    for sample in progress.track(
        sequence=samples,
        description="[blue]INFO     [/blue]" + progress_desc,
        total=len(samples),
    ):
        model.set_params(*sample)
        priors.append(
            sum(
                model.state_dist(t_stage=t, mode=scenario.mode) * p
                for t, p in zip(scenario.t_stages, scenario.t_stages_dist, strict=False)
            )
        )

    return np.stack(priors)


def main(args: argparse.Namespace):
    """Compute the prior state distribution for each sample."""
    yaml_params = {}
    for param_file in args.params:
        yaml_params = deep_update(yaml_params, utils.load_yaml_params(param_file))

    settings = CmdSettings(
        _cli_settings_source=args.cli_settings_source(parsed_args=args),
        **yaml_params,
    )
    logger.debug(settings.model_dump_json(indent=2))

    samples = utils.load_model_samples(args.samples)

    num_scenarios = len(settings.scenarios)
    for i, scenario in enumerate(settings.scenarios):
        _priors = compute_priors_using_cache(
            model_config=settings.model,
            graph_config=settings.graph,
            samples=samples,
            scenario=scenario,
            progress_desc=f"Computing priors for scenario {i + 1}/{num_scenarios}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
