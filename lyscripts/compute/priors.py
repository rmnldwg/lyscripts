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
from typing import Literal

import numpy as np
from lydata.utils import ModalityConfig
from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings, CliSettingsSource
from rich import progress

from lyscripts.compute.utils import HDF5FileStorage, get_cached
from lyscripts.configs import (
    DistributionConfig,
    GraphConfig,
    ModelConfig,
    SamplingConfig,
    ScenarioConfig,
    add_dists,
    construct_model,
)
from lyscripts.utils import merge_yaml_configs

logger = logging.getLogger(__name__)


class CmdSettings(BaseSettings):
    """Settings required to compute priors from model configs and samples."""

    model_config = ConfigDict(extra="allow")

    cache_dir: Path = Field(
        default=Path.cwd() / ".cache",
        description="Cache directory for storing function calls.",
    )
    sampling: SamplingConfig
    priors: HDF5FileStorage = Field(description="Storage for the computed priors.")
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


def compute_priors(
    model_config: ModelConfig,
    graph_config: GraphConfig,
    dist_configs: dict[str, DistributionConfig],
    samples: np.ndarray,
    t_stages: list[int | str],
    t_stages_dist: list[float],
    mode: Literal["HMM", "BN"] = "HMM",
    progress_desc: str = "Computing priors from samples",
) -> np.ndarray:
    """Compute prior state distributions from the ``samples`` for the ``model``.

    This will call the ``model`` method :py:meth:`~lymph.types.Model.state_dist`
    for each of the ``samples``. If ``t_stage`` is not provided, the priors will be
    computed by marginalizing over the provided ``t_stage_dist``. Otherwise, the
    priors will be computed for the given ``t_stage``.
    """
    model = construct_model(model_config, graph_config)
    model = add_dists(model, dist_configs)
    priors = []

    for sample in progress.track(
        sequence=samples,
        description="[blue]INFO     [/blue]" + progress_desc,
        total=len(samples),
    ):
        model.set_params(*sample)
        priors.append(
            sum(
                model.state_dist(t_stage=t, mode=mode) * p
                for t, p in zip(t_stages, t_stages_dist, strict=False)
            )
        )

    return np.stack(priors)


def main(args: argparse.Namespace):
    """Compute the prior state distribution for each sample."""
    yaml_configs = merge_yaml_configs(args.configs)
    cmd = CmdSettings(
        _cli_settings_source=args.cli_settings_source(parsed_args=args),
        **yaml_configs,
    )
    logger.debug(cmd.model_dump_json(indent=2))

    global_attrs = cmd.model_dump(include={"model", "graph", "distributions"})
    cmd.priors.set_attrs(attrs=global_attrs, dataset="/")

    samples = cmd.sampling.load()
    cached_compute_priors = get_cached(compute_priors, cmd.cache_dir)
    num_scenarios = len(cmd.scenarios)

    for i, scenario in enumerate(cmd.scenarios):
        scenario_fields = {"t_stages", "t_stages_dist", "mode"}
        scenario_attrs = scenario.model_dump(include=scenario_fields)

        priors = cached_compute_priors(
            model_config=cmd.model,
            graph_config=cmd.graph,
            dist_configs=cmd.distributions,
            samples=samples,
            progress_desc=f"Computing priors for scenario {i + 1}/{num_scenarios}",
            **scenario_attrs,
        )

        cmd.priors.save(values=priors, dataset=f"{i:03d}")
        cmd.priors.set_attrs(attrs=scenario_attrs, dataset=f"{i:03d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
