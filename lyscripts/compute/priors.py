"""Given samples drawn during an MCMC round, compute the (prior) state distributions.

This is done for each sample and for a list of specified scenarios. The computation is
cached at a location specified by the ``--cache_dir`` argument using ``joblib``.
"""

import argparse
import logging
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field
from pydantic_settings import CliSettingsSource
from rich import progress

from lyscripts.compute.utils import ComputeCmdSettings, HDF5FileStorage, get_cached
from lyscripts.configs import (
    DistributionConfig,
    GraphConfig,
    ModelConfig,
    add_dists,
    construct_model,
)
from lyscripts.utils import merge_yaml_configs

logger = logging.getLogger(__name__)


class CmdSettings(ComputeCmdSettings):
    """Settings required to compute priors from model configs and samples."""

    priors: HDF5FileStorage = Field(description="Storage for the computed priors.")


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
    for each of the ``samples``. The prior state distributions are computed for
    each of the ``t_stages`` and marginalized over using the ``t_stages_dist``.
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
        _fields = {"t_stages", "t_stages_dist", "mode"}
        prior_kwargs = scenario.model_dump(include=_fields)

        priors = cached_compute_priors(
            model_config=cmd.model,
            graph_config=cmd.graph,
            dist_configs=cmd.distributions,
            samples=samples,
            progress_desc=f"Computing priors for scenario {i + 1}/{num_scenarios}",
            **prior_kwargs,
        )

        cmd.priors.save(values=priors, dataset=f"{i:03d}")
        cmd.priors.set_attrs(attrs=prior_kwargs, dataset=f"{i:03d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
