"""Compute posterior state distributions.

The posteriors are computed from drawn samples for a list of defined scenarios. If
priors have already been computed from the samples and the ``--cache_dir`` argument
is the same as during that computation, the priors will automatically be loaded from
the cache.
"""

import argparse
import logging
from pathlib import Path
from typing import Literal

import numpy as np
from lymph import models
from pydantic import Field
from pydantic_settings import CliSettingsSource
from rich import progress

from lyscripts import utils
from lyscripts.compute.priors import compute_priors
from lyscripts.compute.utils import ComputeCmdSettings, HDF5FileStorage, get_cached
from lyscripts.configs import (
    DiagnosisConfig,
    DistributionConfig,
    GraphConfig,
    ModalityConfig,
    ModelConfig,
    add_dists,
    add_modalities,
    construct_model,
)

logger = logging.getLogger(__name__)


class CmdSettings(ComputeCmdSettings):
    """Command line settings for the computation of posterior state distributions."""

    modalities: dict[str, ModalityConfig] = Field(
        default={},
        description=(
            "Maps names of diagnostic modalities to their specificity/sensitivity."
        ),
    )
    posteriors: HDF5FileStorage = Field(
        description="Storage for the computed posteriors."
    )


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """Add an `ArgumentParser` to the subparsers action."""
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


def compute_posteriors(
    model_config: ModelConfig,
    graph_config: GraphConfig,
    dist_configs: dict[str, DistributionConfig],
    modality_configs: dict[str, ModalityConfig],
    priors: np.ndarray,
    diagnosis: DiagnosisConfig,
    midext: bool | None = None,
    mode: Literal["HMM", "BN"] = "HMM",
    progress_desc: str = "Computing posteriors from priors",
) -> np.ndarray:
    """Compute posterior state distributions from ``priors``.

    This calls the ``model`` method :py:meth:`~lymph.types.Model.posterior_state_dist`
    for each of the pre-computed ``priors``, given the specified ``diagnosis`` pattern.

    For the :py:class:`~lymph.models.Midline` model, the ``midext`` argument can be
    used to specify whether the midline extension is present or not.
    """
    model = construct_model(model_config, graph_config)
    model = add_dists(model, dist_configs)
    model = add_modalities(model, modality_configs)
    posteriors = []
    kwargs = {"midext": midext} if isinstance(model, models.Midline) else {}

    for prior in progress.track(
        sequence=priors,
        description="[blue]INFO     [/blue]" + progress_desc,
        total=len(priors),
    ):
        posteriors.append(
            model.posterior_state_dist(
                given_state_dist=prior,
                given_diagnosis=diagnosis,
                mode=mode,
                **kwargs,
            )
        )

    return np.stack(posteriors)


def main(args: argparse.Namespace) -> None:
    """Compute posteriors from priors or drawn samples."""
    yaml_confis = utils.merge_yaml_configs(args.configs)
    cmd = CmdSettings(
        _cli_settings_source=args.cli_settings_source(parsed_args=args),
        **yaml_confis,
    )
    logger.debug(cmd.model_dump_json(indent=2))

    global_attrs = cmd.model_dump(
        include={"model", "graph", "distributions", "modalities"},
    )
    cmd.posteriors.set_attrs(attrs=global_attrs, dataset="/")

    samples = cmd.sampling.load()
    cached_compute_priors = get_cached(compute_priors, cmd.cache_dir)
    cached_compute_posteriors = get_cached(compute_posteriors, cmd.cache_dir)
    num_scenarios = len(cmd.scenarios)

    for i, scenario in enumerate(cmd.scenarios):
        _fields = {"t_stages", "t_stages_dist", "mode"}
        prior_kwargs = scenario.model_dump(include=_fields)

        _priors = cached_compute_priors(
            model_config=cmd.model,
            graph_config=cmd.graph,
            dist_configs=cmd.distributions,
            samples=samples,
            progress_desc=f"Computing priors for scenario {i + 1}/{num_scenarios}",
            **prior_kwargs,
        )

        _fields = {"diagnosis", "midext", "mode"}
        posterior_kwargs = scenario.model_dump(include=_fields)

        posteriors = cached_compute_posteriors(
            model_config=cmd.model,
            graph_config=cmd.graph,
            dist_configs=cmd.distributions,
            modality_configs=cmd.modalities,
            priors=_priors,
            progress_desc=f"Computing posteriors for scenario {i + 1}/{num_scenarios}",
            **posterior_kwargs,
        )

        cmd.posteriors.save(values=posteriors, dataset=f"{i:03d}")
        cmd.posteriors.set_attrs(attrs=prior_kwargs, dataset=f"{i:03d}")
        cmd.posteriors.set_attrs(attrs=posterior_kwargs, dataset=f"{i:03d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
