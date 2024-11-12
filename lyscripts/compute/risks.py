"""Predict risks of involvements for scenarios using drawn MCMC samples.

As the priors and posteriors, this computation, too, uses caching and may skip the
computation of these two initial steps if the cache directory is the same as during
their computation.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from pydantic import Field
from pydantic_settings import CliSettingsSource
from rich import progress

from lyscripts import utils
from lyscripts.compute.posteriors import compute_posteriors
from lyscripts.compute.priors import compute_priors
from lyscripts.compute.utils import ComputeCmdSettings, HDF5FileStorage, get_cached
from lyscripts.configs import (
    DistributionConfig,
    GraphConfig,
    InvolvementConfig,
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
    risks: HDF5FileStorage = Field(description="Storage for the computed risks.")


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
    model = construct_model(model_config, graph_config)
    model = add_dists(model, dist_configs)
    model = add_modalities(model, modality_configs)
    risks = []

    for posterior in progress.track(
        sequence=posteriors,
        description="[blue]INFO     [/blue]" + progress_desc,
        total=len(posteriors),
    ):
        risks.append(
            model.marginalize(involvement=involvement, given_state_dist=posterior)
        )

    return np.stack(risks)


def main(args: argparse.Namespace):
    """Run the main risk prediction routine."""
    yaml_configs = utils.merge_yaml_configs(args.configs)
    cmd = CmdSettings(
        _cli_settings_source=args.cli_settings_source(parsed_args=args),
        **yaml_configs,
    )
    logger.debug(cmd.model_dump_json(indent=2))

    global_attrs = cmd.model_dump(
        include={"model", "graph", "distributions", "modalities"},
    )
    cmd.risks.set_attrs(attrs=global_attrs, dataset="/")

    samples = cmd.sampling.load()
    cached_compute_priors = get_cached(compute_priors, cmd.cache_dir)
    cached_compute_posteriors = get_cached(compute_posteriors, cmd.cache_dir)
    cached_compute_risks = get_cached(compute_risks, cmd.cache_dir)
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

        _posteriors = cached_compute_posteriors(
            model_config=cmd.model,
            graph_config=cmd.graph,
            dist_configs=cmd.distributions,
            modality_configs=cmd.modalities,
            priors=_priors,
            progress_desc=f"Computing posteriors for scenario {i + 1}/{num_scenarios}",
            **posterior_kwargs,
        )

        _fields = {"involvement"}
        risk_kwargs = scenario.model_dump(include=_fields)

        risks = cached_compute_risks(
            model_config=cmd.model,
            graph_config=cmd.graph,
            dist_configs=cmd.distributions,
            modality_configs=cmd.modalities,
            posteriors=_posteriors,
            progress_desc=f"Computing risks for scenario {i + 1}/{num_scenarios}",
            **risk_kwargs,
        )

        cmd.risks.save(values=risks, dataset=f"{i:03d}")
        cmd.risks.set_attrs(attrs=risk_kwargs, dataset=f"{i:03d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
