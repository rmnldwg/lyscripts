"""Prevalence prediction module.

This computes the prevalence of an observed involvement pattern, given a trained model.
It can also compare this prediction to the observed prevalence in the data. As for the
risk prediction, this uses caching and computes the priors first.
"""

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import lydata  # noqa: F401
import numpy as np
import pandas as pd
from lydata import C, Q
from lydata.accessor import NoneQ
from lymph import models
from pydantic import Field
from pydantic_settings import CliSettingsSource
from rich import progress

from lyscripts import utils
from lyscripts.compute.priors import compute_priors
from lyscripts.compute.utils import (
    ComputeCmdSettings,
    HDF5FileStorage,
    get_cached,
)
from lyscripts.configs import (
    DiagnosisConfig,
    DistributionConfig,
    GraphConfig,
    ModalityConfig,
    ModelConfig,
    ScenarioConfig,
    add_dists,
    add_modalities,
    construct_model,
)

logger = logging.getLogger(__name__)


class CmdSettings(ComputeCmdSettings):
    """Command line settings for the computation of prevalences."""

    modalities: dict[str, ModalityConfig] = Field(
        default={},
        description=(
            "Maps names of diagnostic modalities to their specificity/sensitivity."
        ),
    )
    prevalences: HDF5FileStorage = Field(
        description="Storage for the computed prevalences.",
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


def compute_prevalences(
    model_config: ModelConfig,
    graph_config: GraphConfig,
    dist_configs: dict[str, DistributionConfig],
    modality_configs: dict[str, ModalityConfig],
    priors: np.ndarray,
    diagnosis: DiagnosisConfig,
    midext: bool | None = None,
    progress_desc: str = "Computing prevalences from priors",
) -> np.ndarray:
    """Compute the prevalence of a diagnosis given the priors and the model."""
    model = construct_model(model_config, graph_config)
    model = add_dists(model, dist_configs)

    if len(modality_configs) != 1:
        msg = "Only one modality is supported for prevalence prediction."
        logger.error(msg)
        raise ValueError(msg)

    model = add_modalities(model, modality_configs)
    prevalences = []
    kwargs = {"midext": midext} if isinstance(model, models.Midline) else {}

    for prior in progress.track(
        sequence=priors,
        description="[blue]INFO     [/blue]" + progress_desc,
        total=len(priors),
    ):
        obs_dist = model.obs_dist(given_state_dist=prior)
        prevalences.append(
            model.marginalize(
                given_state_dist=obs_dist,
                involvement=diagnosis.to_involvement(next(iter(modality_configs))),
                **kwargs,
            )
        )

    return np.stack(prevalences)


def get_query(diagnosis: DiagnosisConfig) -> Q:
    """Transform a diagnosis into a query for the data."""
    result = NoneQ()
    for side in ["ipsi", "contra"]:
        for modality, pattern in getattr(diagnosis, side, {}).items():
            for lnl, value in pattern.items():
                column = (modality, side, lnl)
                result &= C(column) == value
    return result


def observe_prevalence(
    data: pd.DataFrame,
    scenario_config: ScenarioConfig,
    mapping: dict[int, str] | Callable[[int], str] | None = None,
) -> tuple[int, int]:
    """Extract prevalence defined in a ``scenario`` from the ``data``.

    ``mapping`` defines how the T-stages in the data are supposed to be mapped to the
    T-stages defined in the ``scenario``.

    Warning:
    --------
        When computing prevalences for unilateral models, the contralateral diagnosis
        will still be considered for computing the prevalence in the *data*.

    """
    mapping = mapping or {0: "early", 1: "ealy", 2: "early", 3: "late", 4: "late"}
    data["tumor", "1", "t_stage"] = data.ly.t_stage.map(mapping)

    has_t_stage = C("t_stage").isin(scenario_config.t_stages)
    has_midext = C("midext") == scenario_config.midext
    portion = data.ly.portion(
        query=get_query(scenario_config.diagnosis),
        given=has_t_stage & has_midext,
    )
    return portion.match, portion.total


def main(args: argparse.Namespace):
    """Run the main prevalence prediction routine."""
    yaml_configs = utils.merge_yaml_configs(args.configs)
    cmd = CmdSettings(
        _cli_settings_source=args.cli_settings_source(parsed_args=args),
        **yaml_configs,
    )
    logger.debug(cmd.model_dump_json(indent=2))

    global_attrs = cmd.model_dump(
        include={"model", "graph", "distributions", "modalities"},
    )
    cmd.prevalences.set_attrs(attrs=global_attrs, dataset="/")

    samples = cmd.sampling.load()
    cached_compute_priors = get_cached(compute_priors, cmd.cache_dir)
    cached_compute_prevalences = get_cached(compute_prevalences, cmd.cache_dir)
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

        _fields = {"diagnosis", "midext"}
        prevalence_kwargs = scenario.model_dump(include=_fields)

        prevalences = cached_compute_prevalences(
            model_config=cmd.model,
            graph_config=cmd.graph,
            dist_configs=cmd.distributions,
            modality_configs=cmd.modalities,
            priors=_priors,
            progress_desc=f"Computing prevalences for scenario {i + 1}/{num_scenarios}",
            **prevalence_kwargs,
        )
        # TODO: Add observation of prevalence in the data.
        cmd.prevalences.save(values=prevalences, dataset=f"{i:03d}")
        cmd.prevalences.set_attrs(attrs=prevalence_kwargs, dataset=f"{i:03d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
