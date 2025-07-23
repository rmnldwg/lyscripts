"""Prevalence prediction module.

This computes the prevalence of an observed involvement pattern, given a trained model.
It can also compare this prediction to the observed prevalence in the data. As for the
risk prediction, this uses caching and computes the priors first.
"""

from collections.abc import Callable
from typing import Literal

import lydata  # noqa: F401
import numpy as np
import pandas as pd
from loguru import logger
from lydata import C, Q
from lydata.accessor import NoneQ, QueryPortion
from lydata.utils import is_old
from lymph import models
from pydantic import Field
from rich import progress

from lyscripts.cli import assemble_main
from lyscripts.compute.priors import compute_priors
from lyscripts.compute.utils import (
    BaseComputeCLI,
    HDF5FileStorage,
    get_cached,
)
from lyscripts.configs import (
    DataConfig,
    DiagnosisConfig,
    DistributionConfig,
    GraphConfig,
    ModalityConfig,
    ModelConfig,
    ScenarioConfig,
    add_distributions,
    add_modalities,
    construct_model,
)
from lyscripts.utils import console


def compute_prevalences(
    model_config: ModelConfig,
    graph_config: GraphConfig,
    dist_configs: dict[str, DistributionConfig],
    modality_configs: dict[str, ModalityConfig],
    priors: np.ndarray,
    diagnosis: dict[Literal["ipsi", "contra"], dict],
    midext: bool | None = None,
    progress_desc: str = "Computing prevalences from priors",
) -> np.ndarray:
    """Compute the prevalence of a diagnosis given the priors and the model."""
    model = construct_model(model_config, graph_config)
    model = add_distributions(model, dist_configs)

    if len(modality_configs) != 1:
        msg = "Only one modality is supported for prevalence prediction."
        logger.error(msg)
        raise ValueError(msg)

    model = add_modalities(model, modality_configs)
    prevalences = []
    kwargs = {"midext": midext} if isinstance(model, models.Midline) else {}

    for prior in progress.track(
        sequence=priors,
        description=progress_desc,
        total=len(priors),
        console=console,
    ):
        obs_dist = model.obs_dist(given_state_dist=prior)
        involvement = {
            side: diagnosis.get(side).get(next(iter(modality_configs)))
            for side in ["ipsi", "contra"]
        }

        if isinstance(model, models.Unilateral | models.HPVUnilateral):
            involvement = involvement.get("ipsi")

        prevalence = model.marginalize(
            given_state_dist=obs_dist,
            involvement=involvement,
            **kwargs,
        )

        if isinstance(model, models.Midline):
            # In this case, we need to renormalize the prevalence by the marginalized
            # probability of all states with midline extension. We must do this, because
            # we compute the analogous quantity for the data. In principle, we could
            # also compute the prevalence of the diagnosis *and* midline extension, but
            # we have decided to compute the diagnosis *given* midline extension.
            # https://github.com/lycosystem/lyscripts/blob/ea49ec/lyscripts/compute/prevalences.py#L217-L225
            midext_prob = model.marginalize(
                involvement=None,
                given_state_dist=obs_dist,
                **kwargs,
            )
            prevalence /= midext_prob

        prevalences.append(prevalence)

    return np.stack(prevalences)


def generate_query_from_diagnosis(diagnosis: DiagnosisConfig) -> Q:
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
) -> QueryPortion:
    """Extract prevalence defined in a ``scenario`` from the ``data``.

    ``mapping`` defines how the T-stages in the data are supposed to be mapped to the
    T-stages defined in the ``scenario``.

    It returns the number of patients that match the given scenario and the total
    number of patients that are considered. E.g., in the example below we 79 patients
    are of late T-stage and have a tumor extending over the midline. Of those, 30 were
    diagnosed with contralateral involvement in LNL II based on a CT scan.

    >>> data = next(lydata.load_datasets(year=2021, institution="usz"))
    >>> scenario_config = ScenarioConfig(
    ...     t_stages=["late"],
    ...     midext=True,
    ...     diagnosis=DiagnosisConfig(contra={"CT": {"II": True}}),
    ... )
    >>> observe_prevalence(data, scenario_config)
    QueryPortion(match=np.int64(7), total=np.int64(79))
    """
    mapping = mapping or DataConfig.model_fields["mapping"].default_factory()
    key = ("tumor", "1", "t_stage") if is_old(data) else ("tumor", "info", "t_stage")
    data[key] = data.ly.t_stage.map(mapping)

    has_t_stage = C("t_stage").isin(scenario_config.t_stages)
    if scenario_config.midext is None:
        has_midext = NoneQ()
    else:
        has_midext = C("midext") == scenario_config.midext

    # Note that below we compute the prevalence of the diagnosis *given* midline
    # extension. This means, that when computing the prevalence of the diagnosis in
    # the model, we need to renormalize by diving by the probability of midline
    # extension. For an older - but pretty surely correct - implementation see
    # https://github.com/lycosystem/lyscripts/blob/ea49ec/lyscripts/compute/prevalences.py#L217-L225
    return data.ly.portion(
        query=generate_query_from_diagnosis(scenario_config.diagnosis),
        given=has_t_stage & has_midext,
    )


class PrevalencesCLI(BaseComputeCLI):
    """Predict the prevalence of an involvement pattern from model samples."""

    modalities: dict[str, ModalityConfig] = Field(
        default={},
        description=(
            "Maps names of diagnostic modalities to their specificity/sensitivity."
        ),
    )
    prevalences: HDF5FileStorage = Field(
        description="Storage for the computed prevalences.",
    )
    data: DataConfig

    def cli_cmd(self) -> None:
        """Start the ``prevalences`` subcommand."""
        logger.debug(self.model_dump_json(indent=2))
        global_attrs = self.model_dump(
            include={"model", "graph", "distributions", "modalities"},
        )
        self.prevalences.set_attrs(attrs=global_attrs, dataset="/")

        samples = self.sampling.load()
        cached_compute_priors = get_cached(compute_priors, self.cache_dir)
        cached_compute_prevalences = get_cached(compute_prevalences, self.cache_dir)
        num_scens = len(self.scenarios)

        for i, scenario in enumerate(self.scenarios):
            _fields = {"t_stages", "t_stages_dist", "mode"}
            prior_kwargs = scenario.model_dump(include=_fields)

            _priors = cached_compute_priors(
                model_config=self.model,
                graph_config=self.graph,
                dist_configs=self.distributions,
                samples=samples,
                progress_desc=f"Computing priors for scenario {i + 1}/{num_scens}",
                **prior_kwargs,
            )

            _fields = {"diagnosis", "midext"}
            prevalence_kwargs = scenario.model_dump(include=_fields)

            prevalences = cached_compute_prevalences(
                model_config=self.model,
                graph_config=self.graph,
                dist_configs=self.distributions,
                modality_configs=self.modalities,
                priors=_priors,
                progress_desc=f"Computing prevalences for scenario {i + 1}/{num_scens}",
                **prevalence_kwargs,
            )

            portion = observe_prevalence(
                data=self.data.load(),
                scenario_config=scenario,
                mapping=self.data.mapping,
            )
            self.prevalences.save(values=prevalences, dataset=f"{i:03d}")
            self.prevalences.set_attrs(attrs=prior_kwargs, dataset=f"{i:03d}")
            self.prevalences.set_attrs(attrs=prevalence_kwargs, dataset=f"{i:03d}")
            self.prevalences.set_attrs(
                attrs={
                    "num_match": portion.match,
                    "num_total": portion.total,
                },
                dataset=f"{i:03d}",
            )


if __name__ == "__main__":
    main = assemble_main(settings_cls=PrevalencesCLI, prog_name="compute prevalences")
    main()
