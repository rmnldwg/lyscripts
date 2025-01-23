"""Compute posterior state distributions.

The posteriors are computed from drawn samples for a list of defined scenarios. If
priors have already been computed from the samples and the ``--cache_dir`` argument
is the same as during that computation, the priors will automatically be loaded from
the cache.
"""

from typing import Literal

import numpy as np
from loguru import logger
from lymph import models
from pydantic import Field
from rich import progress

from lyscripts.cli import assemble_main
from lyscripts.compute.priors import compute_priors
from lyscripts.compute.utils import BaseComputeCLI, HDF5FileStorage, get_cached
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


class PosteriorsCLI(BaseComputeCLI):
    """Compute posterior state distributions for different diagnosis scenarios."""

    modalities: dict[str, ModalityConfig] = Field(
        default={},
        description=(
            "Maps names of diagnostic modalities to their specificity/sensitivity."
        ),
    )
    posteriors: HDF5FileStorage = Field(
        description="Storage for the computed posteriors."
    )

    def cli_cmd(self) -> None:
        """Start the ``posteriors`` subcommand.

        This will compute the posterior state distributions, given a personalized
        diagnosis pattern, for each of the scenarios provided to the command.
        """
        logger.debug(self.model_dump_json(indent=2))

        global_attrs = self.model_dump(
            include={"model", "graph", "distributions", "modalities"},
        )
        self.posteriors.set_attrs(attrs=global_attrs, dataset="/")

        samples = self.sampling.load()
        cached_compute_priors = get_cached(compute_priors, self.cache_dir)
        cached_compute_posteriors = get_cached(compute_posteriors, self.cache_dir)
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

            _fields = {"diagnosis", "midext", "mode"}
            posterior_kwargs = scenario.model_dump(include=_fields)

            posteriors = cached_compute_posteriors(
                model_config=self.model,
                graph_config=self.graph,
                dist_configs=self.distributions,
                modality_configs=self.modalities,
                priors=_priors,
                progress_desc=f"Computing posteriors for scenario {i + 1}/{num_scens}",
                **posterior_kwargs,
            )

            self.posteriors.save(values=posteriors, dataset=f"{i:03d}")
            self.posteriors.set_attrs(attrs=prior_kwargs, dataset=f"{i:03d}")
            self.posteriors.set_attrs(attrs=posterior_kwargs, dataset=f"{i:03d}")


if __name__ == "__main__":
    main = assemble_main(settings_cls=PosteriorsCLI, prog_name="compute posteriors")
    main()
