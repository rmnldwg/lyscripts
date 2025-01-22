"""Predict risks of involvements for scenarios using drawn MCMC samples.

As the priors and posteriors, this computation, too, uses caching and may skip the
computation of these two initial steps if the cache directory is the same as during
their computation.
"""

import numpy as np
from loguru import logger
from pydantic import Field
from rich import progress

from lyscripts.cli import assemble_main
from lyscripts.compute.posteriors import compute_posteriors
from lyscripts.compute.priors import compute_priors
from lyscripts.compute.utils import BaseComputeCLI, HDF5FileStorage, get_cached
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


class RisksCLI(BaseComputeCLI):
    """Predict the risk of involvement scenarios from model samples given diagnoses."""

    modalities: dict[str, ModalityConfig] = Field(
        default={},
        description=(
            "Maps names of diagnostic modalities to their specificity/sensitivity."
        ),
    )
    risks: HDF5FileStorage = Field(description="Storage for the computed risks.")

    def cli_cmd(self) -> None:
        """Start the ``risks`` subcommand."""
        logger.debug(self.model_dump_json(indent=2))
        global_attrs = self.model_dump(
            include={"model", "graph", "distributions", "modalities"},
        )
        self.risks.set_attrs(attrs=global_attrs, dataset="/")

        samples = self.sampling.load()
        cached_compute_priors = get_cached(compute_priors, self.cache_dir)
        cached_compute_posteriors = get_cached(compute_posteriors, self.cache_dir)
        cached_compute_risks = get_cached(compute_risks, self.cache_dir)
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

            _posteriors = cached_compute_posteriors(
                model_config=self.model,
                graph_config=self.graph,
                dist_configs=self.distributions,
                modality_configs=self.modalities,
                priors=_priors,
                progress_desc=f"Computing posteriors for scenario {i + 1}/{num_scens}",
                **posterior_kwargs,
            )

            _fields = {"involvement"}
            risk_kwargs = scenario.model_dump(include=_fields)

            risks = cached_compute_risks(
                model_config=self.model,
                graph_config=self.graph,
                dist_configs=self.distributions,
                modality_configs=self.modalities,
                posteriors=_posteriors,
                progress_desc=f"Computing risks for scenario {i + 1}/{num_scens}",
                **risk_kwargs,
            )

            self.risks.save(values=risks, dataset=f"{i:03d}")
            self.risks.set_attrs(attrs=prior_kwargs, dataset=f"{i:03d}")
            self.risks.set_attrs(attrs=posterior_kwargs, dataset=f"{i:03d}")
            self.risks.set_attrs(attrs=risk_kwargs, dataset=f"{i:03d}")


if __name__ == "__main__":
    main = assemble_main(settings_cls=RisksCLI, prog_name="compute risks")
    main()
