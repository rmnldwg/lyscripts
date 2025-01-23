"""Given samples drawn during an MCMC round, compute the (prior) state distributions.

This is done for each sample and for a list of specified scenarios. The computation is
cached at a location specified by the ``--cache_dir`` argument using ``joblib``.
"""

from typing import Literal

import numpy as np
from loguru import logger
from pydantic import Field
from rich import progress

from lyscripts.cli import assemble_main
from lyscripts.compute.utils import BaseComputeCLI, HDF5FileStorage, get_cached
from lyscripts.configs import (
    DistributionConfig,
    GraphConfig,
    ModelConfig,
    add_dists,
    construct_model,
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


class PriorsCLI(BaseComputeCLI):
    """Compute the prior state distributions from MCMC samples."""

    priors: HDF5FileStorage = Field(description="Storage for the computed priors.")

    def cli_cmd(self) -> None:
        """Start the ``priors`` subcommand.

        Given a ``graph``, ``model``, ``distributions`` over diagnosis times, and
        MCMC samples loaded from the ``sampling`` argument, this command computes the
        prior state distributions for each of the specified ``scenarios``.

        Precomputing these state distributions is useful, because they largely only
        depend on T-stage and not on the diagnosis or involvement of interest. Hence,
        computing the :py:mod:`~lyscripts.compute.posteriors` and
        :py:mod:`~lyscripts.compute.risks` can be sped up.

        Note that this command will use `joblib`_ to cache its computations.

        .. _joblib: https://joblib.readthedocs.io/
        """
        logger.debug(self.model_dump_json(indent=2))
        global_attrs = self.model_dump(include={"model", "graph", "distributions"})
        self.priors.set_attrs(attrs=global_attrs, dataset="/")

        samples = self.sampling.load()
        cached_compute_priors = get_cached(compute_priors, self.cache_dir)
        num_scenarios = len(self.scenarios)

        for i, scenario in enumerate(self.scenarios):
            _fields = {"t_stages", "t_stages_dist", "mode"}
            prior_kwargs = scenario.model_dump(include=_fields)

            priors = cached_compute_priors(
                model_config=self.model,
                graph_config=self.graph,
                dist_configs=self.distributions,
                samples=samples,
                progress_desc=f"Computing priors for scenario {i + 1}/{num_scenarios}",
                **prior_kwargs,
            )

            self.priors.save(values=priors, dataset=f"{i:03d}")
            self.priors.set_attrs(attrs=prior_kwargs, dataset=f"{i:03d}")


if __name__ == "__main__":
    main = assemble_main(settings_cls=PriorsCLI, prog_name="compute priors")
    main()
