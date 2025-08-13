"""Perform thermodynamic integration to evaluate the model evidence.

Using the functions provided by the `sample` module, this script implements
thermodynamic integration (TI) in order to compute the model evidence.
This is done by sampling the model parameters at different inverse temperatures
following a specified schedule.
"""

from __future__ import annotations

import os
from typing import Any

import emcee
import h5py
import numpy as np
from loguru import logger
from lydata.utils import ModalityConfig
from pydantic import Field

import lyscripts.sample as sample_module  # Import the module to set its global MODEL
from lyscripts.cli import assemble_main
from lyscripts.configs import (
    BaseCLI,
    DataConfig,
    DistributionConfig,
    GraphConfig,
    ModelConfig,
    SamplingConfig,
    ScheduleConfig,
    add_distributions,
    add_modalities,
    construct_model,
)
from lyscripts.utils import get_hdf5_backend


def init_ti_sampler(
    settings: IntegrateCLI,
    temp_idx: int,
    ndim: int,
    inv_temp: float,
    pool: Any,
) -> emcee.EnsembleSampler:
    """Initialize the ``emcee.EnsembleSampler`` for TI with the given ``settings''."""
    nwalkers = ndim * settings.sampling.walkers_per_dim
    backend = get_hdf5_backend(
        file_path=settings.sampling.storage_file,
        dataset=f"ti/{temp_idx + 1:0>2d}",
        nwalkers=nwalkers,
        ndim=ndim,
    )
    return emcee.EnsembleSampler(
        nwalkers=nwalkers,
        ndim=ndim,
        log_prob_fn=sample_module.log_prob_fn,
        kwargs={"inverse_temp": inv_temp},
        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
        backend=backend,
        pool=pool,
        blobs_dtype=[("log_prob", np.float64)],
        parameter_names=list(MODEL.get_named_params().keys()),
    )


class IntegrateCLI(BaseCLI):
    """Perform thermodynamic integration to compute the model evidence."""

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
    data: DataConfig
    sampling: SamplingConfig
    schedule: ScheduleConfig = Field(
        description="Configuration for generating inverse temperature schedule.",
    )

    def cli_cmd(self) -> None:
        """Start the ``integrate`` subcommand.

        The model construction and setup is done analogously to the
        ``sample`` command. Afterwards, an :py:class:`emcee.EnsembleSampler`
        is initialized (see :py:func:`init_sampler`) and :py:func:`run_sampling`,
        implemented in the ``sample``module, is executed twice for each TI step:
        once for the burn-in phase and once for the actual sampling phase.
        Thereby, the log likelihood is scaled by the respective inverse
        temperature of that step. All necessary settings for the sampling
        are passed by the ``sampling``argument, except for the inverse
        temperatures, which are provided by the ``schedule`` argument.
        """
        # as recommended in https://emcee.readthedocs.io/en/stable/tutorials/parallel/#
        os.environ["OMP_NUM_THREADS"] = "1"

        logger.debug(self.model_dump_json(indent=2))

        # ugly, but necessary for pickling
        global MODEL
        MODEL = construct_model(self.model, self.graph)
        MODEL = add_distributions(MODEL, self.distributions)
        MODEL = add_modalities(MODEL, self.modalities)
        MODEL.load_patient_data(**self.data.get_load_kwargs())
        ndim = MODEL.get_num_dims()

        # set MODEL in the sample module's namespace so log_prob_fn can access it
        sample_module.MODEL = MODEL

        schedule = self.schedule.get_schedule()

        # emcee does not support numpy's new random number generator yet.
        np.random.seed(self.sampling.seed)  # noqa: NPY002

        with sample_module.get_pool(self.sampling.cores) as pool:
            for idx, inv_temp in enumerate(schedule):
                sampler = init_ti_sampler(
                    settings=self,
                    temp_idx=idx,
                    ndim=ndim,
                    inv_temp=inv_temp,
                    pool=pool,
                )

                sample_module.run_sampling(
                    description=f"Burn-in phase: TI step {idx + 1}/{len(schedule)}",
                    sampler=sampler,
                    num_steps=self.sampling.burnin_steps,
                    check_interval=self.sampling.check_interval,
                    trust_factor=self.sampling.trust_factor,
                    relative_thresh=self.sampling.relative_thresh,
                    history_file=self.sampling.history_file,
                )

                sample_module.run_sampling(
                    description=f"Sampling phase: TI step {idx + 1}/{len(schedule)}",
                    sampler=sampler,
                    num_steps=self.sampling.num_steps,
                    reset_backend=True,
                    check_interval=self.sampling.num_steps,
                    thin_by=self.sampling.thin_by,
                )
            # copy last sampling round over to a group in the HDF5 file called "mcmc"
            with h5py.File(self.sampling.storage_file, mode="r+") as h5_file:
                h5_file.copy(
                    f"ti/{len(schedule):0>2d}",
                    h5_file,
                    name=self.sampling.dataset,
                )


if __name__ == "__main__":
    main = assemble_main(settings_cls=IntegrateCLI, prog_name="integrate")
    main()
