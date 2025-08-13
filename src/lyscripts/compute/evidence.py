"""Compute the model evidence from MCMC samples.

Given the samples drawn during thermodynamic integration and their respective log
likelihoods, compute the model log evidence and the Bayesian Information Criterion.
"""

from __future__ import annotations

import json
from pathlib import Path

import emcee
import h5py
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import Field
from scipy.integrate import trapezoid

from lyscripts.cli import assemble_main
from lyscripts.configs import (
    BaseCLI,
    DataConfig,
    SamplingConfig,
    ScheduleConfig,
)

RNG = np.random.default_rng()


def comp_bic(log_probs: np.ndarray, num_params: int, num_data: int) -> float:
    r"""Compute the negative one half of the Bayesian Information Criterion (BIC).

    The BIC is defined as [^1]
    $$ BIC = k \\ln{n} - 2 \\ln{\\hat{L}} $$
    where $k$ is the number of parameters ``num_params``, $n$ the number of datapoints
    ``num_data`` and $\\hat{L}$ the maximum likelihood estimate of the ``log_prob``.
    It is constructed such that the following is an
    approximation of the model evidence:
    $$ p(D \\mid m) \\approx \\exp{\\left( - BIC / 2 \\right)} $$
    which is why this function returns the negative one half of it.

    [^1]: https://en.wikipedia.org/wiki/Bayesian_information_criterion
    """
    return np.max(log_probs) - num_params * np.log(num_data) / 2.0


def compute_evidence(
    temp_schedule: np.ndarray,
    log_probs: np.ndarray,
    num: int = 1000,
) -> tuple[float, float]:
    """Compute the evidence and its standard deviation.

    Given a ``temp_schedule`` of inverse temperatures and corresponding sets of
    ``log_probs``, draw ``num`` "paths" of log-probabilities and compute the evidence
    for each using trapezoidal integration.

    The evidence is then the mean of those ``num`` integrations, while the error is
    their standard deviation.
    """
    integrals = np.zeros(shape=num)
    for i in range(num):
        rand_idx = RNG.choice(log_probs.shape[1], size=log_probs.shape[0])
        drawn_accuracy = log_probs[np.arange(log_probs.shape[0]), rand_idx].copy()
        integrals[i] = trapezoid(y=drawn_accuracy, x=temp_schedule)
    return np.mean(integrals), np.std(integrals)


def compute_ti_results(
    settings: EvidenceCLI,
    temp_schedule: np.ndarray,
    metrics: dict,
    ndim: int,
    h5_file: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the results in case of a thermodynamic integration run."""
    num_temps = len(temp_schedule)

    if num_temps != len(h5_file["ti"]):
        raise RuntimeError(
            f"Parameters suggest temp schedule of length {num_temps}, "
            f"but stored are {len(h5_file['ti'])}",
        )

    nwalker = ndim * settings.sampling.walkers_per_dim
    nsteps = settings.sampling.num_steps
    ti_log_probs = np.zeros(shape=(num_temps, nsteps * nwalker))

    for i, run in enumerate(h5_file["ti"]):
        reader = emcee.backends.HDFBackend(
            settings.sampling.storage_file,
            name=f"ti/{run}",
            read_only=True,
        )
        ti_log_probs[i] = reader.get_blobs(flat=True)["log_prob"]

    evidence, evidence_std = compute_evidence(temp_schedule, ti_log_probs)
    metrics["evidence"] = evidence
    metrics["evidence_std"] = evidence_std

    return temp_schedule, ti_log_probs


class EvidenceCLI(BaseCLI):
    """Compute model evidence from thermodynamic integration samples."""

    data: DataConfig
    sampling: SamplingConfig
    schedule: ScheduleConfig = Field(
        description="Configuration for generating inverse temperature schedule.",
    )
    plots: Path = Field(
        default="./plots",
        description="Directory for storing plots.",
    )
    metrics: Path = Field(
        default="./metrics.json",
        description="Path to metrics file.",
    )

    def cli_cmd(self) -> None:
        """Start the ``evidence`` subcommand.

        Given the MCMC samples from thermodynamic integration provided by the
        ``sampling`` argument and the corresponding inverse temperature schedule,
        specified in the ``schedule`` argument, the model evidence is computed using
        the functions :py:func:`compute_ti_results` and :py:func`compute_evidence`.
        Further the BIC is evaluated.
        """
        data = self.data.load()

        metrics = {}

        temp_schedule = self.schedule.get_schedule()

        with h5py.File(self.sampling.storage_file, mode="r") as h5_file:
            # Get ndim from the HDF5 backend
            backend = emcee.backends.HDFBackend(
                self.sampling.storage_file,
                read_only=True,
                name=self.sampling.dataset,
            )
            ndim = backend.shape[1]
            logger.info(f"Inferred {ndim} parameters from stored samples")

            # if TI has been performed, compute the evidence
            if "ti" in h5_file:
                temp_schedule, ti_log_probs = compute_ti_results(
                    settings=self,
                    temp_schedule=temp_schedule,
                    metrics=metrics,
                    ndim=ndim,
                    h5_file=h5_file,
                )

                logger.info(
                    "Computed results of thermodynamic integration with "
                    f"{len(temp_schedule)} steps",
                )

                # store inverse temperatures and log-probs in CSV file
                self.plots.parent.mkdir(parents=True, exist_ok=True)

                beta_vs_accuracy = pd.DataFrame(
                    np.array(
                        [
                            temp_schedule,
                            np.mean(ti_log_probs, axis=1),
                            np.std(ti_log_probs, axis=1),
                        ],
                    ).T,
                    columns=["β", "accuracy", "std"],
                )
                beta_vs_accuracy.to_csv(self.plots, index=False)
                logger.info(f"Plotted β vs accuracy at {self.plots}")

            # use blobs, because also for TI, this is the unscaled log-prob
            final_log_probs = backend.get_blobs()["log_prob"]
            logger.info(
                f"Opened samples from emcee backend from {self.sampling.storage_file}",
            )

            # store metrics in JSON file
            self.metrics.parent.mkdir(parents=True, exist_ok=True)
            self.metrics.touch(exist_ok=True)

            metrics["BIC"] = comp_bic(
                log_probs=final_log_probs,
                num_params=ndim,
                num_data=len(data),
            )
            metrics["max_llh"] = np.max(final_log_probs)
            metrics["mean_llh"] = np.mean(final_log_probs)

            with open(self.metrics, mode="w", encoding="utf-8") as metrics_file:
                json.dump(metrics, metrics_file)

            logger.info(f"Wrote out metrics to {self.metrics}")


if __name__ == "__main__":
    main = assemble_main(settings_cls=EvidenceCLI, prog_name="compute evidence")
    main()
