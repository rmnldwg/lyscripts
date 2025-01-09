"""Learn the model params from preprocessed input data using MCMC sampling.

This command allows us to infer the parameters of a predefined probabilistic model
from detailed per-patient lymph node level involvement data.

The model, data, and sampling configuration can be specified in one or several YAML
files, and/or via command line arguments.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from loguru import logger

from lyscripts.cli import _assemble_main

try:
    from multiprocess import Pool
except ModuleNotFoundError:
    from multiprocessing import Pool

from pathlib import Path

import emcee
import numpy as np
import pandas as pd
from lydata.utils import ModalityConfig
from lymph.types import ParamsType
from pydantic import Field
from pydantic_settings import (
    CliSettingsSource,
)
from rich.progress import Progress, ProgressColumn, Task, TimeElapsedColumn, track
from rich.text import Text

from lyscripts.configs import (
    BaseCmdSettings,
    DataConfig,
    add_dists,
    add_modalities,
    construct_model,
)
from lyscripts.utils import (
    console,
    get_hdf5_backend,
    merge_yaml_configs,
)

logger = logging.getLogger(__name__)

_BURNIN_KWARGS = {
    "max_burnin",
    "check_interval",
    "trust_factor",
    "relative_thresh",
    "history_file",
}
_SAMPLING_KWARGS = {"nsteps", "thin"}


class CompletedItersColumn(ProgressColumn):
    """A column that displays the completed number of iterations."""

    def __init__(self, table_column=None, it: int = 0):
        """Initialize the column with number of previous iterations."""
        super().__init__(table_column)
        self.it = it

    def render(self, task: Task) -> Text:
        """Render total iterations."""
        if task.completed is None:
            return Text("? it", style="progress.data.steps")
        return Text(f"{task.completed + self.it} it", style="progress.data.steps")


class ItersPerSecondColumn(ProgressColumn):
    """A column that displays the number of iterations per second."""

    def render(self, task: Task) -> Text:
        """Render iterations per second."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("? it/s", style="progress.data.speed")
        return Text(f"{speed:.2f} it/s", style="progress.data.speed")


MODEL = None


def log_prob_fn(theta: ParamsType, inverse_temp: float = 1.0) -> tuple[float, float]:
    """Compute log-prob using global variables because of pickling.

    An inverse temperature ``inverse_temp`` can be provided for thermodynamic
    integration.
    """
    return inverse_temp * MODEL.likelihood(given_params=theta), inverse_temp


def get_starting_state(sampler: emcee.EnsembleSampler) -> np.ndarray:
    """Try to extract a starting state from a ``sampler``.

    Create a random starting state if no one was found.
    """
    try:
        state = sampler.backend.get_last_sample()
        logger.info(
            f"Resuming from {sampler.backend.filename} with {sampler.iteration} "
            "stored iterations."
        )
    except AttributeError:
        state = np.random.uniform(size=(sampler.nwalkers, sampler.ndim))
        logger.debug(f"No stored samples found. Starting from random state {state}.")

    return state


def get_burnin_history(file: Path | None) -> pd.DataFrame:
    """Try to load the history of an interrupted burn-in phase from a file.

    It will look for the given ``file``, but with the suffix ``.tmp``, indicating that
    a previous run was interrupted and can be continued.

    If no file is found, an empty DataFrame is returned.
    """
    if file is None or not file.with_suffix(".tmp").exists():
        return pd.DataFrame(
            columns=["steps", "acor_times", "accept_fracs", "max_log_probs"],
        ).set_index("steps")

    return pd.read_csv(file.with_suffix(".tmp"), index_col="steps")


def is_converged(
    iteration: int,
    new_acor_time: float,
    old_acor_time: float,
    trust_factor: float,
    relative_thresh: float,
) -> bool:
    """Check if the chain has converged based on the autocorrelation time."""
    return (
        new_acor_time * trust_factor < iteration
        and np.abs(new_acor_time - old_acor_time) / new_acor_time < relative_thresh
    )


def _get_columns(it: int = 0) -> list[ProgressColumn]:
    """Get the default progress columns for the MCMC sampling."""
    return [
        *Progress.get_default_columns(),
        ItersPerSecondColumn(),
        CompletedItersColumn(it=it),
        TimeElapsedColumn(),
    ]


def run_burnin(
    sampler: emcee.EnsembleSampler,
    max_burnin: int | None = None,
    check_interval: int = 100,
    trust_factor: float = 50.0,
    relative_thresh: float = 0.05,
    history_file: Path | None = None,
) -> None:
    """Run the burn-in phase of the MCMC sampling.

    This will run the sampler for ``max_burnin`` steps or (if ``max_burnin`` is `None`)
    until convergence is reached. The convergence criterion is based on the
    autocorrelation time of the chain, which is computed every ``check_interval`` steps.
    The chain is considered to have converged if the autocorrelation time is smaller
    than ``trust_factor`` times the number of iterations and the relative change in the
    autocorrelation time is smaller than ``relative_thresh``.

    The samples of the burn-in phase will be stored in the backend of the ``sampler``.
    A history of some burn-in metrics will be stored at ``history_path`` if provided.
    """
    state = get_starting_state(sampler)
    history = get_burnin_history(history_file)
    previous_accepted = 0

    with Progress(*_get_columns(it=sampler.iteration), console=console) as progress:
        task = progress.add_task(
            description="[blue]INFO     [/blue]Burn-in phase ",
            total=max_burnin,
        )
        while sampler.iteration < (max_burnin or np.inf):
            for state in sampler.sample(state, iterations=check_interval):  # noqa: B007, B020
                progress.update(task, advance=1)

            new_acor_time = sampler.get_autocorr_time(tol=0).mean()
            old_acor_time = history.iloc[-1].acor_times if len(history) > 0 else np.inf

            newly_accepted = np.sum(sampler.backend.accepted) - previous_accepted
            new_accept_frac = newly_accepted / (sampler.nwalkers * check_interval)
            previous_accepted = np.sum(sampler.backend.accepted)

            history.loc[sampler.iteration] = [
                new_acor_time,
                new_accept_frac,
                np.max(state.log_prob),
            ]
            logger.debug(history.iloc[-1].to_dict())
            if history_file is not None:
                history.to_csv(history_file.with_suffix(".tmp"), index=True)

            if max_burnin is None and is_converged(
                iteration=sampler.iteration,
                new_acor_time=new_acor_time,
                old_acor_time=old_acor_time,
                trust_factor=trust_factor,
                relative_thresh=relative_thresh,
            ):
                logger.info(f"Sampling converged after {sampler.iteration} steps.")
                break

    if history_file is not None:
        history_file.with_suffix(".tmp").rename(history_file)


def run_sampling(
    sampler: emcee.EnsembleSampler,
    nsteps: int,
    thin: int,
) -> None:
    """Run the MCMC sampling phase to produce ``nsteps`` samples.

    This sampling will definitely produce ``nsteps`` samples, irrespective of the
    ``thin`` parameter, which controls how many steps in between two stored samples are
    skipped. The samples will be stored in the backend of the ``sampler``.

    Note that this will reset the ``sampler``'s backend, assuming the stored samples are
    from the burn-in phase.
    """
    state = get_starting_state(sampler)
    logger.debug("Resetting backend of sampler.")
    sampler.backend.reset(sampler.nwalkers, sampler.ndim)

    for _sample in track(
        sequence=sampler.sample(state, iterations=nsteps * thin, thin=thin, store=True),
        description="[blue]INFO     [/blue]Sampling phase",
        total=nsteps * thin,
        console=console,
    ):
        pass


class DummyPool:
    """Dummy class to allow for no multiprocessing."""

    def __enter__(self) -> None:
        """Enter the context manager."""
        ...

    def __exit__(self, *args) -> None:
        """Exit the context manager."""
        ...


def get_pool(num_cores: int | None) -> Any | DummyPool:  # type: ignore
    """Get a ``multiprocess(ing)`` pool or ``DummyPool``.

    Returns a ``multiprocess(ing)`` pool with ``num_cores`` cores if ``num_cores`` is
    not ``None``. Otherwise, a ``DummyPool`` is returned.
    """
    return Pool(num_cores) if num_cores is not None else DummyPool()


def init_sampler(settings: SampleCLI, ndim: int, pool: Any) -> emcee.EnsembleSampler:
    """Initialize the ``emcee.EnsembleSampler`` with the given ``settings``."""
    nwalkers = ndim * settings.sampling.walkers_per_dim
    backend = get_hdf5_backend(
        file_path=settings.sampling.file,
        dataset=settings.sampling.dataset,
        nwalkers=nwalkers,
        ndim=ndim,
    )
    return emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_prob_fn,
        kwargs={"inverse_temp": settings.sampling.inverse_temp},
        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
        backend=backend,
        pool=pool,
        blobs_dtype=[("inverse_temp", np.float64)],
    )


class SampleCLI(BaseCmdSettings):
    """Use MCMC to infer distributions over model parameters from data."""

    modalities: dict[str, ModalityConfig] = Field(
        default={},
        description=(
            "Maps names of diagnostic modalities to their specificity/sensitivity."
        ),
    )
    data: DataConfig

    def cli_cmd(self) -> None:
        """Start the ``sample`` subcommand."""
        # as recommended in https://emcee.readthedocs.io/en/stable/tutorials/parallel/#
        os.environ["OMP_NUM_THREADS"] = "1"

        logger.debug(self.model_dump_json(indent=2))

        # ugly, but necessary for pickling
        global MODEL
        MODEL = construct_model(self.model, self.graph)
        MODEL = add_dists(MODEL, self.distributions)
        MODEL = add_modalities(MODEL, self.modalities)
        MODEL.load_patient_data(**self.data.get_load_kwargs())
        ndim = MODEL.get_num_dims()

        # emcee does not support numpy's new random number generator yet.
        np.random.seed(self.sampling.seed)

        with get_pool(self.sampling.cores) as pool:
            sampler = init_sampler(self, ndim, pool)
            run_burnin(sampler, **self.sampling.model_dump(include=_BURNIN_KWARGS))
            run_sampling(sampler, **self.sampling.model_dump(include=_SAMPLING_KWARGS))


if __name__ == "__main__":
    main = _assemble_main(settings_cls=SampleCLI, prog_name="sample")
    main()
