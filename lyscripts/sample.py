"""Implementation of flexible MCMC sampling for lymphatic progression models.

This module provides both helpful functions for programmatically building and running
sampling pipelines, as well a CLI interface for th most common sampling use cases.

The core is the :py:func:`run_sampling` function. It has a flexible interface and
built-in convergence detection, as well as bookkeeping for monitoring and resuming
interrupted sampling runs. It can be used both during the burn-in phase and the actual
sampling phase.

For parallelization, the sampling tries to use the ``multiprocess(ing)`` module.
However, we have found that this is often not necessary when the model itself
distributes the computation of its likelihood to multiple cores (as numpy typically
does).
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from lyscripts.cli import assemble_main

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
from pydantic import BaseModel, Field
from rich.progress import Progress, ProgressColumn, Task, TimeElapsedColumn
from rich.text import Text

from lyscripts.configs import (
    BaseCLI,
    DataConfig,
    DistributionConfig,
    GraphConfig,
    ModelConfig,
    SamplingConfig,
    add_dists,
    add_modalities,
    construct_model,
)
from lyscripts.utils import console, get_hdf5_backend


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


class AcorTime(BaseModel, validate_assignment=True):
    """Storage for old and new autocorrelation times."""

    old: float
    new: float

    def update(self, new: float) -> None:
        """Update the autocorrelation time."""
        self.old = self.new
        self.new = new

    @property
    def relative_diff(self) -> float:
        """Get the relative difference between new and old autocorrelation time."""
        return np.abs(self.new - self.old) / self.new


class NumAccepted(BaseModel, validate_assignment=True):
    """Storage for old and new number of accepted proposals."""

    old: int
    new: int

    def update(self, new: int) -> None:
        """Update the number of accepted proposals."""
        self.old = self.new
        self.new = new

    @property
    def newly_accepted(self) -> int:
        """Get the number of newly accepted proposals."""
        return self.new - self.old


MODEL = None


def log_prob_fn(theta: ParamsType, inverse_temp: float = 1.0) -> tuple[float, float]:
    """Compute log-prob using global variables because of pickling.

    An inverse temperature ``inverse_temp`` can be provided for thermodynamic
    integration.
    """
    return inverse_temp * MODEL.likelihood(given_params=theta), inverse_temp


def ensure_initial_state(sampler: emcee.EnsembleSampler) -> np.ndarray:
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


def ensure_history_table(file: Path | None) -> pd.DataFrame:
    """Return the history table from a file or an empty DataFrame.

    It will try to load a history at the given ``file`` location, but with a ``.tmp``
    extension. This is the expected name and location of a history file that was
    stored during an interrupted sampling run.

    If no file is found, an empty DataFrame is returned.
    """
    if file is None or not file.with_suffix(".tmp").exists():
        return pd.DataFrame(
            columns=[
                "steps",
                "acor_times",
                "accept_fracs",
                "max_log_probs",
            ],
        ).set_index("steps")

    return pd.read_csv(file.with_suffix(".tmp"), index_col="steps")


def update_history_table(
    history: pd.DataFrame,
    history_file: Path | None,
    iteration: int,
    acor_time: float,
    accepted_frac: float,
    max_log_prob: float,
) -> pd.DataFrame:
    """Update the history table with the current iteration's information."""
    history.loc[iteration] = [acor_time, accepted_frac, max_log_prob]
    logger.debug(history.iloc[-1].to_dict())

    if history_file is not None:
        history.to_csv(history_file.with_suffix(".tmp"))

    return history


def is_converged(
    iteration: int,
    acor_time: AcorTime,
    trust_factor: float,
    relative_thresh: float,
) -> bool:
    """Check if the chain has converged based on the autocorrelation time.

    The criterion is based on the relative change of the autocorrelation time and
    whether the autocorrelation extimate can be trusted. Essentially, we only trust
    the estimate if it is smaller than ``trust_factor`` times the current ``iteration``.

    More details can be found in the `emcee documentation`_.

    .. _emcee documentation: https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
    """
    return (
        acor_time.new * trust_factor < iteration
        and acor_time.relative_diff < relative_thresh
    )


def _get_columns(it: int = 0) -> list[ProgressColumn]:
    """Get the default progress columns for the MCMC sampling."""
    return [
        *Progress.get_default_columns(),
        ItersPerSecondColumn(),
        CompletedItersColumn(it=it),
        TimeElapsedColumn(),
    ]


def run_sampling(
    sampler: emcee.EnsembleSampler,
    initial_state: np.ndarray | None = None,
    num_steps: int | None = None,
    thin_by: int = 1,
    check_interval: int = 100,
    trust_factor: float = 50.0,
    relative_thresh: float = 0.05,
    history_file: Path | None = None,
    reset_backend: bool = False,
    description: str = "Burn-in phase",
) -> None:
    """Run MCMC sampling.

    This will run the ``sampler`` either for ``num_steps`` steps or - if it set to
    ``None`` - until convergence. Convergence is determined once within a
    ``check_interval`` of steps by the :py:func:`is_converged` function. The
    convergence criterion is based on a trustworthy estimate of the autocorrelation
    time. This is elaborated in the `emcee documentation`_.

    Some bookkeeping parameters may be stored in a ``history_file``. During sampling,
    the history is stored in a temporary file with the suffix ``.tmp``. If the sampling
    is interrupted, the history and the last state of the ``sampler`` can be recovered
    and the sampling can be continued.

    One may choose to ``reset_backend``, e.g. in case the previous sampling was run
    until convergence and now one wants to store a length of the converged chain. This
    may also be thinned by a factor of ``thin_by`` (directly passed to the
    :py:class:`emcee.EnsembleSampler` class).

    .. _emcee documentation: https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
    """
    state = initial_state or ensure_initial_state(sampler)
    history = ensure_history_table(history_file)

    if reset_backend:
        logger.debug("Resetting backend of sampler.")
        sampler.backend.reset(sampler.nwalkers, sampler.ndim)

    acor_time = AcorTime(old=np.inf, new=np.inf)
    accepted = NumAccepted(old=0, new=sampler.backend.accepted.sum())

    with Progress(*_get_columns(it=sampler.iteration), console=console) as progress:
        task = progress.add_task(description=description, total=num_steps)
        while sampler.iteration < (num_steps or np.inf) * thin_by:
            for state in sampler.sample(  # noqa: B007, B020
                initial_state=state,
                iterations=check_interval - sampler.iteration % check_interval,
                thin_by=thin_by,
            ):
                progress.update(task, advance=1)

            acor_time.update(new=sampler.get_autocorr_time(tol=0).mean())
            accepted.update(new=sampler.backend.accepted.sum())

            history = update_history_table(
                history=history,
                history_file=history_file,
                iteration=sampler.iteration,
                acor_time=acor_time.new,
                accepted_frac=(
                    accepted.newly_accepted / (check_interval * sampler.nwalkers)
                ),
                max_log_prob=np.max(state.log_prob),
            )

            if num_steps is None and is_converged(
                iteration=sampler.iteration,
                acor_time=acor_time,
                trust_factor=trust_factor,
                relative_thresh=relative_thresh,
            ):
                logger.info(f"Sampling converged after {sampler.iteration} steps.")
                break

    if history_file is not None:
        history_file.with_suffix(".tmp").rename(history_file)


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
        nwalkers=nwalkers,
        ndim=ndim,
        log_prob_fn=log_prob_fn,
        kwargs={"inverse_temp": settings.sampling.inverse_temp},
        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
        backend=backend,
        pool=pool,
        blobs_dtype=[("inverse_temp", np.float64)],
        parameter_names=settings.sampling.param_names,
    )


class SampleCLI(BaseCLI):
    """Use MCMC to infer distributions over model parameters from data."""

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

    def cli_cmd(self) -> None:
        """Start the ``sample`` subcommand.

        First, it will construct the model from the ``graph`` and ``model`` arguments.
        Then, it will add distributions over diagnose times via the dictionary from
        the ``distributions`` argument. It will also set sensitivity and specificity of
        diagnostic modalities via the dictionary provided through the ``modalities``
        argument. Finally, it will load the patient data as specified via the ``data``
        argument.

        When the model is constructed, an :py:class:`emcee.EnsembleSampler` is
        initialied (see :py:func:`init_sampler`) and :py:func:`run_sampling` is executed
        twice: once for the burn-in phase and once for the actual sampling phase.
        The ``sampling`` argument provides all necessary settings for the sampling.
        """
        # as recommended in https://emcee.readthedocs.io/en/stable/tutorials/parallel/#
        os.environ["OMP_NUM_THREADS"] = "1"

        logger.debug(self.model_dump_json(indent=2))

        # ugly, but necessary for pickling
        global MODEL
        MODEL = construct_model(self.model, self.graph)
        MODEL = add_dists(MODEL, self.distributions)
        MODEL = add_modalities(MODEL, self.modalities)
        MODEL.load_patient_data(**self.data.get_load_kwargs())
        ndim = (
            len(self.sampling.param_names)
            if self.sampling.param_names is not None
            else MODEL.get_num_dims()
        )

        # emcee does not support numpy's new random number generator yet.
        np.random.seed(self.sampling.seed)

        with get_pool(self.sampling.cores) as pool:
            sampler = init_sampler(settings=self, ndim=ndim, pool=pool)
            run_sampling(
                description="Burn-in phase",
                sampler=sampler,
                check_interval=self.sampling.check_interval,
                trust_factor=self.sampling.trust_factor,
                relative_thresh=self.sampling.relative_thresh,
                history_file=self.sampling.history_file,
            )
            run_sampling(
                description="Sampling phase",
                sampler=sampler,
                num_steps=self.sampling.num_steps,
                reset_backend=True,
                thin_by=self.sampling.thin_by,
            )


if __name__ == "__main__":
    main = assemble_main(settings_cls=SampleCLI, prog_name="sample")
    main()
