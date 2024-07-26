"""Learn the model params from preprocessed input data using MCMC sampling.

This command allows us to infer the parameters of a predefined probabilistic model
from detailed per-patient lymph node level involvement data.

The model, data, and sampling configuration can be specified in one or several YAML
files, and/or via command line arguments.
"""

import argparse
import logging
import os
from typing import Any

try:
    from multiprocess import Pool
except ModuleNotFoundError:
    from multiprocessing import Pool

from pathlib import Path

import emcee
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
)
from rich.progress import Progress, TimeElapsedColumn, track

from lyscripts.configs import (
    DataConfig,
    DistributionConfig,
    GraphConfig,
    ModalityConfig,
    ModelConfig,
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


class SamplingConfig(BaseModel):
    """Settings to configure the MCMC sampling."""

    output_file: Path = Field(
        description="Path to the HDF5 file to store the results in."
    )
    history_file: Path | None = Field(
        default=None,
        description="Path to store the burn-in history in (as CSV file).",
    )
    cores: int | None = Field(
        gt=0,
        default=os.cpu_count(),
        description=(
            "Number of cores to use for parallel sampling. If `None`, no parallel "
            "processing is used."
        ),
    )
    seed: int = Field(
        default=42,
        description="Seed for the random number generator.",
    )
    walkers_per_dim: int = Field(
        default=20,
        description="Number of walkers per parameter space dimension.",
    )
    max_burnin: int | None = Field(
        default=None,
        description="Maximum number of burn-in steps.",
    )
    check_interval: int = Field(
        default=50,
        description="Check for convergence each time after this many steps.",
    )
    trust_factor: float = Field(
        default=50.0,
        description=(
            "Trust the autocorrelation time only when it's smaller than this factor "
            "times the length of the chain."
        ),
    )
    relative_thresh: float = Field(
        default=0.05,
        description="Relative threshold for convergence.",
    )
    thin: int = Field(
        default=10, description="How many samples to draw before for saving one."
    )
    nsteps: int = Field(
        default=100,
        description="Number of samples after convergence, regardless of thinning.",
    )


class CmdSettings(BaseSettings):
    """Settings required for the MCMC sampling."""

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


MODEL = None


def log_prob_fn(theta: np.array) -> float:
    """Log probability function using global variables because of pickling."""
    return MODEL.likelihood(given_params=theta)


def get_starting_state(sampler):
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

    return state


def get_burnin_history(file: Path | None) -> pd.DataFrame:
    """Try to load the burn-in history from a CSV file.

    Return an empty DataFrame if no history is found.
    """
    if file is None or not file.exists():
        history = pd.DataFrame(
            columns=["steps", "acor_times", "accept_fracs", "max_log_probs"],
        ).set_index("steps")
    else:
        history = pd.read_csv(file, index_col="steps")

    return history


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
    history_df = get_burnin_history(history_file)
    previous_accepted = 0

    with Progress(
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            description="[blue]INFO     [/blue]Burn-in phase ",
            total=max_burnin,
        )
        while sampler.iteration < (max_burnin or np.inf):
            for state in sampler.sample(state, iterations=check_interval):  # noqa: B007, B020
                progress.update(task, advance=1)

            new_acor_time = sampler.get_autocorr_time(tol=0).mean()
            old_acor_time = (
                history_df.iloc[-1].acor_times if len(history_df) > 0 else np.inf
            )

            newly_accepted = np.sum(sampler.backend.accepted) - previous_accepted
            new_accept_frac = newly_accepted / (sampler.nwalkers * check_interval)
            previous_accepted = np.sum(sampler.backend.accepted)

            history_df.loc[sampler.iteration] = [
                new_acor_time,
                new_accept_frac,
                np.max(state.log_prob),
            ]
            logger.debug(history_df.iloc[-1].to_dict())
            if history_file is not None:
                history_df.to_csv(history_file, index=True)

            if max_burnin is None and is_converged(
                iteration=sampler.iteration,
                new_acor_time=new_acor_time,
                old_acor_time=old_acor_time,
                trust_factor=trust_factor,
                relative_thresh=relative_thresh,
            ):
                logger.info(f"Sampling converged after {sampler.iteration} steps.")
                break


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
        continue


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


def main(args: argparse.Namespace) -> None:
    """Run the MCMC sampling."""
    # as recommended in https://emcee.readthedocs.io/en/stable/tutorials/parallel/#
    os.environ["OMP_NUM_THREADS"] = "1"

    yaml_configs = merge_yaml_configs(args.configs)

    settings = CmdSettings(
        _cli_settings_source=args.cli_settings_source(parsed_args=args),
        **yaml_configs,
    )
    logger.debug(settings.model_dump_json(indent=2))

    # ugly, but necessary for pickling
    global MODEL
    MODEL = construct_model(settings.model, settings.graph)
    MODEL = add_dists(MODEL, settings.distributions)
    MODEL = add_modalities(MODEL, settings.modalities)
    MODEL.load_patient_data(**settings.data.get_load_kwargs())

    # emcee does not support numpy's new random number generator yet.
    np.random.seed(settings.sampling.seed)
    ndim = MODEL.get_num_dims()
    nwalkers = ndim * settings.sampling.walkers_per_dim

    with get_pool(settings.sampling.cores) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_prob_fn,
            moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
            backend=get_hdf5_backend(settings.sampling.output_file, nwalkers, ndim),
            pool=pool,
        )
        kwargs = {
            "max_burnin",
            "check_interval",
            "trust_factor",
            "relative_thresh",
            "history_file",
        }
        run_burnin(
            sampler,
            **settings.sampling.model_dump(include=kwargs),
        )
        run_sampling(
            sampler,
            **settings.sampling.model_dump(include={"nsteps", "thin"}),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
