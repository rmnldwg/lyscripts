"""Learn the model params using the preprocessed input data and MCMC sampling.

This is the central script performing for our project on modelling lymphatic spread
in head & neck cancer. We use it for model comparison via the thermodynamic
integration functionality and use the sampled parameter estimates for risk
predictions. This risk estimate may in turn some day guide clinicians to make more
objective decisions with respect to defining the *elective clinical target volume*
(CTV-N) in radiotherapy.
"""

import argparse
import logging
import os
import sys

try:
    from multiprocess import Pool
except ModuleNotFoundError:
    from multiprocessing import Pool

from pathlib import Path

import emcee
import numpy as np
import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings, CliSettingsSource
from rich.progress import Progress, TimeElapsedColumn, track

from lyscripts.configs import (
    DataConfig,
    DistributionConfig,
    GraphConfig,
    ModalityConfig,
    ModelConfig,
    SamplingConfig,
    add_dists,
    add_modalities,
    construct_model,
)
from lyscripts.utils import (
    initialize_backend,
    load_yaml_params,
)

logger = logging.getLogger(__name__)


class SamplingSettings(BaseSettings):
    """Settings required for the MCMC sampling."""

    graph: GraphConfig = GraphConfig()
    model: ModelConfig = ModelConfig()
    distributions: dict[str, DistributionConfig] = Field(
        default={},
        description="Distributions over diagnosis times.",
    )
    modalities: dict[str, ModalityConfig] = Field(
        default={},
        description="Diagnostic modalities to use in the model.",
    )
    data: DataConfig
    sampling: SamplingConfig = SamplingConfig()


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
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to the HDF5 file to store the results in",
    )
    parser.add_argument(
        "--history",
        type=Path,
        nargs="?",
        help="Path to store the burn-in history in (as CSV file).",
    )
    parser.add_argument(
        "-p",
        "--params",
        default=[],
        nargs="*",
        help=(
            "Path to parameter file(s). Subsequent files overwrite previous ones. "
            "Command line arguments take precedence over all files."
        ),
    )

    parser.set_defaults(
        run_main=main,
        cli_settings_source=CliSettingsSource(
            settings_cls=SamplingSettings,
            root_parser=parser,
        ),
    )


MODEL = None


def log_prob_fn(theta: np.array) -> float:
    """Log probability function using global variables because of pickling."""
    return MODEL.likelihood(given_params=theta)


def get_starting_state(sampler):
    """Try to extract a starting state from a `sampler`."""
    try:
        state = sampler.backend.get_last_sample()
        logger.info(
            f"Resuming from {sampler.backend.filename} with {sampler.iteration} "
            "stored iterations."
        )
    except AttributeError:
        state = np.random.uniform(size=(sampler.nwalkers, sampler.ndim))

    return state


def init_burnin_history():
    """Initialize the burnin history DataFrame."""
    return pd.DataFrame(
        columns=["steps", "acor_times", "accept_fracs", "max_log_probs"],
    ).set_index("steps")


def is_converged(
    iteration: int,
    new_acor_time: float,
    old_acor_time: float,
    trust_factor: float,
    relative_thresh: float,
) -> bool:
    """Check if the chain has converged based on the autocorrelation time."""
    return (
        new_acor_time * trust_factor
        < iteration & np.abs(new_acor_time - old_acor_time) / new_acor_time
        < relative_thresh
    )


def run_burnin(
    sampler: emcee.EnsembleSampler,
    max_burnin: int | None = None,
    check_interval: int = 100,
    trust_factor: float = 50.0,
    relative_thresh: float = 0.05,
) -> pd.DataFrame:
    """Run the burnin phase of the MCMC sampling.

    This will run the sampler for ``burnin`` steps or (if ``burnin`` is `None`) until
    convergence is reached. The convergence criterion is based on the autocorrelation
    time of the chain, which is computed every `check_interval` steps. The chain is
    considered to have converged if the autocorrelation time is smaller than
    `trust_fac` times the number of iterations and the relative change in the
    autocorrelation time is smaller than `rel_thresh`.

    The samples of the burnin phase will be stored, such that one can resume a
    cancelled run. Also, metrics collected during the burnin phase will be returned
    in a pandas DataFrame.
    """
    state = get_starting_state(sampler)
    history = init_burnin_history()
    previous_accepted = 0

    with Progress(
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            description="[blue]INFO     [/blue]Burn-in phase ",
            total=max_burnin,
        )
        while sampler.iteration < (max_burnin or np.inf):
            logger.debug("| step   | acor time | accept frac | max log prob |")
            logger.debug("| -----: | --------: | ----------: | -----------: |")

            for state in sampler.sample(state, iterations=check_interval):  # noqa: B007, B020
                progress.update(task, advance=1)

            new_acor_time = sampler.get_autocorr_time(tol=0).mean()
            old_acor_time = history.acor_times[-1] if len(history) > 0 else np.inf

            newly_accepted = np.sum(sampler.backend.accepted) - previous_accepted
            new_accept_frac = newly_accepted / (sampler.nwalkers * check_interval)
            previous_accepted = np.sum(sampler.backend.accepted)

            history.loc[sampler.iteration] = [
                new_acor_time,
                new_accept_frac,
                np.max(state.log_prob),
            ]
            logger.debug(
                f"| {sampler.iteration:>6d} "
                f"| {new_acor_time:>9.2f} "
                f"| {new_accept_frac:>11.2%} "
                f"| {np.max(state.log_prob):>12.2f} |"
            )

            if max_burnin is None and is_converged(
                iteration=sampler.iteration,
                new_acor_time=new_acor_time,
                old_acor_time=old_acor_time,
                trust_factor=trust_factor,
                relative_thresh=relative_thresh,
            ):
                break

    return history


def run_sampling(
    sampler: emcee.EnsembleSampler,
    nsteps: int,
    thin: int,
) -> None:
    """Run the MCMC sampling phase to produce `nsteps` samples.

    This sampling will definitely produce `nsteps` samples, irrespective of the `thin`
    parameter, which controls how many steps in between two stored samples are skipped.
    The samples will be stored in the backend of the `sampler`.

    Note that this will reset the `sampler`'s backend, assuming the stored samples are
    from the burnin phase.
    """
    state = get_starting_state(sampler)
    sampler.backend.reset(sampler.nwalkers, sampler.ndim)

    for _sample in track(
        sequence=sampler.sample(state, iterations=nsteps * thin, thin=thin, store=True),
        description="[blue]INFO     [/blue]Sampling phase",
        total=nsteps * thin,
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


def main(args: argparse.Namespace) -> None:
    """Run the MCMC sampling."""
    # as recommended in https://emcee.readthedocs.io/en/stable/tutorials/parallel/#
    os.environ["OMP_NUM_THREADS"] = "1"

    params = {}
    for param_file in args.params:
        params.update(load_yaml_params(param_file))

    # despite using a subparser, pydantic will still consume `sys.argv[1:]`, meaning
    # that the subcommand will be interpreted as an argument. To avoid this, we have to
    # manually remove the first argument.
    sys.argv = sys.argv[1:]
    settings = SamplingSettings(
        _cli_parse_args=True,
        _cli_use_class_docs_for_groups=True,
        _cli_settings_source=args.cli_settings_source(args=True),
        **params,
    )
    logger.info(settings.model_dump_json(indent=2))

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

    if args.cores == 0:
        real_or_dummy_pool = DummyPool()
    else:
        real_or_dummy_pool = Pool(args.cores)

    with real_or_dummy_pool as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_prob_fn,
            moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
            backend=initialize_backend(args.output, nwalkers, ndim),
            pool=pool,
        )
        kwargs_set = {"max_burnin", "check_interval", "trust_factor", "relative_thresh"}
        burnin_history = run_burnin(
            sampler,
            **settings.sampling.model_dump(include=kwargs_set),
        )
        run_sampling(sampler, nsteps=args.nsteps, thin=args.thin)

    if args.history is not None:
        logger.info(f"Saving burn-in history to {args.history}.")
        burnin_history.to_csv(args.history, index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
