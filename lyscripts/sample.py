"""
Learn the spread probabilities of the HMM for lymphatic tumor progression using
the preprocessed data as input and MCMC as sampling method.

This is the central script performing for our project on modelling lymphatic spread
in head & neck cancer. We use it for model comparison via the thermodynamic
integration functionality and use the sampled parameter estimates for risk
predictions. This risk estimate may in turn some day guide clinicians to make more
objective decisions with respect to defining the *elective clinical target volume*
(CTV-N) in radiotherapy.
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import logging
import os
from multiprocessing import Pool
from pathlib import Path

import emcee
import numpy as np
import pandas as pd

from lyscripts.decorators import log_state
from lyscripts.utils import (
    create_model,
    initialize_backend,
    load_patient_data,
    load_yaml_params,
)

logger = logging.getLogger(__name__)


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """
    Add an `ArgumentParser` to the subparsers action.
    """
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """
    Add arguments needed to run this script to a `subparsers` instance
    and run the respective main function when chosen.
    """
    parser.add_argument(
        "--input", type=Path,
        help="Path to training data files"
    )
    parser.add_argument(
        "--output", type=Path,
        help="Path to the HDF5 file to store the results in"
    )

    parser.add_argument(
        "--walkers-per-dim", type=int, default=10,
        help="Number of walkers per dimension",
    )
    parser.add_argument(
        "--burnin", type=int, nargs="?",
        help="Number of burnin steps. If not provided, sampler runs until convergence."
    )
    parser.add_argument(
        "--nsteps", type=int, default=100,
        help="Number of MCMC steps to run"
    )
    parser.add_argument(
        "--params", default="./params.yaml", type=Path,
        help="Path to parameter file"
    )
    parser.add_argument(
        "--modalities", nargs="+",
        default=["max_llh"],
        help="List of modalities for inference. Must be defined in `params.yaml`"
    )
    parser.add_argument(
        "--plots", default="./plots", type=Path,
        help="Directory to store plot of acor times",
    )
    parser.add_argument(
        "--ti", action="store_true",
        help="Perform thermodynamic integration"
    )
    parser.add_argument(
        "--cores", type=int, nargs="?",
        help=(
            "Number of parallel workers (CPU cores/threads) to use. If not provided, "
            "it will use all cores. If set to zero, multiprocessing will not be used."
        )
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed value to reproduce the same sampling round."
    )

    parser.set_defaults(run_main=main)


MODEL = None
INV_TEMP = 1.

def log_prob_fn(theta: np.array) -> float:
    """log probability function using global variables because of pickling."""
    llh = MODEL.likelihood(given_param_args=theta)
    if np.isinf(llh):
        return -np.inf, -np.inf
    return INV_TEMP * llh, llh


@log_state()
def run_burnin(
    sampler: emcee.EnsembleSampler,
    burnin: int | None,
    check_interval: int = 100,
) -> list[float]:
    """Run the burnin phase of the MCMC sampling."""
    try:
        state = sampler.backend.get_last_sample()
        logger.info(f"Resuming after {sampler.iteration} iterations.")
    except AttributeError:
        state = np.random.uniform(size=(sampler.nwalkers, sampler.ndim))
    acor_times = []

    for _sample in sampler.sample(state, iterations=burnin):
        if sampler.iteration % check_interval != 0:
            continue

        new_acor_time = sampler.get_autocorr_time(tol=0).mean()
        old_acor_time = acor_times[-1] if len(acor_times) > 0 else np.inf
        acor_times.append(new_acor_time)

        is_converged = burnin is not None
        is_converged &= new_acor_time * 100 < sampler.iteration
        is_converged &= np.abs(new_acor_time - old_acor_time) / new_acor_time < 0.05

        if is_converged:
            logger.info(f"Converged after {sampler.iteration} iterations.")
            break

    return acor_times


@log_state()
def run_sampling(
    sampler: emcee.EnsembleSampler,
    nsteps: int,
) -> None:
    """Run the MCMC sampling phase."""
    sampler.run_mcmc(sampler.backend.get_last_sample(), nsteps)


def main(args: argparse.Namespace) -> None:
    """Main function to run the MCMC sampling."""
    # as recommended in https://emcee.readthedocs.io/en/stable/tutorials/parallel/#
    os.environ["OMP_NUM_THREADS"] = "1"

    params = load_yaml_params(args.params)
    inference_data = load_patient_data(args.input)

    # ugly, but necessary for pickling
    global MODEL
    MODEL = create_model(params)

    mapping = params["model"].get("mapping", None)
    MODEL.load_patient_data(inference_data, mapping=mapping)

    sampling_config = params.get("sampling", {})
    ndim = MODEL.get_num_dims()
    nwalkers = ndim * sampling_config.get("walkers_per_dim", args.walkers_per_dim)
    thin_by = sampling_config.get("thin_by", 1)

    # emcee does not support numpy's new random number generator yet.
    np.random.seed(args.seed)
    hdf5_backend = initialize_backend(args.output, nwalkers, ndim)
    moves_mix = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]

    with Pool(args.cores) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob_fn,
            moves=moves_mix,
            backend=hdf5_backend,
            pool=pool,
        )
        acor_times = run_burnin(sampler, burnin=args.burnin)
        run_sampling(sampler, nsteps=args.nsteps)

    x_axis = np.arange(sampler.iteration)
    plots = {
        "acor_times": acor_times,
        "accept_rates": sampler.acceptance_fraction,
    }

    args.plots.mkdir(parents=True, exist_ok=True)

    for name, y_axis in plots.items():
        tmp_df = pd.DataFrame(
            np.array([x_axis, y_axis]).T,
            columns=["x", name],
        )
        tmp_df.to_csv(args.plots/(name + ".csv"), index=False)

    logger.info(f"Stored {len(plots)} plots about burnin phases at {args.plots}")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
