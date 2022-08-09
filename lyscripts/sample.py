"""
Learn the spread probabilities of the HMM for lymphatic tumor progression using
the preprocessed data as input and MCMC as sampling method.
"""
import argparse
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import emcee
import h5py
import lymph
import numpy as np
import pandas as pd
import yaml
from rich.progress import track
from scipy.special import factorial

from .helpers import ConsoleReport, get_graph_from_, report


class ConvenienceSampler(emcee.EnsembleSampler):
    """Class that adds some useful defaults to the `emcee.EnsembleSampler`."""
    def __init__(
        self,
        nwalkers,
        ndim,
        log_prob_fn,
        pool=None,
        backend=None,
    ):
        """Initialize sampler with sane defaults."""
        moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
        super().__init__(nwalkers, ndim, log_prob_fn, pool, moves, backend=backend)

    def run_sampling(
        self,
        min_steps: int = 0,
        max_steps: int = 10000,
        initial_state: Optional[Union[emcee.State, np.ndarray]] = None,
        check_interval: int = 100,
        trust_threshold: float = 50.,
        rel_acor_threshold: float = 0.05,
        report: Optional[ConsoleReport] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run a round of sampling with at least `min_steps` and at most `max_steps`.

        Every `check_interval` iterations, convergence is checked based on
        three criteria:
        - did it run for at least `min_steps`?
        - has the acor time crossed the N / `trust_threshold`?
        - did the acor time change less than `rel_acor_threshold`?

        If a `ConsoleReport` is provided, progress will be displayed.

        Returns dictionary containing the autocorrelation times obtained during the
        run (along with the iteration number at which they were computed) and the
        numpy array with the coordinates of the last draw.
        """
        if max_steps < min_steps:
            warnings.warn(
                "Sampling param min_steps is larger than max_steps. Swapping."
            )
            tmp = max_steps
            max_steps = min_steps
            min_steps = tmp

        if initial_state is None:
            initial_state = np.random.uniform(
                low=0., high=1.,
                size=(self.nwalkers, self.ndim)
            )

        iterations = []
        acor_times = []
        old_acor = np.inf
        is_converged = False

        samples_iterator = self.sample(
            initial_state=initial_state,
            iterations=max_steps,
            **kwargs
        )
        if report is not None:
            samples_iterator = track(
                sequence=samples_iterator,
                total=max_steps,
                description=description,
                console=report,
            )
        coords = None
        for sample in samples_iterator:
            # after `check_interval` number of samples...
            coords = sample.coords
            if self.iteration < min_steps or self.iteration % check_interval:
                continue

            # ...compute the autocorrelation time and store it in an array.
            new_acor = self.get_autocorr_time(tol=0)
            iterations.append(self.iteration)
            acor_times.append(np.mean(new_acor))

            # check convergence
            is_converged = self.iteration >= min_steps
            is_converged &= np.all(new_acor * trust_threshold < self.iteration)
            rel_acor_diff = np.abs(old_acor - new_acor) / new_acor
            is_converged &= np.all(rel_acor_diff < rel_acor_threshold)

            # if it has converged, stop
            if is_converged:
                break

            old_acor = new_acor

        accept_rate = 100. * np.mean(self.acceptance_fraction)
        if report is not None:
            if is_converged:
                report.success(
                    description,
                    f"converged after {self.iteration} steps with "
                    f"an acceptance rate of {accept_rate:.2f}%"
                )
            else:
                report.info(
                    description,
                    f"finished: Max. number of steps ({max_steps}) reached "
                    f"(acceptance rate {accept_rate:.2f}%)"
                )

        return {
            "iterations": iterations,
            "acor_times": acor_times,
            "final_state": coords,
        }

def run_mcmc_with_burnin(
    nwalkers: int,
    ndim: int,
    log_prob_fn: Callable,
    nsteps: int,
    burnin: int,
    persistent_backend: emcee.backends.HDFBackend,
    keep_burnin: bool = False,
    report: Optional[ConsoleReport] = None,
):
    """
    Draw samples from the `log_prob_fn` using the `ConvenienceSampler` (subclass of
    `emcee.EnsembleSampler`).

    This function first draws `burnin` (that are discarded if `keep_burnin` is set to
    `False`) and afterwards it performs a second sampling round, starting where the
    burnin left off, of length `nsteps - burnin` that will be stored in the
    `persistent_backend`.

    If `report` is given, the progress will be displayed.
    """
    with Pool() as pool:
        # Burnin phase
        if keep_burnin:
            burnin_backend = persistent_backend
        else:
            burnin_backend = emcee.backends.Backend()
        burnin_sampler = ConvenienceSampler(
            nwalkers, ndim, log_prob_fn,
            pool=pool, backend=burnin_backend,
        )
        burnin_result = burnin_sampler.run_sampling(
            max_steps=burnin,
            check_interval=burnin+1,  # this makes sure convergence is never checked
            report=report,
            description="Burn-in ",
        )

        # real sampling phase
        sampler = ConvenienceSampler(
            nwalkers, ndim, log_prob_fn,
            pool=pool, backend=persistent_backend,
        )
        sampler.run_sampling(
            max_steps=nsteps-burnin,
            initial_state=burnin_result["final_state"],
            check_interval=nsteps-burnin,
            report=report,
            description="Sampling"
        )


def binom_pmf(k: Union[List[int], np.ndarray], n: int, p: float):
    """Binomial PMF"""
    if p > 1. or p < 0.:
        raise ValueError("Binomial prob must be btw. 0 and 1")
    q = (1. - p)
    binom_coeff = factorial(n) / (factorial(k) * factorial(n - k))
    return binom_coeff * p**k * q**(n - k)

def parametric_binom_pmf(n: int) -> Callable:
    """Return a parametric binomial PMF"""
    def inner(t, p):
        """Parametric binomial PMF"""
        return binom_pmf(t, n, p)
    return inner

def add_tstage_marg(
    model: Union[lymph.Unilateral, lymph.Bilateral, lymph.MidlineBilateral],
    t_stages: List[str],
    first_binom_prob: float,
    max_t: int,
):
    """Add margializors over diagnose times to `model`."""
    for i,stage in enumerate(t_stages):
        if i == 0:
            model.diag_time_dists[stage] = binom_pmf(
                k=np.arange(max_t + 1),
                n=max_t,
                p=first_binom_prob
            )
        else:
            model.diag_time_dists[stage] = parametric_binom_pmf(n=max_t)


def setup_model(
    graph_params: Dict[str, Any],
    model_params: Dict[str, Any],
    modalities_params: Optional[Dict[str, Any]] = None,
) -> Union[lymph.Unilateral, lymph.Bilateral, lymph.MidlineBilateral]:
    """Create a model instance as defined by some YAML params."""
    graph = get_graph_from_(graph_params)

    model_cls = getattr(lymph, model_params["class"])
    model = model_cls(graph, **model_params["kwargs"])

    if modalities_params is not None:
        model.modalities = modalities_params

    add_tstage_marg(
        model,
        t_stages=model_params["t_stages"],
        first_binom_prob=model_params["first_binom_prob"],
        max_t=model_params["max_t"],
    )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to training data files"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path to the HDF5 file to store the results in"
    )
    parser.add_argument(
        "--params", default="params.yaml",
        help="Path to parameter YAML file"
    )
    parser.add_argument(
        "--ti", action="store_true",
        help="Use thermodynamic integration from prior to given likelihood"
    )

    # Parse arguments and prepare paths
    args = parser.parse_args()
    input_path = Path(args.input)
    params_path = Path(args.params)

    with report.status("Read in parameters..."):
        with open(params_path, mode='r') as params_file:
            params = yaml.safe_load(params_file)
        report.success(f"Read in params from {params_path}")

    with report.status("Read in training data..."):
        # Only read in two header rows when using the Unilateral model
        is_unilateral = params["model"]["class"] == "Unilateral"
        header = [0, 1] if is_unilateral else [0, 1, 2]
        inference_data = pd.read_csv(input_path, header=header)
        report.success(f"Read in training data from {input_path}")


    with report.status("Set up model & load data..."):
        MODEL = setup_model(
            graph_params=params["graph"],
            model_params=params["model"],
            modalities_params=params["modalities"],
        )
        MODEL.patient_data = inference_data
        ndim = len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric
        nwalkers = ndim * params["sampling"]["walkers_per_dim"]
        report.success(
            f"Set up {type(MODEL)} model with {ndim} parameters and loaded "
            f"{len(inference_data)} patients"
        )

    if args.ti:
        with report.status("Prepare thermodynamic integration..."):
            # make sure path to output file exists
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # set up sampling params
            ladder = np.logspace(0., 1., num=params["sampling"]["len_ladder"])
            ladder = (ladder - 1.) / 9.
            nsteps = params["sampling"]["kwargs"]["max_steps"] // len(ladder)
            burnin = nsteps - params["sampling"]["keep_steps"]
            report.success("Prepared thermodynamic integration.")

        for i,inv_temp in enumerate(ladder):
            report.print(f"TI round {i+1}/{len(ladder)} with β = {inv_temp:.3f}")

            # set up backend
            hdf5_backend = emcee.backends.HDFBackend(output_path, name=f"ti/{i+1:0>2d}")

            # create log-probability function
            def log_prob_fn(theta):
                """Compute log-probability of parameter sample `theta`."""
                llh = MODEL.likelihood(given_params=theta, log=True)
                if np.isinf(llh):
                    return -np.inf, -np.inf
                return inv_temp * llh, llh

            run_mcmc_with_burnin(
                nwalkers, ndim, log_prob_fn, nsteps, burnin,
                persistent_backend=hdf5_backend,
                keep_burnin=False,
                report=report,
            )

        # copy last sampling round over to a group in the HDF5 file called "mcmc"
        # because that is what other scripts expect to see
        h5_file = h5py.File(output_path, "r+")
        h5_file.copy(f"ti/{len(ladder):0>2d}", h5_file, name="mcmc")
        report.success("Finished thermodynamic integration.")

    else:
        with report.status("Prepare sampling params & backend..."):
            # make sure path to output file exists
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # set up sampling params
            ndim = len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric
            nwalkers = ndim * params["sampling"]["walkers_per_dim"]

            # prepare backend
            hdf5_backend = emcee.backends.HDFBackend(output_path, name="mcmc")

            def log_prob_fn(theta,) -> float:
                """
                Compute the log-probability of data loaded in `model`, given the
                parameters `theta`.
                """
                return MODEL.likelihood(given_params=theta, log=True)

            report.success(f"Prepared sampling params & backend at {output_path}")

        with Pool() as pool:
            sampler = ConvenienceSampler(
                nwalkers, ndim, log_prob_fn,
                pool=pool,
                backend=hdf5_backend,
            )
            result = sampler.run_sampling(
                report=report,
                description="Sampling"
                **params["sampling"]["kwargs"]
            )
