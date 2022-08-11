"""
Learn the spread probabilities of the HMM for lymphatic tumor progression using
the preprocessed data as input and MCMC as sampling method.
"""
import argparse
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import emcee
import h5py
import numpy as np
import pandas as pd
import yaml

from .helpers import CustomProgress, model_from_config, report


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
        progress_desc: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run a round of sampling with at least `min_steps` and at most `max_steps`.

        Every `check_interval` iterations, convergence is checked based on
        three criteria:
        - did it run for at least `min_steps`?
        - has the acor time crossed the N / `trust_threshold`?
        - did the acor time change less than `rel_acor_threshold`?

        If a `progress_desc` is provided, progress will be displayed.

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
        # wrap the iteration over samples in a rich tracker,
        # if verbosity is desired
        if progress_desc is not None:
            report_progress = CustomProgress(console=report)
            samples_iterator = report_progress.track(
                sequence=samples_iterator,
                total=max_steps,
                description=progress_desc,
            )
            report_progress.start()

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

        iterations.append(self.iteration)
        acor_times.append(np.mean(self.get_autocorr_time(tol=0)))

        accept_rate = 100. * np.mean(self.acceptance_fraction)
        accept_rate_str = f"acceptance rate was {accept_rate:.2f}%"

        if progress_desc is not None:
            report_progress.stop()
            if is_converged:
                report.success(
                    progress_desc, "converged,", accept_rate_str
                )
            else:
                report.info(
                    progress_desc, "finished: Max. steps reached,", accept_rate_str
                )

        return {
            "iterations": iterations,
            "acor_times": acor_times,
            "accept_rate": accept_rate,
            "final_state": coords,
        }

def run_mcmc_with_burnin(
    nwalkers: int,
    ndim: int,
    log_prob_fn: Callable,
    nsteps: int,
    persistent_backend: emcee.backends.HDFBackend,
    sampling_kwargs: Optional[dict] = None,
    burnin: Optional[int] = None,
    keep_burnin: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Draw samples from the `log_prob_fn` using the `ConvenienceSampler` (subclass of
    `emcee.EnsembleSampler`).

    This function first draws `burnin` samples (that are discarded if `keep_burnin`
    is set to `False`) and afterwards it performs a second sampling round, starting
    where the burnin left off, of length `nsteps` that will be stored in the
    `persistent_backend`.

    When `burnin` is not given, the burnin phase will still take place and it will
    sample until convergence using convergence criteria defined via `sampling_kwargs`,
    after which it will draw another `nsteps` samples.

    If `verbose` is set to `True`, the progress will be displayed.

    Returns a dictionary with some information about the burnin phase.
    """
    if sampling_kwargs is None:
        sampling_kwargs = {}

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

        if burnin is None:
            burnin_result = burnin_sampler.run_sampling(
                progress_desc="Burn-in " if verbose else None,
                **sampling_kwargs,
            )
        else:
            burnin_result = burnin_sampler.run_sampling(
                min_steps=burnin,
                max_steps=burnin,
                progress_desc="Burn-in " if verbose else None,
            )

        # persistent sampling phase
        sampler = ConvenienceSampler(
            nwalkers, ndim, log_prob_fn,
            pool=pool, backend=persistent_backend,
        )
        sampler.run_sampling(
            min_steps=nsteps,
            max_steps=nsteps,
            initial_state=burnin_result["final_state"],
            progress_desc="Sampling" if verbose else None,
        )

        return burnin_result


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
        "--plots", default="plots",
        help="Directory to store plot of acor times",
    )
    parser.add_argument(
        "--ti", action="store_true",
        help="Perform thermodynamic integration"
    )

    # Parse arguments and prepare paths
    args = parser.parse_args()

    with report.status("Read in parameters..."):
        params_path = Path(args.params)
        with open(params_path, mode='r') as params_file:
            params = yaml.safe_load(params_file)
        report.success(f"Read in params from {params_path}")

    with report.status("Read in training data..."):
        input_path = Path(args.input)
        # Only read in two header rows when using the Unilateral model
        is_unilateral = params["model"]["class"] == "Unilateral"
        header = [0, 1] if is_unilateral else [0, 1, 2]
        DATA = pd.read_csv(input_path, header=header)
        report.success(f"Read in training data from {input_path}")

    with report.status("Set up model & load data..."):
        MODEL = model_from_config(
            graph_params=params["graph"],
            model_params=params["model"],
            modalities_params=params["modalities"],
        )
        MODEL.patient_data = DATA
        ndim = len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric
        nwalkers = ndim * params["sampling"]["walkers_per_dim"]
        report.success(
            f"Set up {type(MODEL)} model with {ndim} parameters and loaded "
            f"{len(DATA)} patients"
        )

    if args.ti:
        with report.status("Prepare thermodynamic integration..."):
            # make sure path to output file exists
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # set up sampling params
            temp_schedule = params["sampling"]["temp_schedule"]
            nsteps = params["sampling"]["nsteps"]
            burnin = params["sampling"]["burnin"]
            report.success("Prepared thermodynamic integration.")

        # record some information about each burnin phase
        x_axis = temp_schedule.copy()
        plots = {
            "acor_times": [],
            "accept_rates": [],
        }
        for i,inv_temp in enumerate(temp_schedule):
            report.print(f"TI round {i+1}/{len(temp_schedule)} with β = {inv_temp}")

            # set up backend
            hdf5_backend = emcee.backends.HDFBackend(output_path, name=f"ti/{i+1:0>2d}")

            # create log-probability function
            def log_prob_fn(theta):
                """Compute log-probability of parameter sample `theta`."""
                llh = MODEL.likelihood(given_params=theta, log=True)
                if np.isinf(llh):
                    return -np.inf, -np.inf
                return inv_temp * llh, llh

            burnin_info = run_mcmc_with_burnin(
                nwalkers, ndim, log_prob_fn, nsteps,
                persistent_backend=hdf5_backend,
                burnin=burnin,
                keep_burnin=False,
                verbose=True
            )
            plots["acor_times"].append(burnin_info["acor_times"][-1])
            plots["accept_rates"].append(burnin_info["accept_rate"])

        # copy last sampling round over to a group in the HDF5 file called "mcmc"
        # because that is what other scripts expect to see, e.g. for plotting risks
        h5_file = h5py.File(output_path, "r+")
        h5_file.copy(f"ti/{len(temp_schedule):0>2d}", h5_file, name="mcmc")
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
                """Compute the log-probability of data loaded in `model`, given the
                parameters `theta`.

                Note that the `llh` is also passed as second return value to be stored
                in the `emcee` blobs, where it is also stored in the TI case.
                """
                llh = MODEL.likelihood(given_params=theta, log=True)
                return llh, llh

            report.success(f"Prepared sampling params & backend at {output_path}")

        burnin_info = run_mcmc_with_burnin(
            nwalkers, ndim, log_prob_fn,
            nstep=params["sampling"]["nstep"],
            persistent_backend=hdf5_backend,
            sampling_kwargs=params["sampling"]["kwargs"],
            verbose=True,
        )
        x_axis = np.array(burnin_info["iterations"])
        plots = {"acor_times": burnin_info["acor_times"]}

    with report.status("Store plots about burnin phases..."):
        plot_path = Path(args.plots)
        plot_path.mkdir(parents=True, exist_ok=True)

        for name, y_axis in plots.items():
            tmp_df = pd.DataFrame(
                np.array([x_axis, y_axis]).T,
                columns=["x", name],
            )
            tmp_df.to_csv(plot_path/(name + ".csv"), index=False)

        report.success(
            f"Stored {len(plots)} plots about burnin phases at {plot_path}"
        )
