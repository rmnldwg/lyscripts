"""
Learn the spread probabilities of the HMM for lymphatic tumor progression using
the preprocessed data as input and MCMC as sampling method.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import emcee
import h5py
import lymph
import numpy as np
import pandas as pd
import yaml
from lymph import EnsembleSampler

from .helpers import get_graph_from_, report

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
        "--plots", default=None,
        help="Path to folder containing the plots."
    )
    parser.add_argument(
        "--metrics", default=None,
        help="Path to metrics file (JSON)."
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

    # Only read in two header rows when using the Unilateral model
    if params["model"]["class"] == "Unilateral":
        header = [0, 1]
    else:
        header = [0, 1, 2]

    with report.status("Read in training data..."):
        inference_data = pd.read_csv(input_path, header=header)
        report.success(f"Read in training data from {input_path}")

    with report.status("Set up model & load data..."):
        model_cls = getattr(lymph, params["model"]["class"])
        graph = get_graph_from_(params["model"]["graph"])
        MODEL = model_cls(graph=graph, **params["model"]["kwargs"])
        MODEL.modalities = params["modalities"]

        # use fancy new time marginalization functionality
        for i,t_stage in enumerate(params["model"]["t_stages"]):
            if i == 0:
                MODEL.diag_time_dists[t_stage] = lymph.utils.fast_binomial_pmf(
                    k=np.arange(params["model"]["max_t"] + 1),
                    n=params["model"]["max_t"],
                    p=params["model"]["first_binom_prob"],
                )
            else:
                def binom_pmf(t,p):
                    if p > 1. or p < 0.:
                        raise ValueError("Binomial probability must be between 0 and 1")
                    return lymph.utils.fast_binomial_pmf(t, params["model"]["max_t"], p)

                MODEL.diag_time_dists[t_stage] = binom_pmf

        MODEL.patient_data = inference_data
        ndim = len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric
        report.success(
            f"Set up model with {ndim} parameters and loaded {len(inference_data)} "
            "patients"
        )

    if args.ti:
        with report.status("Prepare thermodynamic integration..."):
            # make sure path to output file exists
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # set up sampling params
            ladder = np.logspace(0., 1., num=params["sampling"]["len_ladder"])
            ladder = (ladder - 1.) / 9.
            nwalkers = ndim * params["sampling"]["walkers_per_dim"]
            coords = np.random.uniform(size=(nwalkers,ndim))
            nsteps = params["sampling"]["kwargs"]["max_steps"] // len(ladder)
            report.success("Prepared thermodynamic integration.")

            # initialize metrics and plots
            acor_times = np.zeros_like(ladder)
            accept_rates = np.zeros_like(ladder)
            accuracies = np.zeros_like(ladder)

        for i,inv_temp in enumerate(ladder):
            report.print(f"TI round {i+1}/{len(ladder)} with β = {inv_temp:.3f}")
            backend = emcee.backends.HDFBackend(
                filename=output_path,
                name=f"ti_round_{i+1:0>2d}",
            )
            moves = [
                (emcee.moves.DEMove(),        0.8),
                (emcee.moves.DESnookerMove(), 0.2)
            ]

            def log_prob_fn(theta):
                """Return log probability of model scaled with inverse temperature."""
                llh = MODEL.likelihood(given_params=theta, log=True)
                if np.isinf(llh):
                    return -np.inf, -np.inf
                return inv_temp * llh, llh

            with Pool() as pool:
                sampler = emcee.EnsembleSampler(
                    nwalkers=nwalkers,
                    ndim=ndim,
                    log_prob_fn=log_prob_fn,
                    pool=pool,
                    backend=backend,
                    moves=moves,
                )
                state = sampler.run_mcmc(coords, nsteps=nsteps, progress=True)
                coords = state.coords

            # compute some metrics
            acor_times[i] = np.mean(sampler.get_autocorr_time(tol=0))
            accept_rates[i] = np.mean(sampler.acceptance_fraction)
            accuracies[i] = np.mean(sampler.get_blobs(
                discard=nsteps - params["sampling"]["keep_steps"]
            ))
            report.print(
                f"Finished round {i+1} with acceptance rate {accept_rates[i]:.3f}"
            )

        # copy last sampling round over to a group in the HDF5 file called "mcmc"
        h5_file = h5py.File(output_path, "r+")
        h5_file.copy(f"ti_round_{len(ladder):0>2d}", h5_file, name="mcmc")
        report.success("Finished thermodynamic integration.")

        with report.status("Compute plots and metrics..."):
            plots = {}
            plots["acor"] = pd.DataFrame(
                data=np.stack([ladder, acor_times], axis=0).T,
                columns=["beta", "acor"]
            )
            plots["accept"] = pd.DataFrame(
                data=np.stack([ladder, accept_rates], axis=0).T,
                columns=["beta", "accept"]
            )
            plots["accuracy"] = pd.DataFrame(
                data=np.stack([ladder, accuracies], axis=0).T,
                columns=["beta", "accuracy"]
            )
            evidence = 0
            for i in range(len(ladder) - 1):
                temp_diff = ladder[i+1] - ladder[i]
                accuracy_mean = (accuracies[i+1] + accuracies[i]) / 2.
                evidence -= temp_diff * accuracy_mean

            report.success("Computed plots and metrics.")

    else:
        with report.status("Prepare sampling params & backend..."):
            # make sure path to output file exists
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # set up sampling params
            ndim = len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric
            backend = emcee.backends.HDFBackend(output_path)
            report.success(f"Prepared sampling params & backend at {output_path}")

        # function needs to be defined at runtime and have no `args` and `kwargs`,
        # otherwise `multiprocessing` won't work properly (i.e. give no performance
        # benefit). Variables from outside are capitalized.
        def log_prob_fn(theta,) -> float:
            """
            Compute the log-probability of data loaded in `model`, given the
            parameters `theta`.
            """
            return MODEL.likelihood(given_params=theta, log=True)

        with Pool() as pool:
            nwalker = ndim * params["sampling"]["walkers_per_dim"]
            sampler = EnsembleSampler(
                nwalkers=nwalker,
                ndim=ndim,
                log_prob_fn=log_prob_fn,
                backend=backend,
                pool=pool,
            )
            plots = {}
            plots["acor"] = sampler.run_sampling(**params["sampling"]["kwargs"])
            report.success("Sampling done.")

    if args.plots is not None:
        # make sure path to plots file exists
        plots_path = Path(args.plots)
        plots_path.mkdir(parents=True, exist_ok=True)

        with report.status("Storing plots..."):
            for name,df in plots.items():
                df.to_csv(plots_path / (name + ".csv"), index=False)
            report.success(f"Stored plots at {plots_path}")

    if args.metrics is not None:
        # make sure path to metrics file exists
        metrics_path = Path(args.metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.touch(exist_ok=True)

        with report.status("Write out metrics..."):
            metrics = {}
            metrics["evidence"] = evidence

            # write out metrics again
            with open(metrics_path, mode="w") as metrics_file:
                json.dump(metrics, metrics_file)

            report.success(f"Wrote out metrics to {metrics_path}")
