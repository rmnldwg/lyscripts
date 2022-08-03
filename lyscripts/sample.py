"""
Learn the spread probabilities of the HMM for lymphatic tumor progression using
the preprocessed data as input and MCMC as sampling method.
"""
import argparse
from multiprocessing import Pool
from pathlib import Path
import yaml
import json

import emcee
import lymph
import numpy as np
import pandas as pd
from lymph import EnsembleSampler

from helpers import get_graph_from_, report


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
        "-p", "--params", default="params.yaml",
        help="Path to parameter YAML file"
    )
    parser.add_argument(
        "--acor", default=None,
        help="Path to CSV file storing the autocorrelation of the chain."
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
        MODEL = model_cls(graph=graph)
        MODEL.modalities = params["modalities"]
        MODEL.patient_data = inference_data
        report.success("Set up model & loaded data")

    with report.status("Prepare sampling params & backend..."):
        # make sure path to output file exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # set up sampling params
        ndim = len(MODEL.spread_probs) + 1
        FIRST_BINOM_PROB = params["model"]["first_binom_prob"]
        MAX_T = params["model"]["max_t"]
        T_STAGES = params["model"]["t_stages"]
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
        num_t_stages = len(T_STAGES)
        spread_probs = theta[:-num_t_stages+1]
        later_binom_probs = theta[-num_t_stages+1:]
        new_theta = np.concatenate(
            [spread_probs, [FIRST_BINOM_PROB, later_binom_probs]]
        )

        return MODEL.binom_marg_log_likelihood(
            theta=new_theta,
            t_stages=T_STAGES,
            max_t=MAX_T
        )

    with Pool() as pool:
        nwalker = ndim * params["sampling"]["walkers_per_dim"]
        sampler = EnsembleSampler(
            nwalkers=nwalker,
            ndim=ndim,
            log_prob_fn=log_prob_fn,
            backend=backend,
            pool=pool,
        )
        acor_df = sampler.run_sampling(**params["sampling"]["kwargs"])
        report.success("Sampling done.")

    if args.acor is not None:
        # make sure path to acor file exists
        acor_path = Path(args.acor)
        acor_path.parent.mkdir(parents=True, exist_ok=True)

        with report.status("Storing autocorrlation times..."):
            acor_df.to_csv(acor_path, index=False)
            report.success(f"Stored autocorrelation times at {acor_path}")
