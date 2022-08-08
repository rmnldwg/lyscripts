"""
Perform a thermodynamic integration from the (uniform) prior of the model to its
posterior inorder to obtain an estimate of the model's evidence.
"""
import argparse
from pathlib import Path

import lymph
import numpy as np
import pandas as pd
import yaml

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
        "-p", "--params", default="params.yaml",
        help="Path to parameter YAML file"
    )
    parser.add_argument(
        "--acor", default=None,
        help="Path to CSV file storing the autocorrelation of the chain."
    )
    parser.add_argument(
        "-l", "--ladder", choices=["linear", "geometric"], default="geometric",
        help="Define the type of sequence of the inverse temperature",
    )

    # Parse arguments and prepare paths
    args = parser.parse_args()
    input_path = Path(args.input)
    params_path = Path(args.params)

    with report.status("Read in parameters..."):
        with open(params_path, 'r') as params_file:
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
        ndims = len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric
        report.success(
            f"Set up model with {ndims} parameters and loaded {len(inference_data)} "
            "patients"
        )