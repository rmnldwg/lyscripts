"""
Evaluate the performance of the trained model by computing quantities like the
Bayesian information criterion (BIC) or (if thermodynamic integration was performed)
the actual evidence (with error) of the model.
"""
import argparse
import json
from pathlib import Path
from typing import Tuple

import emcee
import h5py
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from lyscripts.utils import cli_load_yaml_params, model_from_config, report


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
        "data", type=Path,
        help="Path to the tables of patient data (CSV)."
    )
    parser.add_argument(
        "model", type=Path,
        help="Path to model output files (HDF5)."
    )

    parser.add_argument(
        "-p", "--params", default="./params.yaml", type=Path,
        help="Path to parameter file"
    )
    parser.add_argument(
        "--plots", default="./plots", type=Path,
        help="Directory for storing plots"
    )
    parser.add_argument(
        "--metrics", default="./metrics.json", type=Path,
        help="Path to metrics file"
    )

    parser.set_defaults(run_main=main)


def comp_bic(
    log_probs: np.ndarray,
    num_params: int,
    num_data: int
) -> float:
    """
    Compute the negative one half of the Bayesian Information Criterion (BIC).

    The BIC is defined as [^1]
    $$ BIC = k \\ln{n} - 2 \\ln{\\hat{L}} $$
    where $k$ is the number of parameters `num_params`, $n$ the number of datapoints
    `num_data` and $\\hat{L}$ the maximum likelihood estimate of the `log_prob`.
    It is constructed such that the following is an
    approximation of the model evidence:
    $$ p(D \\mid m) \\approx \\exp{\\left( - BIC / 2 \\right)} $$
    which is why this function returns the negative one half of it.

    [^1]: https://en.wikipedia.org/wiki/Bayesian_information_criterion
    """
    return np.max(log_probs) - num_params * np.log(num_data) / 2.

def compute_evidence(
    temp_schedule: np.ndarray,
    log_probs: np.ndarray,
    num: int = 1000,
) -> Tuple[float, float]:
    """Compute the evidene and its standard deviation.

    Given a `temp_schedule` of inverse temperatures and corresponding sets of
    `log_probs`, draw `num` "paths" of log-probabilities and compute the evidence for
    each using trapezoidal integration.

    The evidence is then the mean of those `num` integrations, while the error is their
    standard deviation.
    """
    integrals = np.zeros(shape=num)
    for i in range(num):
        rand_idx = np.random.choice(log_probs.shape[1], size=log_probs.shape[0])
        drawn_accuracy = log_probs[np.arange(log_probs.shape[0]),rand_idx].copy()
        integrals[i] = trapezoid(y=drawn_accuracy, x=temp_schedule)
    return np.mean(integrals), np.std(integrals)


def main(args: argparse.Namespace):
    """
    To evaluate the performance of a sampling round, this program follows these steps:

    1. Read in the paramter file `params.yaml` and the data that was used for inference
    2. Recreate an instance of the model that was used during the training stage
    3. If thermodynamic integration (TI) [^2] was performed, compute the accuracy for
    every TI step and integrate over it to obtain the evidence. This is computed for
    a number of samples of the computed accuracies to also obtain an error on the
    evidence.
    4. Use the recreated model and data to compute quantities like the mean and maximum
    likelihood, as well as the Bayesian information criterion (BIC).

    Everything computed is then stored in a `metrics.json` file and for TI, a CSV is
    created that shows how the accuracy developed during the runs.

    The output of running `lyscripts evaluate --help` shows this:

    ```
    usage: lyscripts evaluate [-h] [-p PARAMS] [--plots PLOTS] [--metrics METRICS]
                            data model

    Evaluate the performance of the trained model by computing quantities like the
    Bayesian information criterion (BIC) or (if thermodynamic integration was performed)
    the actual evidence (with error) of the model.


    POSITIONAL ARGUMENTS
    data                 Path to the tables of patient data (CSV).
    model                Path to model output files (HDF5).

    OPTIONAL ARGUMENTS
    -h, --help           show this help message and exit
    -p, --params PARAMS  Path to parameter file (default: ./params.yaml)
    --plots PLOTS        Directory for storing plots (default: ./plots)
    --metrics METRICS    Path to metrics file (default: ./metrics.json)
    ```

    [^2]: https://doi.org/10.1007/s11571-021-09696-9
    """
    metrics = {}

    params = cli_load_yaml_params(args.params)

    with report.status("Read in patient data..."):
        # Only read in two header rows when using the Unilateral model
        is_unilateral = params["model"]["class"] == "Unilateral"
        header = [0, 1] if is_unilateral else [0, 1, 2]
        DATA = pd.read_csv(args.data, header=header)
        report.success(f"Read in patient data from {args.data}")

    with report.status("Recreate model & load data..."):
        MODEL = model_from_config(
            graph_params=params["graph"],
            model_params=params["model"],
        )
        ndim = len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric
        report.success(
            f"Recreated {type(MODEL)} model with {ndim} parameters and loaded "
            f"{len(DATA)} patients"
        )

    h5_file = h5py.File(args.model, mode="r")
    # check if TI has been performed
    if "ti" in h5_file:
        with report.status("Compute results of thermodynamic integration..."):
            temp_schedule = params["sampling"]["temp_schedule"]
            num_temps = len(temp_schedule)
            if num_temps != len(h5_file["ti"]):
                raise RuntimeError(
                    f"Parameters suggest temp schedule of length {num_temps}, "
                    f"but stored are {len(h5_file['ti'])}"
                )
            nwalker = ndim * params["sampling"]["walkers_per_dim"]
            nsteps = params["sampling"]["nsteps"]
            ti_log_probs = np.zeros(shape=(num_temps, nsteps * nwalker))
            for i,round in enumerate(h5_file["ti"]):
                reader = emcee.backends.HDFBackend(
                    args.model,
                    name=f"ti/{round}",
                    read_only=True,
                )
                ti_log_probs[i] = reader.get_blobs(flat=True)

            evidence, evidence_std = compute_evidence(temp_schedule, ti_log_probs)
            metrics["evidence"] = evidence
            metrics["evidence_std"] = evidence_std
            report.success(
                f"Computed results of thermodynamic integration with {num_temps} steps"
            )

        with report.status("Plot β vs accuracy..."):
            args.plots.parent.mkdir(exist_ok=True)

            tmp_df = pd.DataFrame(
                np.array([
                    temp_schedule,
                    np.mean(ti_log_probs, axis=1),
                    np.std(ti_log_probs, axis=1)
                ]).T,
                columns=["β", "accuracy", "std"],
            )
            tmp_df.to_csv(args.plots, index=False)
            report.success(f"Plotted β vs accuracy at {args.plots}")

    with report.status("Open samples from emcee backend..."):
        backend = emcee.backends.HDFBackend(args.model, read_only=True, name="mcmc")
        # use blobs, because also for TI, this is the unscaled log-prob
        final_log_probs = backend.get_blobs()
        report.success(f"Opened samples from emcee backend from {args.model}")

    with report.status("Write out metrics..."):
        # store metrics in JSON file
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.touch(exist_ok=True)

        # further populate metrics dictionary
        metrics["BIC"] = comp_bic(
            final_log_probs, ndim, len(DATA),
        )
        metrics["max_llh"] = np.max(final_log_probs)
        metrics["mean_llh"] = np.mean(final_log_probs)

        # write out metrics again
        with open(args.metrics, mode="w") as metrics_file:
            json.dump(metrics, metrics_file)

        report.success(f"Wrote out metrics to {args.metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
