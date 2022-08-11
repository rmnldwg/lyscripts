"""
Evaluate the performance of the model.
"""
import argparse
import json
from pathlib import Path

import emcee
import h5py
import lymph
import numpy as np
import pandas as pd
import yaml
from scipy.integrate import trapezoid

from .helpers import model_from_config, report


def comp_bic(
    log_probs: np.ndarray,
    num_params: int,
    num_data: int
) -> float:
    """
    Compute the negative one half of the Bayesian Information Criterion (BIC).

    The BIC is defined as
    $$ BIC = k \\ln{n} - 2 \\ln{\\hat{L}} $$
    where $k$ is the number of parameters `num_params`, $n$ the number of datapoints
    `num_data` and $\\hat{L}$ the maximum likelihood estimate of the `log_prob`.
    It is constructed such that the following is an
    approximation of the model evidence:
    $$ p(D \\mid m) \\approx \\exp{\\left( - BIC / 2 \\right)} $$
    which is why this function returns the negative one half of it.
    """
    return np.max(log_probs) - num_params * np.log(num_data) / 2.

def comp_evidence_error(
    accuracies: np.ndarray,
    errors: np.ndarray,
    temp_schedule: np.ndarray,
    num: int = 1000,
) -> float:
    """Compute the error (standard deviation) of the evidence.

    It is computed by drawing a `num` new accuracies from normal distributions
    that have as their respective means the provided `accuracies` and as standard
    deviations the provided `errors`, which should be the standard deviations of the
    `accuracies`. Then, each of the drawn sequence of accuracies is integrated over
    `temp_schedule`.
    """
    drawn_accuracies = np.random.normal(
        loc=accuracies,
        scale=errors,
        size=(num, len(accuracies))
    )
    integrals = trapezoid(y=drawn_accuracies, x=temp_schedule, axis=1)
    return np.std(integrals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d", "--data", required=True,
        help="Path to the tables of patient data (CSV)."
    )
    parser.add_argument(
        "-m", "--model", required=True,
        help="Path to model output files (HDF5)."
    )
    parser.add_argument(
        "-p", "--params", default="params.yaml",
        help="Path to parameter file (YAML)."
    )
    parser.add_argument(
        "--plots", default="plots",
        help="Directory for storing plots"
    )
    parser.add_argument(
        "--metrics", default="metrics.json",
        help="Path to metrics file (JSON)."
    )

    args = parser.parse_args()
    metrics = {}

    with report.status("Read in parameters..."):
        params_path = Path(args.params)
        with open(params_path, mode='r') as params_file:
            params = yaml.safe_load(params_file)
        report.success(f"Read in params from {params_path}")

    with report.status("Open samples from emcee backend..."):
        model_path = Path(args.model)
        backend = emcee.backends.HDFBackend(model_path, read_only=True, name="mcmc")
        nstep = backend.iteration
        chain = backend.get_chain(flat=True)

        # use blobs, because also for TI, this is the unscaled log-prob
        log_probs = backend.get_blobs()
        report.success(f"Opened samples from emcee backend from {model_path}")

    with report.status("Read in patient data..."):
        data_path = Path(args.data)
        # Only read in two header rows when using the Unilateral model
        is_unilateral = params["model"]["class"] == "Unilateral"
        header = [0, 1] if is_unilateral else [0, 1, 2]
        DATA = pd.read_csv(data_path, header=header)
        report.success(f"Read in patient data from {data_path}")

    with report.status("Recreate model & load data..."):
        MODEL = model_from_config(
            graph_params=params["graph"],
            model_params=params["model"],
            modalities_params=params["modalities"],
        )
        MODEL.patient_data = DATA
        ndim = len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric
        nwalkers = ndim * params["sampling"]["walkers_per_dim"]
        report.success(
            f"Recreated {type(MODEL)} model with {ndim} parameters and loaded "
            f"{len(DATA)} patients"
        )

    # Only read in two header rows when using the Unilateral model
    if isinstance(MODEL, (lymph.Bilateral, lymph.MidlineBilateral)):
        def ipsi_and_contra_log_llh(theta,) -> float:
            """
            Compute the log-probability of ipsi- and contralateral data for a bilateral
            model separately.
            """
            MODEL.check_and_assign(theta)

            if isinstance(MODEL, lymph.Bilateral):
                ipsi_log_llh = MODEL.ipsi._likelihood(log=True)
                contra_log_llh = MODEL.contra._likelihood(log=True)
            elif isinstance(MODEL, lymph.MidlineBilateral):
                ipsi_log_llh = MODEL.ext.ipsi._likelihood(log=True)
                ipsi_log_llh += MODEL.noext.ipsi._likelihood(log=True)
                contra_log_llh = MODEL.ext.contra._likelihood(log=True)
                contra_log_llh += MODEL.noext.contra._likelihood(log=True)
            else:
                raise TypeError(f"Model class {type(MODEL)} not supported.")

            return ipsi_log_llh, contra_log_llh

        with report.status("Compute metrics for sides separately..."):
            ipsi_log_llh = np.zeros(shape=len(chain))
            contra_log_llh = np.zeros(shape=len(chain))
            for i,sample in enumerate(chain):
                ipsi_log_llh[i], contra_log_llh[i] = ipsi_and_contra_log_llh(sample)

            if isinstance(MODEL, lymph.Bilateral):
                num_params = (
                    len(MODEL.ipsi.spread_probs)
                    + MODEL.diag_time_dists.num_parametric
                )
            elif isinstance(MODEL, lymph.MidlineBilateral):
                num_params = (
                    len(MODEL.ext.ipsi.spread_probs)
                    + MODEL.diag_time_dists.num_parametric
                )

            metrics["ipsi_BIC"] = comp_bic(
                ipsi_log_llh,
                num_params,
                len(DATA),
            )
            metrics["contra_BIC"] = comp_bic(
                contra_log_llh,
                num_params,
                len(DATA),
            )
            report.success("Computed metrics for sides separately")


    h5_file = h5py.File(name=model_path, mode="r")
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
            accuracies = np.zeros_like(temp_schedule)
            errors = np.zeros_like(temp_schedule)
            for i,round in enumerate(h5_file["ti"]):
                reader = emcee.backends.HDFBackend(
                    model_path,
                    name=f"ti/{round}",
                    read_only=True,
                )
                log_probs = reader.get_blobs()
                accuracies[i] = np.mean(log_probs)
                errors[i] = np.std(log_probs)

            metrics["evidence"] = trapezoid(y=accuracies, x=temp_schedule)
            metrics["evidence_std"] = comp_evidence_error(
                accuracies, errors, temp_schedule
            )
            report.success(
                f"Computed results of thermodynamic integration with {num_temps} steps"
            )

        with report.status("Plot β vs accuracy..."):
            plot_path = Path(args.plots) / "ti" / "accuracies.csv"
            plot_path.parent.mkdir(exist_ok=True)

            tmp_df = pd.DataFrame(
                np.array([temp_schedule, accuracies]).T,
                columns=["β", "accuracy"],
            )
            tmp_df.to_csv(plot_path, index=False)
            report.success(f"Plotted β vs accuracy at {plot_path}")

    with report.status("Write out metrics..."):
        # store metrics in JSON file
        metrics_path = Path(args.metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.touch(exist_ok=True)

        # further populate metrics dictionary
        metrics["BIC"] = comp_bic(
            log_probs,
            len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric,
            len(DATA),
        )
        metrics["max_llh"] = np.max(log_probs)
        metrics["mean_llh"] = np.mean(log_probs)

        # write out metrics again
        with open(metrics_path, mode="w") as metrics_file:
            json.dump(metrics, metrics_file)

        report.success(f"Wrote out metrics to {metrics_path}")
