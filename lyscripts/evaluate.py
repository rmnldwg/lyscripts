"""
Evaluate the performance of the model.

## WARNING!
What I have been doing here is wrong! I tried to compute the marginal log-likelihood
from the stored log-likelihoods during sampling. That does not work!
"""
import argparse
import json
from pathlib import Path

import emcee
import lymph
import numpy as np
import pandas as pd
import yaml
from rich.progress import track

from .helpers import get_graph_from_, report


def comp_bic(
    log_probs: np.ndarray,
    num_params: int,
    num_data: int
) -> float:
    """
    Compute the Bayesian Information Criterion (BIC).
    """
    return num_params * np.log(num_data) - 2 * np.max(log_probs)

def comp_enhanced_bic(
    log_probs: np.ndarray,
    num_params: int,
    num_data: int
) -> float:
    """
    Compute the enhanced Bayesian Information Criterion (eBIC, my invention), where
    the maximum likelihood estimate is replaced with the expected likelihood.
    """
    return num_params * np.log(num_data) - 2 * np.mean(log_probs)


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
        "--metrics", required=True,
        help="Path to metrics file (JSON)."
    )

    args = parser.parse_args()
    data_path = Path(args.data)
    model_path = Path(args.model)
    params_path = Path(args.params)
    metrics = {}

    with report.status("Read in parameters..."):
        with open(params_path, mode='r') as params_file:
            params = yaml.safe_load(params_file)
        report.success(f"Read in params from {params_path}")

    with report.status("Open samples from emcee backend..."):
        backend = emcee.backends.HDFBackend(model_path, read_only=True)
        nstep = backend.iteration
        backend_kwargs = {
            "flat": True,
            "discard": nstep - params["sampling"]["keep_steps"]
        }
        chain = backend.get_chain(**backend_kwargs)
        log_probs = backend.get_log_prob(**backend_kwargs)
        report.success(f"Opened samples from emcee backend from {model_path}")

    with report.status("Read in patient data..."):
        header_rows = [0,1] if params["model"]["class"] == "Unilateral" else [0,1,2]
        inference_data = pd.read_csv(data_path, header=header_rows)
        report.success(f"Read in patient data from {data_path}")

    with report.status("Recreate model to compute more metrics..."):
        model_cls = getattr(lymph, params["model"]["class"])
        graph = get_graph_from_(params["model"]["graph"])
        MODEL = model_cls(graph=graph)
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
        report.success("Recreated model to compare more metrics.")

    # Only read in two header rows when using the Unilateral model
    if params["model"]["class"] in ["Bilateral", "MidlineBilateral"]:
        def ipsi_and_contra_log_llh(theta,) -> float:
            """
            Compute the log-probability of ipsi- and contralateral data for a bilateral
            model separately.
            """
            MODEL.check_and_assign(theta)

            if isinstance(MODEL, lymph.Bilateral):
                ipsi_log_llh = MODEL.ipsi._likelihood(log=True)
                contra_log_llh = MODEL.contra._log_likelihood(log=True)
            elif isinstance(MODEL, lymph.MidlineBilateral):
                ipsi_log_llh = MODEL.ext.ipsi._likelihood(log=True)
                ipsi_log_llh += MODEL.noext.ipsi._likelihood(log=True)
                contra_log_llh = MODEL.ext.contra._likelihood(log=True)
                contra_log_llh += MODEL.noext.contra._likelihood(log=True)
            else:
                raise TypeError(f"Model class {type(MODEL)} not supported.")

            return ipsi_log_llh, contra_log_llh

        with report.status("Compute metrics for sides separately..."):
            ipsi_log_llh = np.zeros_like(log_probs)
            contra_log_llh = np.zeros_like(log_probs)
            for i,sample in track(
                enumerate(chain),
                description="Computing metrics...",
            ):
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
                len(inference_data),
            )
            metrics["contra_BIC"] = comp_bic(
                contra_log_llh,
                num_params,
                len(inference_data),
            )
            metrics["ipsi_eBIC"] = comp_enhanced_bic(
                ipsi_log_llh,
                num_params,
                len(inference_data),
            )
            metrics["contra_eBIC"] = comp_enhanced_bic(
                contra_log_llh,
                num_params,
                len(inference_data),
            )
            report.success("Computed metrics for sides separately")

    with report.status("Write out metrics..."):
        # store metrics in JSON file
        metrics_path = Path(args.metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.touch(exist_ok=True)

        # read in metrics already present
        with open(metrics_path, mode="r") as metrics_file:
            try:
                metrics = json.load(metrics_file)
            except json.decoder.JSONDecodeError:
                metrics = {}

        # populate metrics dictionary
        metrics["BIC"] = comp_bic(
            log_probs,
            len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric,
            len(inference_data),
        )
        metrics["eBIC"] = comp_enhanced_bic(
            log_probs,
            len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric,
            len(inference_data),
        )
        metrics["max_log_likelihood"] = np.max(log_probs)
        metrics["mean_log_likelihood"] = np.mean(log_probs)

        # write out metrics again
        with open(metrics_path, mode="w") as metrics_file:
            json.dump(metrics, metrics_file)

        report.success(f"Wrote out metrics to {metrics_path}")
