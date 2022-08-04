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
import scipy as sp
import yaml
from rich.progress import track

from .helpers import get_graph_from_, report


def log_estimator(log_probs: np.ndarray) -> float:
    """
    Compute the MCMC estimator for a series of log probabilities.
    """
    num_samples = len(log_probs)
    return sp.special.logsumexp(log_probs) - np.log(num_samples)


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

    # Only read in two header rows when using the Unilateral model
    if params["model"]["class"] in ["Bilateral", "MidlineBilateral"]:
        with report.status("Read in patient data..."):
            inference_data = pd.read_csv(data_path, header=[0,1,2])
            report.success(f"Read in patient data from {data_path}")

        with report.status("Recreate model to compute more metrics..."):
            model_cls = getattr(lymph, params["model"]["class"])
            graph = get_graph_from_(params["model"]["graph"])
            MODEL = model_cls(graph=graph)
            MODEL.modalities = params["modalities"]
            MODEL.patient_data = inference_data
            FIRST_BINOM_PROB = params["model"]["first_binom_prob"]
            MAX_T = params["model"]["max_t"]
            T_STAGES = params["model"]["t_stages"]
            time = np.arange(0, MAX_T + 1)

        def ipsi_and_contra_log_llh(theta,) -> float:
            """
            Compute the log-probability of ipsi- and contralateral data for a bilateral
            model separately.
            """
            num_t_stages = len(T_STAGES)
            spread_probs = theta[:-num_t_stages+1]
            later_binom_probs = theta[-num_t_stages+1:]

            time_dists = {}
            for i,stage in enumerate(T_STAGES):
                if i == 0:
                    time_dists[stage] = lymph.utils.fast_binomial_pmf(
                        time, MAX_T, FIRST_BINOM_PROB
                    )
                else:
                    time_dists[stage] = lymph.utils.fast_binomial_pmf(
                        time, MAX_T, later_binom_probs[i-1]
                    )
            MODEL.spread_probs = spread_probs

            if isinstance(MODEL, lymph.Bilateral):
                ipsi_log_llh = MODEL.ipsi._log_likelihood(
                    t_stages=T_STAGES,
                    max_t=MAX_T,
                    time_dists=time_dists,
                )
                contra_log_llh = MODEL.contra._log_likelihood(
                    t_stages=T_STAGES,
                    max_t=MAX_T,
                    time_dists=time_dists,
                )
            elif isinstance(MODEL, lymph.MidlineBilateral):
                ipsi_log_llh = MODEL.ext.ipsi._log_likelihood(
                    t_stages=T_STAGES,
                    max_t=MAX_T,
                    time_dists=time_dists,
                )
                ipsi_log_llh += MODEL.noext.ipsi._log_likelihood(
                    t_stages=T_STAGES,
                    max_t=MAX_T,
                    time_dists=time_dists,
                )
                contra_log_llh = MODEL.ext.contra._log_likelihood(
                    t_stages=T_STAGES,
                    max_t=MAX_T,
                    time_dists=time_dists,
                )
                contra_log_llh += MODEL.noext.contra._log_likelihood(
                    t_stages=T_STAGES,
                    max_t=MAX_T,
                    time_dists=time_dists,
                )
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

            metrics["ipsi_log_llh"] = log_estimator(ipsi_log_llh)
            metrics["contra_log_llh"] = log_estimator(contra_log_llh)
            report.success("Computed metrics for sides separately")

    with report.status("Write out metrics..."):
        # populate metrics dictionary
        metrics["marg_log_likelihood"] = log_estimator(log_probs)
        metrics["max_log_likelihood"] = np.max(log_probs)
        metrics["mean_accept_frac"] = np.mean(backend.accepted) / nstep
        metrics["mean_acor_time"] = np.mean(backend.get_autocorr_time(tol=0))

        # store metrics in JSON file
        metrics_path = Path(args.metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, mode='w') as metrics_file:
            metrics_file.write(json.dumps(metrics))
        report.success(f"Wrote out metrics to {metrics_path}")
