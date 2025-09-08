"""Learn the spread probabilities of the HMM for lymphatic tumor progression using
the preprocessed data as input and the mixture model.
"""

# pylint: disable=logging-fstring-interpolation
import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from lymixture.em import expectation, maximization
from lymph import models

from lyscripts.utils import (
    assign_modalities,
    create_mixture,
    load_patient_data,
    load_yaml_params,
    to_numpy,
)

logger = logging.getLogger(__name__)


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """Add an ``ArgumentParser`` to the subparsers action."""
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """Add arguments to a ``subparsers`` instance and run its main function when chosen.

    This is called by the parent module that is called via the command line.
    """
    parser.add_argument("-i", "--input", type=Path, help="Path to training data files")
    parser.add_argument(
        "--history",
        type=Path,
        nargs="?",
        help="Path to store history in (as CSV file).",
    )
    parser.add_argument(
        "-p",
        "--params",
        default="./params.yaml",
        type=Path,
        help="Path to parameter file.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Seed value to reproduce the same sampling round.",
    )
    parser.add_argument(
        "-m",
        "--multi_fit",
        type=bool,
        default=False,
        help="Whether to fit multiple models for later uncertainty evaluation",
    )
    parser.add_argument(
        "-sp",
        "--starting_point",
        type=Path,
        default=None,
        help="Starting point for optimization if we do not want to start from a random point",
    )

    parser.set_defaults(run_main=main)


MIXTURE = None


def log_prob_fn() -> float:
    """Log probability function using global variables because of pickling."""
    return MIXTURE.likelihood(
        use_complete=True, given_resps=MIXTURE.get_resps(norm=True)
    )


def check_convergence(
    params_history, likelihood_history, steps_back_list, absolute_tolerance=0.01
):
    current_params = params_history[-1]
    current_likelihood = likelihood_history[-1]
    for steps_back in steps_back_list:
        previous_params = params_history[-steps_back - 1]
        if np.allclose(to_numpy(current_params), to_numpy(previous_params)):
            logger.info(
                f"Converged after {len(params_history)} steps. due to parameter similarity"
            )
            return True  # Return True if any of the steps is close
        elif np.isclose(
            current_likelihood,
            likelihood_history[-steps_back - 1],
            rtol=0,
            atol=absolute_tolerance,
        ):
            logger.info(
                f"Converged after {len(params_history)} steps. due to likelihood similarity"
            )
            return True
    return False


def run_EM(tolerance, history_dir=None):
    """Run the EM algorithm to determine the optimal parameters."""
    os.makedirs(history_dir, exist_ok=True)
    is_converged = False
    iteration = 0
    params = MIXTURE.get_params()
    params_history = []
    likelihood_history = []
    params_history.append(params.copy())
    likelihood_history.append(MIXTURE.likelihood(use_complete=False))
    # Number of steps to look back for convergence
    look_back_steps = 3

    while not is_converged:
        logger.info(f"Iteration: {iteration}")
        logger.info(f"Likelihood: {likelihood_history[-1]}")
        latent = expectation(MIXTURE, params, log=True)
        MIXTURE.set_resps(np.exp(latent))
        params = maximization(MIXTURE, latent)

        # Append current params and likelihood to history
        params_history.append(params.copy())
        likelihood_history.append(MIXTURE.likelihood(use_complete=False))
        if history_dir != None:
            llh_history = pd.DataFrame(likelihood_history)
            llh_history.columns = ["likelihoods"]
            llh_history.to_csv(history_dir + "/llh.csv", index=False)
            param_history = pd.DataFrame(params_history)
            param_history.to_csv(history_dir + "/params.csv", index=False)
            MIXTURE.get_mixture_coefs().to_csv(
                history_dir + "/mixture_coef.csv", index=False
            )
        # Check if converged
        if iteration >= 3:  # Ensure enough history is available
            is_converged = check_convergence(
                params_history,
                likelihood_history,
                list(range(1, look_back_steps + 1)),
                tolerance,
            )
        iteration += 1
    df = pd.DataFrame.from_dict(MIXTURE.get_params(), orient="index", columns=["value"])
    df.to_csv(history_dir + "/optimal_params.csv")
    return params_history, likelihood_history


def process_dataset(
    dataset, folder_path, initial_params, model_build_params, index, look_back_steps=3
):
    os.makedirs(folder_path, exist_ok=True)
    subpath_optimal_params = "optimal_params"
    os.makedirs(os.path.join(folder_path, subpath_optimal_params), exist_ok=True)
    subpath_params_history = "params_history"
    os.makedirs(os.path.join(folder_path, subpath_params_history), exist_ok=True)
    subpath_likelihood_history = "likelihood_history"
    os.makedirs(os.path.join(folder_path, subpath_likelihood_history), exist_ok=True)

    logger.info(f"Starting dataset {index}")
    mixture = create_mixture(model_build_params)
    mapping = model_build_params["model"].get("mapping", None)
    if isinstance(mixture.components[0], models.Unilateral):
        mixture.load_patient_data(
            dataset,
            split_by=model_build_params["model"].get(
                "split_by", ("tumor", "1", "subsite")
            ),
            mapping=mapping,
        )
        assign_modalities(
            model=mixture, config=model_build_params.get("inference_modalities", {})
        )
    else:
        raise ValueError("Only Unilateral has been implemented so far")

    mixture.set_params(**initial_params)
    mixture.normalize_mixture_coefs()
    tolerance = model_build_params["model"].get("likelihood_tolerance", 0.01)
    mixture.set_params(**initial_params)
    params = initial_params.copy()

    mixture.normalize_mixture_coefs()
    params_history = [params.copy()]
    likelihood_history = [mixture.likelihood(use_complete=False)]

    is_converged = False
    count = 0
    logger.info(f"[Dataset {index}] started")
    file_prefix = f"dataset_{index}"

    while not is_converged:
        latent = expectation(mixture, params, log=True)
        mixture.set_resps(np.exp(latent))
        params = maximization(mixture, latent)

        params_history.append(params.copy())
        likelihood_history.append(mixture.likelihood(use_complete=False))

        llh_history = pd.DataFrame(likelihood_history)
        llh_history.columns = ["likelihoods"]
        llh_history.to_csv(
            os.path.join(
                folder_path,
                subpath_likelihood_history,
                f"{file_prefix}_likelihood_history.csv",
            ),
            index=False,
        )
        param_history = pd.DataFrame(params_history)
        param_history.to_csv(
            os.path.join(
                folder_path, subpath_params_history, f"{file_prefix}_param_history.csv"
            ),
            index=False,
        )
        if count >= look_back_steps:
            is_converged = check_convergence(
                params_history,
                likelihood_history,
                list(range(1, look_back_steps + 1)),
                tolerance,
            )

        count += 1

    logger.info(f"[Dataset {index}] Converged after {count+1} steps")

    df = pd.DataFrame.from_dict(mixture.get_params(), orient="index", columns=["value"])
    df.to_csv(
        os.path.join(
            folder_path, subpath_optimal_params, f"{file_prefix}_optimal_params.csv"
        )
    )


def main(args: argparse.Namespace) -> None:
    """Main function to run the EM algorithm for a mixture model"""
    params = load_yaml_params(args.params)
    global MIXTURE
    MIXTURE = create_mixture(params)
    original_data = load_patient_data(params["general"]["data"])

    mapping = params["model"].get("mapping", None)
    if isinstance(MIXTURE.components[0], models.Unilateral):
        MIXTURE.load_patient_data(
            original_data,
            split_by=params["model"].get("split_by", ("tumor", "1", "subsite")),
            mapping=mapping,
        )
        assign_modalities(model=MIXTURE, config=params.get("inference_modalities", {}))

    else:
        raise ValueError("Only Unilateral has been implemented so far")

    if args.starting_point is None:
        rng = np.random.default_rng(params["em"].get("seed", 42))
        starting_values = {k: rng.uniform() for k in MIXTURE.get_params()}
    else:
        logger.info(f"Using starting point from {args.starting_point}")
        starting_df = pd.read_csv(
            args.starting_point, index_col=0
        )  # Use first column as index
        starting_values = starting_df["value"].to_dict()

    if args.multi_fit:
        datasets = []
        history_dir = params["sampling"]["output_path"]
        os.makedirs(history_dir, exist_ok=True)
        for i in range(params["sampling"]["n_bootstraps"]):
            file_path = os.path.join(args.input, f"dataset_resample_{i}.csv")
            if os.path.exists(file_path):
                loaded_dataset = pd.read_csv(file_path, header=[0, 1, 2])
                datasets.append(loaded_dataset)
        with ProcessPoolExecutor(max(1, os.cpu_count() - 2)) as executor:
            futures = [
                executor.submit(
                    process_dataset, dataset, history_dir, starting_values, params, i
                )
                for i, dataset in enumerate(datasets)
            ]
    else:
        MIXTURE.set_params(**starting_values)
        MIXTURE.normalize_mixture_coefs()
        tolerance = params["model"].get("likelihood_tolerance", 0.01)
        history_dir = params["fitting"]["folder_path"]
        logger.info(f"Saving history to {history_dir}.")
        params_history, likelihood_history = run_EM(
            tolerance=tolerance, history_dir=history_dir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
