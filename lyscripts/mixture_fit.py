"""
Learn the spread probabilities of the HMM for lymphatic tumor progression using
the preprocessed data as input and the mixture model.
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import logging
import os
from collections import namedtuple

try:
    from multiprocess import Pool
except ModuleNotFoundError:
    from multiprocessing import Pool

from pathlib import Path

import emcee
import numpy as np
import pandas as pd
from lymph import models
from lymixture import LymphMixture
from lymixture.em import expectation, maximization
from rich.progress import Progress, TimeElapsedColumn, track

from lyscripts.utils import (
    create_mixture,
    load_patient_data,
    load_yaml_params,
    to_numpy,
    assign_modalities
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
    parser.add_argument(
        "-i", "--input", type=Path, required=True,
        help="Path to training data files"
    )
    # parser.add_argument(
    #     "-o", "--output", type=Path, required=True,
    #     help="Path to the HDF5 file to store the results in"
    # )
    parser.add_argument(
        "--history", type=Path, nargs="?",
        help="Path to store history in (as CSV file)."
    )
    parser.add_argument(
        "-p", "--params", default="./params.yaml", type=Path,
        help="Path to parameter file."
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
        help="Seed value to reproduce the same sampling round."
    )
    parser.add_argument(
        "-sp", "--starting_point", type=bool, default=None,
        help="Starting point for optimization if we do not want to start from a random point"
    )


    parser.set_defaults(run_main=main)


MIXTURE = None

def log_prob_fn() -> float:
    """log probability function using global variables because of pickling."""
    return MIXTURE.likelihood(use_complete = True, given_resps = MIXTURE.get_resps(norm = True))


def check_convergence(params_history, likelihood_history, steps_back_list, absolute_tolerance = 0.01):
    current_params = params_history[-1]
    current_likelihood = likelihood_history[-1]
    for steps_back in steps_back_list:
        previous_params = params_history[-steps_back - 1]
        if np.allclose(to_numpy(current_params), to_numpy(previous_params)):
            logger.info(f"Converged after {len(params_history)} steps. due to parameter similarity")
            return True  # Return True if any of the steps is close
        elif np.isclose(current_likelihood, likelihood_history[-steps_back - 1],rtol = 0, atol = absolute_tolerance):
            logger.info(f"Converged after {len(params_history)} steps. due to likelihood similarity")
            return True
    return False


def run_EM(tolerance, history_dir = None):
    """Run the EM algorithm to determine the optimal parameters.
    """
    is_converged = False
    iteration = 0
    params = MIXTURE.get_params()
    params_history = []
    likelihood_history = []
    params_history.append(params.copy())
    likelihood_history.append(MIXTURE.likelihood(use_complete = False))
    # Number of steps to look back for convergence
    look_back_steps = 3

    while not is_converged:
        logger.info(f"Iteration: {iteration}")
        logger.info(f"Likelihood: {likelihood_history[-1]}")
        latent = expectation(MIXTURE, params)
        params = maximization(MIXTURE, latent)
        
        # Append current params and likelihood to history
        params_history.append(params.copy())
        likelihood_history.append(MIXTURE.likelihood(use_complete = False))
        if history_dir != None:
            llh_history = pd.DataFrame(likelihood_history)
            llh_history.columns = ['likelihoods']
            llh_history.to_csv(history_dir + '/llh.csv', index=False)
            param_history = pd.DataFrame(params_history)
            param_history.to_csv(history_dir + '/params.csv', index=False)
            MIXTURE.get_mixture_coefs().to_csv(history_dir + '/mixture_coef.csv', index=False)
        # Check if converged
        if iteration >= 3:  # Ensure enough history is available
            is_converged = check_convergence(params_history, likelihood_history,list(range(1,look_back_steps+1)),tolerance)
        iteration += 1
    return params_history, likelihood_history

def main(args: argparse.Namespace) -> None:
    """Main function to run the EM algorithm for a mixture model"""

    params = load_yaml_params(args.params)
    inference_data = load_patient_data(args.input)

    # ugly, but necessary for pickling
    global MIXTURE
    MIXTURE = create_mixture(params)

    mapping = params["model"].get("mapping", None)
    if isinstance(MIXTURE.components[0], models.Unilateral):
        side = params["model"].get("side", "ipsi")
        MIXTURE.load_patient_data(inference_data, split_by= params["model"].get("split_by", ("tumor", "1", "subsite")), mapping=mapping)
        assign_modalities(model=MIXTURE, config=params.get("inference_modalities", {}))

    else:
        raise "Only Unilateral has been implemented so far"
    # emcee does not support numpy's new random number generator yet.
    rng = np.random.default_rng(params["em"].get("seed", 42))
    starting_values = {k: rng.uniform() for k in MIXTURE.get_params()}
    MIXTURE.set_params(**starting_values)
    MIXTURE.normalize_mixture_coefs()
    tolerance = params['model'].get('likelihood_tolerance', 0.01)
    history_dir = params['general']['history_dir']
    logger.info(f"Saving history to {history_dir}.")
    params_history, likelihood_history = run_EM(tolerance = tolerance, history_dir = history_dir)
    
    llh_history = pd.DataFrame(likelihood_history)
    llh_history.columns = ['likelihoods']
    llh_history.to_csv(history_dir + '/llh.csv', index=False)
    param_history = pd.DataFrame(params_history)
    param_history.to_csv(history_dir + '/params.csv', index=False)
    MIXTURE.get_mixture_coefs().to_csv(history_dir + '/mixture_coef.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
