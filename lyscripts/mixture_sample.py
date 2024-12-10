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
from lymixture.em import sample_fixed_mixture, sample_model_params, _set_params, _get_params
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
        "-m", "--mixture_coefs", type=Path, required=True,
        help="File path of mixture coefficients"
    )
    parser.add_argument(
        "-p", "--params", default="./params.yaml", type=Path,
        help="Path to parameter file."
    )
    parser.add_argument(
        "-mp", "--model_params", type=Path, required = True,
        help="File path of mixture coefficients"
    )
    parser.add_argument(
        "--mode", type=str, default = "fixed_mixture",
        help = "Mode of sampling. Use either 'fixed_mixture' or 'fixed_latent'"
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Output path for samples"
    )


    parser.set_defaults(run_main=main)


MIXTURE = None

def log_prob_fn() -> float:
    """log probability function using global variables because of pickling."""
    return MIXTURE.likelihood(use_complete = True, given_resps = MIXTURE.get_resps(norm = True))

def main(args: argparse.Namespace) -> None:
    """Main function to sample parameters for a mixture model"""

    params = load_yaml_params(args.params)
    model_params = pd.read_csv(args.model_params,header = [0])
    mixture_df = pd.read_csv(args.mixtures_coefs)
    inference_data = load_patient_data(args.input)
    param_dict = dict(model_params.iloc[-1])
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

    
    MIXTURE.set_params(**param_dict)
    MIXTURE.set_resps(mixture_df)
    if args.mode == "fixed_mixture":
        backend, samples = sample_fixed_mixture(MIXTURE, steps = params['sampling'].get('steps'),filename = args.output+"fixed_mixture")
    elif args.mode == "fixed_latent":
        backend, samples = sample_model_params(MIXTURE, steps = params['sampling'].get('steps'),filename = args.output+"fixed_latent")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
