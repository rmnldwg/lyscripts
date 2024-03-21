"""
Given samples drawn during an MCMC round, precompute the (prior) state distribution for
each sample. This may then later on be used to compute risks and prevalences more
quickly.
"""
import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from lymph import types
from rich import progress

from lyscripts.decorators import log_state
from lyscripts.utils import create_model, load_model_samples, load_yaml_params

logger = logging.getLogger(__name__)


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """Add an `ArgumentParser` to the subparsers action."""
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """Add arguments needed to run this script to a `subparsers` instance."""
    parser.add_argument(
        "-s", "--samples", type=Path, required=True,
        help="Path to the drawn samples (HDF5 file)."
    )
    parser.add_argument(
        "-p", "--params", default="./params.yaml", type=Path,
        help="Path to parameter file."
    )
    parser.add_argument(
        "--state-dists", default="./distributions.hdf5", type=Path,
        help="Path to file for storing state distributions."
    )
    parser.add_argument(
        "--kwargs", default="{}", type=json.loads,
        help="Keyword arguments for the `state_dist()` method of the model."
    )
    parser.set_defaults(run_main=main)


@log_state()
def compute_state_dists(
    samples: np.ndarray,
    model: types.ModelT,
    **state_dist_kwargs,
) -> np.ndarray:
    """Compute an array of state distributions from the `samples` for the `model`."""
    num_samples = len(samples)
    state_dists = np.empty(shape=(num_samples, *model.state_dist(**state_dist_kwargs).shape))
    for i, sample in progress.track(
        sequence=enumerate(samples),
        description="[blue]INFO     [/blue]Computing state distributions",
        total=num_samples,
    ):
        model.set_params(*sample)
        state_dists[i] = model.state_dist(**state_dist_kwargs)
    return state_dists


@log_state()
def store_in_hdf5(file_path: Path, array: np.ndarray) -> None:
    """Store the `state_dists` in an HDF5 file at `file_path`."""
    with h5py.File(file_path, "w") as file:
        previous = file.get("state_dists", default=None)
        if previous is not None:
            logger.info("Overwriting previous state distributions")
            del file["state_dists"]

        file.create_dataset("state_dists", data=array)


def main(args: argparse.Namespace):
    """Precompute the prior state distribution for each sample."""
    params = load_yaml_params(args.params)
    model = create_model(params)
    samples = load_model_samples(args.samples, flat=True)
    state_dists = compute_state_dists(samples, model, **args.kwargs)
    store_in_hdf5(Path(args.state_dists), state_dists)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
