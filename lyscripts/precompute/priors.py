"""
Given samples drawn during an MCMC round, precompute the (prior) state distribution for
each sample. This may then later on be used to compute risks and prevalences more
quickly.

The computed priors are stored in an HDF5 file under the key ``mode + t_stage``.
"""
import argparse
import logging
from pathlib import Path
from typing import Any, Literal

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
    """Add arguments needed to run this script to a ``subparsers`` instance."""
    parser.add_argument(
        "-s", "--samples", type=Path, required=True,
        help="Path to the drawn samples (HDF5 file)."
    )
    parser.add_argument(
        "-p", "--params", default="./params.yaml", type=Path,
        help="Path to parameter file defining the model (YAML)."
    )
    parser.add_argument(
        "--priors", default="./priors.hdf5", type=Path,
        help="Path to file for storing the computed prior distributions."
    )

    t_or_dist_group = parser.add_mutually_exclusive_group()
    t_or_dist_group.add_argument(
        "--t-stage", type=str,
        help="T-stage to compute the posterior for."
    )
    t_or_dist_group.add_argument(
        "--t-stage-dist", type=float, nargs="+",
        help="Distribution to marginalize over unknown T-stages."
    )

    parser.add_argument(
        "--mode", choices=["HMM", "BN"], default="HMM",
        help="Mode of the model to use for the computation."
    )
    parser.set_defaults(run_main=main)


def compute_priors_from_samples(
    model: types.ModelT,
    samples: np.ndarray,
    t_stage: str | int | None = None,
    t_stage_dist: list[float] | np.ndarray | None = None,
    mode: Literal["HMM", "BN"] = "HMM",
) -> np.ndarray:
    """Compute prior state distributions from the ``samples`` for the ``model``.

    This will call the ``model`` method :py:meth:`~lymph.types.Model.state_dist`
    for each of the ``samples``. If ``t_stage`` is not provided, the priors will be
    computed by marginalizing over the provided ``t_stage_dist``. Otherwise, the
    priors will be computed for the given ``t_stage``.
    """
    if t_stage_dist is not None and not np.isclose(sum(t_stage_dist), 1.):
        raise ValueError("The provided t_stage_dist does not sum to 1.")

    priors = np.empty(shape=(len(samples), *model.state_dist().shape))

    for i, sample in progress.track(
        sequence=enumerate(samples),
        description="Computing priors from samples",
        total=len(samples),
    ):
        model.set_params(*sample)
        if t_stage is None:
            priors[i] = sum(
                model.state_dist(t_stage=t, mode=mode) * p
                for t, p in zip(model.t_stages, t_stage_dist)
            )
        else:
            priors[i] = model.state_dist(t_stage=t_stage, mode=mode)
    return priors


@log_state()
def store_in_hdf5(
    file_path: Path,
    array: np.ndarray,
    name: str,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Store the ``array`` in an HDF5 file at ``file_path`` under the key ``name``."""
    with h5py.File(file_path, "w") as file:
        previous = file.get(name, default=None)
        if previous is not None:
            logger.info("Overwriting previous state distributions")
            del file[name]

        dset = file.create_dataset(name, data=array)
        if attrs is not None:
            dset.attrs.update(attrs)


def main(args: argparse.Namespace):
    """Precompute the prior state distribution for each sample."""
    params = load_yaml_params(args.params)
    model = create_model(params)
    samples = load_model_samples(args.samples, flat=True)
    priors = compute_priors_from_samples(
        model=model,
        samples=samples,
        t_stage=args.t_stage,
        t_stage_dist=args.t_stage_dist,
    )
    attrs = {"mode": str(args.mode)}
    if args.t_stage_dist is not None:
        attrs["t_stage_dist"] = args.t_stage_dist
    else:
        attrs["t_stage"] = args.t_stage
    store_in_hdf5(
        file_path=args.priors,
        array=priors,
        name=str(args.mode) + "_" + str(args.t_stage),
        attrs=attrs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
