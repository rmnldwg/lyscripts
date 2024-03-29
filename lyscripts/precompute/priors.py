"""
Given samples drawn during an MCMC round, precompute the (prior) state distribution for
each sample. This may then later on be used to compute risks and prevalences more
quickly.

The computed priors are stored in an HDF5 file under the key ``mode + t_stage``.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from lymph import types
from rich import progress

from lyscripts import utils
from lyscripts.precompute.utils import HDF5FileCache
from lyscripts.scenario import Scenario, add_scenario_arguments

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

    add_scenario_arguments(parser, for_comp="priors")
    parser.set_defaults(run_main=main)


def compute_priors_using_cache(
    model: types.Model,
    samples: np.ndarray | None = None,
    cache: HDF5FileCache | None = None,
    scenario: Scenario | None = None,
    progress_desc: str = "Computing priors from samples",
) -> np.ndarray:
    """Compute prior state distributions from the ``samples`` for the ``model``.

    This will call the ``model`` method :py:meth:`~lymph.types.Model.state_dist`
    for each of the ``samples``. If ``t_stage`` is not provided, the priors will be
    computed by marginalizing over the provided ``t_stage_dist``. Otherwise, the
    priors will be computed for the given ``t_stage``.
    """
    if scenario is None:
        scenario = Scenario()

    priors_hash = scenario.md5_hash("priors")
    if cache is not None and priors_hash in cache:
        logger.info("Priors already computed. Skipping.")
        priors, _ = cache[priors_hash]
        return priors
    elif cache is None:
        logger.warning("No persistent priors cache provided.")
        cache = {}

    if samples is None:
        raise ValueError("No samples provided.")

    priors = np.empty(shape=(len(samples), *model.state_dist().shape))

    for i, sample in progress.track(
        sequence=enumerate(samples),
        description="[blue]INFO     [/blue]" + progress_desc,
        total=len(samples),
    ):
        model.set_params(*sample)
        priors[i] = sum(
            model.state_dist(t_stage=t, mode=scenario.mode) * p
            for t, p in zip(scenario.t_stages, scenario.t_stages_dist)
        )

    cache[priors_hash] = (priors, scenario.as_dict("priors"))
    return priors


def main(args: argparse.Namespace):
    """Precompute the prior state distribution for each sample."""
    params = utils.load_yaml_params(args.params)

    _priors = compute_priors_using_cache(
        model=utils.create_model(params),
        samples=utils.load_model_samples(args.samples),
        cache=HDF5FileCache(args.priors),
        scenario=Scenario.from_namespace(args),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
