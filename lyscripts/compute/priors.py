"""
Given samples drawn during an MCMC round, compute the (prior) state distribution for
each sample. This may then later on be used to compute risks and prevalences more
quickly.

The computed priors are stored in an HDF5 file under a hash key of the scenario they
were computed for. This scenario consists of the T-stages it was computed for and the
distribution that was used to marginalize over them, as well as the model's computation
mode (hidden Markov model or Bayesian network).
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from lymph import types
from rich import progress

from lyscripts import utils
from lyscripts.compute.utils import HDF5FileCache
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
        "--samples", type=Path, required=True,
        help="Path to the drawn samples (HDF5 file)."
    )
    parser.add_argument(
        "--priors", type=Path, required=True,
        help="Path to file for storing the computed prior distributions."
    )
    parser.add_argument(
        "--params", type=Path, required=True,
        help="Path to parameter file defining the model (YAML)."
    )
    parser.add_argument(
        "--scenarios", type=Path, required=False,
        help=(
            "Path to a YAML file containing a `scenarios` key with a list of "
            "diagnosis scenarios to compute the posteriors for."
        )
    )

    add_scenario_arguments(parser, for_comp="priors")
    parser.set_defaults(run_main=main)


def compute_priors_using_cache(
    model: types.Model,
    cache: HDF5FileCache,
    samples: np.ndarray | None = None,
    scenario: Scenario | None = None,
    cache_hit_msg: str = "Priors already computed. Skipping.",
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
    if priors_hash in cache:
        logger.info(cache_hit_msg)
        priors, _ = cache[priors_hash]
        return priors

    if samples is None:
        raise ValueError("No samples provided.")

    priors = []

    for sample in progress.track(
        sequence=samples,
        description="[blue]INFO     [/blue]" + progress_desc,
        total=len(samples),
    ):
        model.set_params(*sample)
        priors.append(sum(
            model.state_dist(t_stage=t, mode=scenario.mode) * p
            for t, p in zip(scenario.t_stages, scenario.t_stages_dist)
        ))

    priors = np.stack(priors)
    cache[priors_hash] = (priors, scenario.as_dict("priors"))
    return priors


def main(args: argparse.Namespace):
    """compute the prior state distribution for each sample."""
    params = utils.load_yaml_params(args.params)

    if args.scenarios is None:
        # create a single scenario from the stdin arguments...
        scenarios = [Scenario.from_namespace(args)]
        num_scens = len(scenarios)
    else:
        # ...or load the scenarios from a YAML file
        scenarios = Scenario.list_from_params(utils.load_yaml_params(args.scenarios))
        num_scens = len(scenarios)
        logger.info(f"Using {num_scens} loaded scenarios. May ignore some arguments.")

    model = utils.create_model(params)
    samples = utils.load_model_samples(args.samples)
    priors_cache = HDF5FileCache(args.priors)

    for i, scenario in enumerate(scenarios):
        _priors = compute_priors_using_cache(
            model=model,
            cache=priors_cache,
            samples=samples,
            scenario=scenario,
            progress_desc=f"Computing priors for scenario {i + 1}/{num_scens}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
