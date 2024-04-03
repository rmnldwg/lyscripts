"""
Predict risks of involvements using the posteriors that were computed using the
:py:mod:`.compute.posteriors` command.

The structure of these scenarios is similar to how scenarios are defined for the
:py:mod:`.compute.prevalences` script.
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import logging
from pathlib import Path

import numpy as np
from lymph import models, types
from rich.progress import track

from lyscripts import utils
from lyscripts.compute.posteriors import compute_posteriors_using_cache
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
    """Add arguments needed to run this script to a `subparsers` instance."""
    parser.add_argument(
        "--posteriors", type=Path, required=True,
        help="Path to the computed posteriors (HDF5 file)."
    )
    parser.add_argument(
        "--risks", type=Path, required=True,
        help="Path to file for storing the computed risks."
    )
    parser.add_argument(
        "--params", type=Path, required=True,
        help="Path to parameter file defining the model (YAML)."
    )
    parser.add_argument(
        "--scenarios", type=Path, required=False,
        help=(
            "Path to a YAML file containing a `scenarios` key with a list of "
            "diagnosis scenarios and involvement patterns of interest to compute the "
            "risks for."
        )
    )

    add_scenario_arguments(parser, for_comp="risks")
    parser.set_defaults(run_main=main)


def compute_risks_using_cache(
    model: types.Model,
    scenario: Scenario,
    posteriors_cache: HDF5FileCache,
    risks_cache: HDF5FileCache,
    cache_hit_msg: str = "Risks already computed. Skipping.",
    progress_desc: str = "Computing risks from posteriors",
) -> np.ndarray:
    """Compute the risks of involvements for a given scenario."""
    risks_hash = scenario.md5_hash("risks")

    if risks_hash in risks_cache:
        logger.info(cache_hit_msg)
        risks, _ = risks_cache[risks_hash]
        return risks

    try:
        posteriors = compute_posteriors_using_cache(
            model=model,
            scenario=scenario,
            priors_cache=None,
            posteriors_cache=posteriors_cache,
            cache_hit_msg="Loaded computed posteriors.",
        )
    except ValueError as val_err:
        msg = "No computed posteriors found for the given scenario."
        logger.error(msg)
        raise ValueError(msg) from val_err

    kwargs = {"midext": scenario.midext} if isinstance(model, models.Midline) else {}
    risks = []

    for posterior in track(
        posteriors,
        description="[blue]INFO     [/blue]" + progress_desc,
        total=len(posteriors),
    ):
        risks.append(model.marginalize(
            involvement=scenario.involvement,
            given_state_dist=posterior,
            **kwargs,
        ))

    risks = np.stack(risks)
    risks_cache[risks_hash] = (risks, scenario.as_dict("risks"))
    return risks


def main(args: argparse.Namespace):
    """Run the main risk prediction routine."""
    params = utils.load_yaml_params(args.params)
    model = utils.create_model(params)
    lnls = list(params["graph"]["lnl"].keys())

    if args.scenarios is None:
        # create a single scenario from the stdin arguments...
        scenarios = [Scenario.from_namespace(
            namespace=args,
            lnls=lnls,
            is_uni=isinstance(model, models.Unilateral),
            side=params["model"].get("side", "ipsi"),
        )]
        num_scens = len(scenarios)
    else:
        # ...or load the scenarios from a YAML file
        scenarios = Scenario.list_from_params(
            params=utils.load_yaml_params(args.scenarios),
            is_uni=isinstance(model, models.Unilateral),
            side=params["model"].get("side", "ipsi"),
        )
        num_scens = len(scenarios)
        logger.info(f"Using {num_scens} loaded scenarios. May ignore some arguments.")

    posteriors_cache = HDF5FileCache(args.posteriors)
    risks_cache = HDF5FileCache(args.risks)

    for i, scenario in enumerate(scenarios):
        _risks = compute_risks_using_cache(
            model=model,
            scenario=scenario,
            posteriors_cache=posteriors_cache,
            risks_cache=risks_cache,
            progress_desc=f"Computing risks for scenario {i + 1}/{num_scens}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
