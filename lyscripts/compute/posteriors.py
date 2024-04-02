"""
Compute posterior state distributions from computed priors (see
:py:mod:`.compute.priors`). The posteriors are computed for a given "scenario" (or
many of them), which typically define a clinical diagnosis w.r.t. the lymphatic
involvement of a patient.

In the resulting HDF5 file, the posteriors are stored under MD5 hashes of the
corresponding scenarios, similar to the priors. But the posteriors take into account
more information.

Warning:
    The command skips the computation of the priors if it finds them in the cache. But
    this cache only accounts for the scenario, *NOT* the samples. So, if the samples
    change, you need to force a recomputation of the priors (e.g., by deleting them).
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import logging
from pathlib import Path

import numpy as np
from lymph import models, types
from rich import progress

from lyscripts import utils
from lyscripts.compute.priors import compute_priors_using_cache
from lyscripts.compute.utils import HDF5FileCache, get_modality_subset
from lyscripts.scenario import Scenario, add_scenario_arguments

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
        "--priors", type=Path, required=True,
        help=(
            "Path to the computed priors (HDF5 file). They must have been "
            "computed from the same model and scenarios as the posteriors."
        )
    )
    parser.add_argument(
        "--posteriors", type=Path, required=True,
        help="Path to file for storing the computed posterior distributions."
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

    add_scenario_arguments(parser, for_comp="posteriors")
    parser.set_defaults(run_main=main)


def compute_posteriors_using_cache(
    model: types.Model,
    scenario: Scenario,
    priors_cache: HDF5FileCache | None,
    posteriors_cache: HDF5FileCache,
    cache_hit_msg: str = "Posteriors already computed. Skipping.",
    progress_desc: str = "Computing posteriors from priors",
) -> np.ndarray:
    """Compute posteriors from prior state distributions.

    This will call the ``model`` method :py:meth:`~lymph.types.Model.posterior_state_dist`
    for each of the ``priors``, given the specified ``diagnosis`` pattern.
    """
    posteriors_hash = scenario.md5_hash("posteriors")

    if posteriors_hash in posteriors_cache:
        logger.info(cache_hit_msg)
        posteriors, _ = posteriors_cache[posteriors_hash]
        return posteriors

    try:
        priors = compute_priors_using_cache(
            model=model,
            cache=priors_cache,
            scenario=scenario,
            cache_hit_msg="Loaded computed priors.",
        )
    except ValueError as val_err:
        msg = "No computed priors found for the given scenario."
        logger.error(msg)
        raise ValueError(msg) from val_err

    kwargs = {"midext": scenario.midext} if isinstance(model, models.Midline) else {}
    posteriors = []

    for prior in progress.track(
        sequence=priors,
        description="[blue]INFO     [/blue]" + progress_desc,
        total=len(priors),
    ):
        posteriors.append(model.posterior_state_dist(
            given_state_dist=prior,
            given_diagnosis=scenario.diagnosis,
            **kwargs,
        ))

    posteriors = np.stack(posteriors)
    posteriors_cache[posteriors_hash] = (posteriors, scenario.as_dict("posteriors"))
    return posteriors


def main(args: argparse.Namespace) -> None:
    """Compute posteriors from priors or drawn samples."""
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

    priors_cache = HDF5FileCache(args.priors)
    posteriors_cache = HDF5FileCache(args.posteriors)

    for i, scenario in enumerate(scenarios):
        utils.assign_modalities(
            model=model,
            config=params["modalities"],
            subset=get_modality_subset(scenario.diagnosis),
            clear=True,
        )
        _posteriors = compute_posteriors_using_cache(
            model=model,
            scenario=scenario,
            priors_cache=priors_cache,
            posteriors_cache=posteriors_cache,
            progress_desc=f"Computing posteriors for scenario {i + 1}/{num_scens}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
