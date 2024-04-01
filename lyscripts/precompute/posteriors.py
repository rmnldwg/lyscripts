"""
Compute posterior state distributions from precomputed priors (see
:py:mod:`.precompute.priors`). The posteriors are computed for a given "scenario" (or
many of them), which typically define a clinical diagnosis w.r.t. the lymphatic
involvement of a patient.

Warning:
    The command skips the computation of the priors if it finds them in the cache. But
    this cache only accounts for the scenario, *NOT* the samples. So, if the samples
    change, you need to force a recomputation of the priors (e.g., by deleting them).
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
from lymph import models, types
from rich import progress

from lyscripts import utils
from lyscripts.precompute.priors import compute_priors_using_cache
from lyscripts.precompute.utils import HDF5FileCache, get_modality_subset
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
            "Path to the precomputed priors (HDF5 file). They must have been "
            "computed from the same model and scenarios as the posteriors."
        )
    )
    parser.add_argument(
        "--params", default="./params.yaml", type=Path,
        help="Path to parameter file defining the model (YAML)."
    )
    parser.add_argument(
        "--posteriors", type=Path, required=True,
        help="Path to file for storing the computed posterior distributions."
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


def load_from_hdf5(file_path: Path, name: str) -> np.ndarray:
    """Load an array from an HDF5 file."""
    with h5py.File(file_path, mode="r") as file:
        return file[name][()]


def compute_posteriors_using_cache(
    model: types.Model,
    scenario: Scenario,
    priors_cache: HDF5FileCache,
    posteriors_cache: HDF5FileCache,
    side: str = "ipsi",
    cache_hit_msg: str = "Posteriors already computed. Skipping.",
    progress_desc: str = "Computing posteriors from priors",
) -> np.ndarray:
    """Compute posteriors from prior state distributions.

    This will call the ``model`` method :py:meth:`~lymph.types.Model.posterior_state_dist`
    for each of the ``priors``, given the specified ``diagnosis`` pattern.
    """
    if isinstance(model, models.Unilateral):
        scenario = scenario.for_side(side)

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
            cache_hit_msg="Loaded precomputed priors.",
        )
    except ValueError as val_err:
        msg = "No precomputed priors found for the given scenario."
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
        scenarios = [Scenario.from_namespace(args, lnls=lnls)]
        num_scens = len(scenarios)
    else:
        # ...or load the scenarios from a YAML file
        scenarios = Scenario.from_params(utils.load_yaml_params(args.scenarios))
        num_scens = len(scenarios)
        logger.info(f"Using {num_scens} loaded scenarios. May ignore some arguments.")

    if args.priors is None:
        logger.warning("No persistent priors cache provided.")
        priors_cache = {}
    else:
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
            side=params["model"].get("side", "ipsi"),
            priors_cache=priors_cache,
            posteriors_cache=posteriors_cache,
            progress_desc=f"Computing posteriors for scenario {i + 1}/{num_scens}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
