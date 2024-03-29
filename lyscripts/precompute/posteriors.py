"""
Compute posterior state distributions from precomputed priors (see
:py:mod:`.precompute.priors`) or from drawn samples (see :py:mod:`.sample`). The
posteriors are computed for a given "scenario" (or many of them), which typically
define a clinical diagnosis w.r.t. the lymphatic involvement of a patient.
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from lymph import models, types
from rich import progress

from lyscripts import utils
from lyscripts.precompute.priors import compute_priors_using_cache
from lyscripts.precompute.utils import HDF5FileCache
from lyscripts.scenario import add_scenario_arguments

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
        "-s", "--samples", type=Path, required=False,
        help="Path to the drawn samples (HDF5 file)."
    )
    parser.add_argument(
        "--priors", type=Path, required=False,
        help=(
            "Path to the prior state distributions (HDF5 file). If samples are "
            "provided, this will be used as output to store the computed posteriors. "
            "If no samples are provided, this will be used as input to load the priors."
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


def scenario_and_modalities_from_stdin(
    args: argparse.Namespace,
    lnl_names: list[str],
    mod_name: str = "tmp",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create scenarios and modalities from the stdin arguments."""
    scenario = {
        "t_stage": args.t_stage,
        "t_stage_dist": args.t_stage_dist,
        "midext": args.midext,
        "mode": args.mode,
        "diagnosis": {
            "ipsi": {mod_name: utils.make_pattern(args.ipsi_diagnosis, lnl_names)},
            "contra": {mod_name: utils.make_pattern(args.contra_diagnosis, lnl_names)},
        },
    }
    modalities = {mod_name: {"spec": args.spec, "sens": args.sens, "kind": args.kind}}
    return scenario, modalities


def get_modality_subset(scenario: dict[str, Any]) -> set[str]:
    """Get the subset of modalities used in the ``scenario``.

    >>> scenario = {"diagnosis": {
    ...     "ipsi": {
    ...         "MRI": {"II": True, "III": False},
    ...         "PET": {"II": False, "III": True},
    ...      },
    ...     "contra": {"MRI": {"II": False, "III": None}},
    ... }}
    >>> sorted(get_modality_subset(scenario))
    ['MRI', 'PET']
    """
    modality_set = set()
    diagnosis = scenario["diagnosis"]

    for side in ["ipsi", "contra"]:
        if side in diagnosis:
            modality_set |= set(diagnosis[side].keys())

    return modality_set


def compute_posteriors_from_priors(
    model: types.Model,
    priors: np.ndarray,
    diagnosis: types.DiagnosisType | dict[str, types.DiagnosisType],
    progress_desc: str = "Computing posteriors from priors",
    midext: bool | None = None,
) -> np.ndarray:
    """Compute posteriors from prior state distributions.

    This will call the ``model`` method :py:meth:`~lymph.types.Model.posterior_state_dist`
    for each of the ``priors``, given the specified ``diagnosis`` pattern.
    """
    posteriors = np.empty(shape=priors.shape)
    if isinstance(model, models.Midline):
        kwargs = {"midext": midext}
    else:
        kwargs = {}

    for i, prior in progress.track(
        sequence=enumerate(priors),
        description=progress_desc,
        total=len(priors),
    ):
        posteriors[i] = model.posterior_state_dist(
            given_state_dist=prior,
            given_diagnosis=diagnosis,
            **kwargs,
        )
    return posteriors


def compute_posteriors_using_cache(
    model: types.Model,
    scenario: dict[str, Any],
    modalities: dict[str, Any],
    side: str = "ipsi",
    samples: np.ndarray | None = None,
    priors_cache: HDF5FileCache | None = None,
    posteriors_cache: HDF5FileCache | None = None,
    progress_desc: str = "Computing posteriors from priors",
) -> np.ndarray:
    """Compute posteriors from prior state distributions.

    This will call the ``model`` method :py:meth:`~lymph.types.Model.posterior_state_dist`
    for each of the ``priors``, given the specified ``diagnosis`` pattern.
    """
    expected_keys = ["mode", "t_stage", "t_stage_dist", "midext", "diagnosis"]
    scenario = {k: scenario.get(k) for k in expected_keys}
    posterior_hash = hashlib.md5(str(scenario).encode()).hexdigest()

    if posteriors_cache is not None and posterior_hash in posteriors_cache:
        logger.info("Posteriors already computed. Skipping.")
        posteriors, _ = posteriors_cache[posterior_hash]
        return posteriors
    elif posteriors_cache is None:
        logger.warning("No persistent posteriors cache provided.")
        posteriors_cache = {}

    expected_keys = ["mode", "t_stage", "t_stage_dist"]
    prior_scenario = {k: scenario.get(k) for k in expected_keys}
    priors = compute_priors_using_cache(
        model=model,
        samples=samples,
        priors_cache=priors_cache,
        progress_desc=progress_desc.replace("posteriors", "priors"),
        **prior_scenario,
    )

    subset = get_modality_subset(scenario)
    utils.assign_modalities(model=model, config=modalities, subset=subset)

    kwargs = {"midext": scenario["midext"]} if isinstance(model, models.Midline) else {}
    is_uni = isinstance(model, models.Unilateral)
    diagnosis = scenario["diagnosis"][side] if is_uni else scenario["diagnosis"]

    posteriors = np.empty(shape=priors.shape)
    for i, prior in progress.track(
        sequence=enumerate(priors),
        description="[blue]INFO     [/blue]" + progress_desc,
        total=len(priors),
    ):
        posteriors[i] = model.posterior_state_dist(
            given_state_dist=prior,
            given_diagnosis=diagnosis,
            **kwargs,
        )

    posteriors_cache[posterior_hash] = (posteriors, scenario)
    return posteriors


def main(args: argparse.Namespace) -> None:
    """Compute posteriors from priors or drawn samples."""
    if args.samples is None and args.priors is None:
        raise ValueError("Either --samples or --priors must be provided.")

    params = utils.load_yaml_params(args.params)
    lnl_names = params["graph"]["lnl"].keys()
    model = utils.create_model(params)
    samples = utils.load_model_samples(args.samples) if args.samples else None

    if args.scenarios is None:
        # create a single scenario from the stdin arguments...
        scenario, modalities = scenario_and_modalities_from_stdin(args, lnl_names)
        scenarios = [scenario]
        num_scens = len(scenarios)
    else:
        # ...or load the scenarios from a YAML file
        scenarios = utils.load_yaml_params(args.scenarios)["scenarios"]
        modalities = params["modalities"]
        num_scens = len(scenarios)
        logger.info(f"Using {num_scens} loaded scenarios. May ignore some arguments.")

    if args.priors is None:
        logger.warning("No persistent priors cache provided.")
        priors_cache = {}
    else:
        priors_cache = HDF5FileCache(args.priors)
    posteriors_cache = HDF5FileCache(args.posteriors)

    for i, scenario in enumerate(scenarios):
        _posteriors = compute_posteriors_using_cache(
            model=model,
            scenario=scenario,
            modalities=modalities,
            side=params["model"].get("side", "ipsi"),
            samples=samples,
            priors_cache=priors_cache,
            posteriors_cache=posteriors_cache,
            progress_desc=f"Computing posteriors for scenario {i + 1}/{num_scens}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
