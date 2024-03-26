"""
Compute posterior state distributions from precomputed priors (see
:py:mod:`.precompute.priors`) or from drawn samples (see :py:mod:`.sample`). The
posteriors are computed for a given "scenario" (or many of them), which typically
define a clinical diagnosis w.r.t. the lymphatic involvement of a patient.
"""
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
from lyscripts.precompute.priors import compute_priors_from_samples
from lyscripts.precompute.utils import HDF5FileCache, str_dict

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


def optional_bool(value: str) -> bool | None:
    """Convert a string to a boolean or ``None``.

    ``None`` is returned when the string is one of
    ``["none", "unknown", "na", "?", "x"]``. Everything else is just passed to
    ``bool(value)``.
    """
    if value.lower() in ["none", "unknown", "na", "?", "x"]:
        return None

    if value.lower() in ["true", "t", "yes", "y", "involved", "metastatic"]:
        return True

    if value.lower() in ["false", "f", "no", "n", "healthy", "benign"]:
        return False

    raise ValueError(f"Could not convert '{value}' to a boolean or None.")


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
        "--posteriors", default="./posteriors.hdf5", type=Path,
        help="Path to file for storing the computed posterior distributions."
    )
    parser.add_argument(
        "--scenarios", type=Path, required=False,
        help=(
            "Path to a YAML file containing a `scenarios` key with a list of "
            "diagnosis scnearios to compute the posteriors for."
        )
    )

    parser.add_argument(
        "--spec", type=float, default=0.76,
        help="Specificity of the diagnostic modality to compute the posterior with."
    )
    parser.add_argument(
        "--sens", type=float, default=0.81,
        help="Sensitivity of the diagnostic modality to compute the posterior with."
    )
    parser.add_argument(
        "--kind", choices=["clinical", "pathological"], default="clinical",
        help="Kind of diagnostic modality to compute the posterior with."
    )
    parser.add_argument(
        "--ipsi-diagnose", nargs="+", type=optional_bool,
        help=(
            "Provide the ipsilateral diagnosis as an involvement pattern of "
            "True/False/None for each LNL. Will be ignored for contralateral only "
            "models."
        )
    )
    parser.add_argument(
        "--contra-diagnose", nargs="+", type=optional_bool,
        help=(
            "Provide the contralateral diagnosis as an involvement pattern of "
            "True/False/None for each LNL. Will be ignored for ipsilateral only "
            "models."
        ),
    )
    t_or_dist_group = parser.add_mutually_exclusive_group()
    t_or_dist_group.add_argument(
        "--t-stage", type=str,
        help="T-stage to compute the posterior for. Only used with samples."
    )
    t_or_dist_group.add_argument(
        "--t-stage-dist", type=float, nargs="+",
        help="Distribution to marginalize over unknown T-stages. Only used with samples."
    )
    parser.add_argument(
        "--midext", type=bool, default=None,
        help="Midline extension of the tumor. Only used with the Midline model."
    )
    parser.add_argument(
        "--mode", choices=["HMM", "BN"], default="HMM",
        help="Mode of the model to use for the computation. Only used with samples."
    )

    parser.set_defaults(run_main=main)


def load_from_hdf5(file_path: Path, name: str) -> np.ndarray:
    """Load an array from an HDF5 file."""
    with h5py.File(file_path, mode="r") as file:
        return file[name][()]


def create_pattern_dict(
    from_list: list[bool | None] | None,
    lnls: list[str],
) -> dict[str, bool | None]:
    """Create a dictionary from a list of bools and Nones."""
    if from_list is None:
        return {lnl: None for lnl in lnls}

    return {lnl: value for lnl, value in zip(lnls, from_list)}


def get_modality_subset(scenario: dict[str, Any]) -> set[str]:
    """Get the subset of modalities used in the ``scenario``.

    >>> scenario = {"diagnose": {
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
    diagnose = scenario["diagnose"]

    for side in ["ipsi", "contra"]:
        if side in diagnose:
            modality_set |= set(diagnose[side].keys())

    return modality_set


def compute_posteriors_from_priors(
    model: types.ModelT,
    priors: np.ndarray,
    diagnose: types.DiagnoseType | dict[str, types.DiagnoseType],
    progress_desc: str = "Computing posteriors from priors",
    midext: bool | None = None,
) -> np.ndarray:
    """Compute posteriors from prior state distributions.

    This will call the ``model`` method :py:meth:`~lymph.types.Model.posterior_state_dist`
    for each of the ``priors``, given the specified ``diagnose`` pattern.
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
            given_diagnoses=diagnose,
            **kwargs,
        )
    return posteriors


def main(args: argparse.Namespace) -> None:
    """Compute posteriors from priors or drawn samples."""
    params = utils.load_yaml_params(args.params)
    model = utils.create_model(params)
    is_uni = isinstance(model, models.Unilateral)
    side = params["model"].get("side", "ipsi")
    lnl_names = params["graph"]["lnl"].keys()

    if args.priors is None:
        prior_cache = {}
    else:
        prior_cache = HDF5FileCache(args.priors)
    posterior_cache = HDF5FileCache(args.posteriors)

    if args.samples is not None:
        samples = utils.load_model_samples(args.samples)

    if args.scenarios is None:
        # create a single scenario from the stdin arguments...
        scenarios = {
            "t_stage": args.t_stage,
            "t_stage_dist": args.t_stage_dist,
            "midext": args.midext,
            "mode": args.mode,
            "diagnose": {
                "ipsi": {"tmp": create_pattern_dict(args.ipsi_diagnose, lnl_names)},
                "contra": {"tmp": create_pattern_dict(args.contra_diagnose, lnl_names)},
            },
        }
        modalities = {"tmp": {"spec": args.spec, "sens": args.sens, "kind": args.kind}}
        num_scens = len(scenarios)
    else:
        # ...or load the scenarios from a YAML file
        scenarios = utils.load_yaml_params(args.scenarios)["scenarios"]
        num_scens = len(scenarios)
        logger.info(f"Using {num_scens} loaded scenarios. May ignore some arguments.")
        modalities = params["modalities"]

    for i, scenario in enumerate(scenarios):
        scenario.pop("involvement")
        scenario_attrs = {
            "t_stage": scenario.get("t_stage"),
            "t_stage_dist": scenario.get("t_stage_dist"),
            "mode": scenario.get("mode", "HMM"),
        }
        # compute a persistent hash from the scenario attributes
        prior_hash = hashlib.md5(str(scenario_attrs).encode()).hexdigest()

        if prior_hash not in prior_cache:
            # compute priors for the cache...
            priors = compute_priors_from_samples(
                model=model,
                samples=samples,
                description=f"Computing priors for scenario {i+1}/{num_scens}",
                **scenario_attrs,
            )
            prior_cache[prior_hash] = (priors, scenario_attrs)
        else:
            # ...or fetch them from the cache
            priors, attrs = prior_cache[prior_hash]
            logger.info(f"Loading priors for scenario {i+1}/{num_scens} from cache.")
            if str_dict(scenario_attrs) != attrs:
                raise RuntimeError(
                    f"Same hash {prior_hash}, but different attributes: "
                    f"{scenario_attrs} != {attrs}"
                )

        subset = get_modality_subset(scenario)
        utils.assign_modalities(modalities, model, subset=subset)
        diagnose = scenario["diagnose"]

        scenario_attrs.update(**scenario)
        posterior_hash = hashlib.md5(str(scenario_attrs).encode()).hexdigest()

        if posterior_hash not in posterior_cache:
            posteriors = compute_posteriors_from_priors(
                model=model,
                priors=priors,
                diagnose=diagnose[side] if is_uni else diagnose,
                progress_desc=f"Computing posteriors for scenario {i+1}/{num_scens}",
            )
            posterior_cache[posterior_hash] = (posteriors, scenario_attrs)
        else:
            logger.info(f"Loading posteriors for scenario {i+1}/{num_scens} from cache.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
