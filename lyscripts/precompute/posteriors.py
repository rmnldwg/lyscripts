"""
Given samples drawn during an MCMC round, precompute the posterior state distribution
for each sample or given prior state distributon. This may then later on be used to
compute risks and prevalences more quickly.
"""
import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
from lymph import models, types
from rich import progress

from lyscripts import utils
from lyscripts.precompute.priors import (
    compute_priors_from_samples,
    store_in_hdf5,
)
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

    scenarios_or_stdin_group = parser.add_mutually_exclusive_group()
    scenarios_or_stdin_group.add_argument(
        "--scenarios", type=Path, required=False,
        help=(
            "Path to a YAML file containing a `scenarios` key with a list of "
            "diagnosis scnearios to compute the posteriors for."
        )
    )
    stdin_group = scenarios_or_stdin_group.add_argument_group(
        description="Scenario from stdin."
    )
    stdin_group.add_argument(
        "--spec", type=float, default=0.76,
        help="Specificity of the diagnostic modality to compute the posterior with."
    )
    stdin_group.add_argument(
        "--sens", type=float, default=0.81,
        help="Sensitivity of the diagnostic modality to compute the posterior with."
    )
    stdin_group.add_argument(
        "--kind", choices=["clinical", "pathological"], default="clinical",
        help="Kind of diagnostic modality to compute the posterior with."
    )
    stdin_group.add_argument(
        "--ipsi-diagnose", nargs="+", type=optional_bool,
        help=(
            "Provide the ipsilateral diagnosis as an involvement pattern of "
            "True/False/None for each LNL. Will be ignored for contralateral only "
            "models."
        )
    )
    stdin_group.add_argument(
        "--contra-diagnose", nargs="+", type=optional_bool,
        help=(
            "Provide the contralateral diagnosis as an involvement pattern of "
            "True/False/None for each LNL. Will be ignored for ipsilateral only "
            "models."
        ),
    )
    stdin_group.add_argument(
        "--label", type=str,
        help="Label for the scenario entered via stdin. Used to name the HDF5 dataset.",
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


def compute_posteriors_from_priors(
    model: types.ModelT,
    priors: np.ndarray,
    diagnoses: types.DiagnoseType | dict[str, types.DiagnoseType],
    progress_desc: str = "Computing posteriors from priors",
) -> np.ndarray:
    """Compute posteriors from prior state distributions.

    This will call the ``model`` method :py:meth:`~lymph.types.Model.posterior_state_dist`
    for each of the ``priors``, given the specified ``diagnose`` pattern.
    """
    posteriors = np.empty(shape=priors.shape)

    for i, prior in progress.track(
        sequence=enumerate(priors),
        description=progress_desc,
        total=len(priors),
    ):
        posteriors[i] = model.posterior_state_dist(
            given_state_dist=prior,
            given_diagnoses=diagnoses,
        )
    return posteriors


def main(args: argparse.Namespace) -> None:
    """Compute posteriors from priors or drawn samples."""
    if args.samples is None and args.priors is None:
        raise ValueError("Either --samples or --priors must be provided.")

    params = load_yaml_params(args.params)
    model = create_model(params)
    side = params["model"].get("side", "ipsi")
    lnl_names = params["graph"]["lnl"].keys()

    # compute or fetch priors
    if args.samples is not None:
        samples = load_model_samples(args.samples)
        priors = compute_priors_from_samples(
            model=model,
            samples=samples,
            t_stage=args.t_stage,
            t_stage_dist=args.t_stage_dist,
            mode=args.mode,
        )
        if args.priors is not None:
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
    else:
        priors = load_from_hdf5(
            file_path=args.priors,
            name=str(args.mode) + "_" + str(args.t_stage),
        )

    if args.scenarios is None:
        if not isinstance(model, models.Unilateral):
            diagnose = {
                "ipsi": {"_": create_pattern_dict(args.ipsi_diagnose, lnl_names)},
                "contra": {"_": create_pattern_dict(args.contra_diagnose, lnl_names)},
            }
        else:
            side_pattern = getattr(args, f"{side}_diagnose")
            diagnose = {"_": create_pattern_dict(side_pattern, lnl_names)}

        model.clear_modalities()
        model.set_modality("_", spec=args.spec, sens=args.sens, kind=args.kind)

        posteriors = compute_posteriors_from_priors(
            model=model,
            priors=priors,
            diagnoses=diagnose,
        )
        store_in_hdf5(
            file_path=args.posteriors,
            array=posteriors,
            name=args.label or "posteriors",
        )
    else:
        scenarios = load_yaml_params(args.scenarios)["scenarios"]
        for i, scenario in enumerate(scenarios):
            utils.assign_modalities(scenario["modalities"], model)
            posteriors = compute_posteriors_from_priors(
                model=model,
                priors=priors,
                diagnoses=scenario["diagnose"],
                progress_desc=f"Computing posteriors for scenario {i+1}/{len(scenarios)}",
            )
            store_in_hdf5(
                file_path=args.posteriors,
                array=posteriors,
                name=scenario["label"],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
