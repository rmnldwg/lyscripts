"""
Given samples drawn during an MCMC round, precompute the posterior state distribution
for each sample or given prior state distributon. This may then later on be used to
compute risks and prevalences more quickly.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from lymph import models, types

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
    ex_group = parser.add_mutually_exclusive_group(required=True)
    ex_group.add_argument(
        "-s", "--samples", type=Path,
        help="Path to the drawn samples (HDF5 file)."
    )
    ex_group.add_argument(
        "--priors", type=Path,
        help="Path to the prior state distributions (HDF5 file)."
    )
    parser.add_argument(
        "--params", default="./params.yaml", type=Path,
        help="Path to parameter file defining the model (YAML)."
    )
    parser.add_argument(
        "--spec", type=float,
        help="Specificity of the diagnostic modality to compute the posterior with."
    )
    parser.add_argument(
        "--sens", type=float,
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
    parser.add_argument(
        "--t-stage", type=str,
        help="T-stage to compute the posterior for. Only used with samples."
    )
    parser.add_argument(
        "--mode", choices=["HMM", "BN"], default="HMM",
        help="Mode of the model to use for the computation. Only used with samples."
    )
    parser.set_defaults(run_main=main)


def create_pattern_dict(
    from_list: list[bool | None],
    lnls: list[str],
) -> dict[str, bool | None]:
    """Create a dictionary from a list of bools and Nones."""
    if from_list is None:
        return {lnl: None for lnl in lnls}

    return {lnl: value for lnl, value in zip(lnls, from_list)}


def compute_posteriors_from_samples(
    model: types.ModelT,
    samples: np.ndarray,
    diagnose: types.DiagnoseType | dict[str, types.DiagnoseType],
    t_stage: str | int,
    mode: str,
) -> np.ndarray:
    """Compute posteriors from drawn samples.

    TODO: Think about marginalization over T-stages.
    """
    posteriors = np.empty(shape=(len(samples), *model.state_dist().shape))
    for i, sample in enumerate(samples):
        model.set_params(*sample)
        posteriors[i] = model.posterior_state_dist(
            given_diagnose=diagnose,
            t_stage=t_stage,
            mode=mode,
        )
    return posteriors


def main(args: argparse.Namespace) -> None:
    """Compute posteriors from priors or drawn samples."""
    params = load_yaml_params(args.params)
    model = create_model(params)
    side = params["model"].get("side", "ipsi")
    lnl_names = params["graph"]["lnl"].keys()

    if not isinstance(model, models.Unilateral):
        diagnose = {
            "ipsi": {"_": create_pattern_dict(args.ipsi_diagnose, lnl_names)},
            "contra": {"_": create_pattern_dict(args.contra_diagnose, lnl_names)},
        }
    else:
        side_pattern = getattr(args, f"{side}_diagnose")
        diagnose = {"_": create_pattern_dict(side_pattern, lnl_names)}

    model.clear_all_modalities()
    model.add_modality("_", spec=args.spec, sens=args.sens, kind=args.kind)

    if args.samples is not None:
        samples = load_model_samples(args.samples)
        posteriors = compute_posteriors_from_samples(
            model=model,
            samples=samples,
            diagnose=diagnose,
            t_stage=args.t_stage,
            mode=args.mode,
        )
    else:
        priors = load_model_samples(args.priors)
        posteriors = compute_posteriors_from_priors(
            model=model,
            priors=priors,
            diagnose=diagnose,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
