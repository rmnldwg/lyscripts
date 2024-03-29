"""
Predict prevalences of observed involvement pattern using the samples or prior state
distributions that were previously inferred or computed. These computed prevalences can
be compared to the prevalence of the respective pattern in the data, if provided.

This may make use of :py:mod:`.predict.priors` to precompute and cache the prior state
distributions in an HDF5 file. The prevalences themselves are also stored in an HDF5
file.

Formally, the prevalence is the likelihood of the observed involvement pattern that we
are interested in, given the model and samples. We compute this by calling the model's
:py:meth:`~lymph.types.Model.state_dist` method for each of the samples and mutiply
it with the :py:func:`~lymph.matrix.observation_matrix` to get the likelihood of the
observed involvement pattern.
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from lymph import models, types, utils
from rich.progress import track

from lyscripts import utils
from lyscripts.precompute.priors import compute_priors_using_cache
from lyscripts.precompute.utils import HDF5FileCache
from lyscripts.predict.utils import complete_pattern
from lyscripts.scenario import add_scenario_arguments

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
        "-s", "--samples", type=Path, required=False,
        help="Path to the drawn samples (HDF5 file)."
    )
    parser.add_argument(
        "-p", "--params", default="./params.yaml", type=Path,
        help="Path to parameter file defining the model (YAML)."
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
        "-d", "--data", type=Path, required=False,
        help="Path to the patient data (CSV file)."
    )
    parser.add_argument(
        "--prevalences", type=Path, required=True,
        help="Path to the HDF5 file for storing the computed prevalences."
    )
    parser.add_argument(
        "--scenarios", type=Path, required=False,
        help=(
            "Path to a YAML file containing a `scenarios` key with a list of "
            "involvement scenarios to compute the posteriors for."
        )
    )

    add_scenario_arguments(parser, for_comp="prevalences")
    parser.set_defaults(run_main=main)


def get_match_idx(
    match_idx,
    pattern: dict[str, bool | None],
    data: pd.DataFrame,
    lnls: list[str],
    invert: bool = False,
) -> pd.Series:
    """Get indices of rows in the ``data`` where the diagnosis matches the ``pattern``.

    This uses the ``match_idx`` as a starting point and updates it according to the
    ``pattern``. If ``invert`` is set to ``True``, the function returns the inverted
    indices.

    >>> pattern = {"II": True, "III": None}
    >>> lnls = ["II", "III"]
    >>> data = pd.DataFrame.from_dict({
    ...     "II":  [True, False],
    ...     "III": [False, False],
    ... })
    >>> get_match_idx(True, pattern, data, lnls)
    0     True
    1    False
    Name: II, dtype: bool
    """
    for lnl in lnls:
        if lnl not in pattern or pattern[lnl] is None:
            continue
        if invert:
            match_idx |= data[lnl] != pattern[lnl]
        else:
            match_idx &= data[lnl] == pattern[lnl]

    return match_idx


def does_t_stage_match(data: pd.DataFrame, t_stages: list[str] | None) -> pd.Series:
    """Return indices of the ``data`` where ``t_stage`` of the patients matches."""
    if t_stages is None:
        return pd.Series([True] * len(data))
    return data["tumor", "1", "t_stage"].isin(t_stages)


def does_midline_ext_match(
    data: pd.DataFrame,
    midext: bool | None = None
) -> pd.Index:
    """Return indices of ``data`` where ``midline_ext`` of the patients matches."""
    midext_col = data["tumor", "1", "extension"]
    if midext is None:
        return midext_col.isna()

    return midext_col == midext


def get_midline_ext_prob(data: pd.DataFrame, t_stage: str) -> float:
    """Get the prevalence of midline extension from ``data`` for ``t_stage``."""
    if data.columns.nlevels == 2:
        return None

    has_matching_t_stage = does_t_stage_match(data, [t_stage])
    eligible_data = data[has_matching_t_stage]
    has_matching_midline_ext = does_midline_ext_match(eligible_data, midext=True)
    matching_data = eligible_data[has_matching_midline_ext]
    return len(matching_data) / len(eligible_data)


def compute_observed_prevalence(
    involvement: dict[str, dict[str, bool]],
    data: pd.DataFrame,
    lnls: list[str],
    t_stage: int | str | None = None,
    midext: bool | None = None,
    modality: str = "max_llh",
    **_kwargs,
):
    """Extract prevalence of a ``pattern`` from the ``data``.

    This also considers the ``t_stage`` of the patients. If the ``data`` contains
    bilateral information, one can choose to factor in whether or not the patient's
    ``midline_ext`` should be considered as well.

    By giving a list of ``lnls``, one can restrict the matching algorithm to only those
    lymph node levels that are provided via this list.

    When ``invert`` is set to ``True``, the function returns 1 minus the prevalence.
    """
    involvement = complete_pattern(involvement, lnls)
    if t_stage is not None:
        t_stages = [0,1,2] if t_stage == "early" else [3,4]
    else:
        t_stages = None
    has_t_stage = does_t_stage_match(data, t_stages)
    eligible_data = data.loc[has_t_stage].reset_index()

    # filter the data by the involvement, which includes the involvement pattern itself
    # and the midline extension status
    has_midext = does_midline_ext_match(eligible_data, midext)
    do_lnls_match = pd.Series([True] * len(eligible_data))
    for side in ["ipsi", "contra"]:
        do_lnls_match = get_match_idx(
            do_lnls_match,
            involvement[side],
            eligible_data[modality, side],
            lnls=lnls,
        )

    try:
        matching_data = eligible_data.loc[do_lnls_match & has_midext]
        len_matching_data = len(matching_data)
    except KeyError:
        # return X, X if no actual pattern was selected
        len_matching_data = len(eligible_data)

    return len_matching_data, len(eligible_data)


def compute_prevalences_using_cache(
    model: types.Model,
    scenario: dict[str, Any],
    side: str = "ipsi",
    samples: np.ndarray | None = None,
    priors_cache: HDF5FileCache | None = None,
    prevalences_cache: HDF5FileCache | None = None,
    progress_desc: str = "Computing prevalences from priors",
) -> np.ndarray:
    """Compute prevalences from ``priors``."""
    expected_keys = ["mode", "t_stage", "t_stage_dist"]
    prior_scenario = {k: scenario.get(k) for k in expected_keys}
    priors = compute_priors_using_cache(
        model=model,
        samples=samples,
        priors_cache=priors_cache,
        progress_desc=progress_desc.replace("prevalences", "priors"),
        **prior_scenario,
    )

    if len(model.get_modalities()) != 1:
        raise ValueError("Exactly one modality must be set in the model.")

    is_uni = isinstance(model, models.Unilateral)
    scenario_hash = hashlib.md5(str(scenario).encode()).hexdigest()

    if scenario_hash in prevalences_cache:
        logger.info("Prevalences already computed. Skipping.")
        prevalences, _ = prevalences_cache[scenario_hash]
    else:
        involvement = scenario["involvement"]
        prevalences = np.empty(shape=(priors.shape[0],))
        for i, prior in track(
            sequence=enumerate(priors),
            description="[blue]INFO     [/blue]" + progress_desc,
            total=len(priors),
        ):
            obs_dist = model.obs_dist(given_state_dist=prior)
            # marginalizing the distribution over observational states is not quite the
            # intended use of the method. But as long as exactly one modality is set in
            # the model, this should work as expected.
            prevalences[i] = model.marginalize(
                involvement=involvement[side] if is_uni else involvement,
                given_state_dist=obs_dist,
                midext=scenario["midext"],
            )

    prevalences_cache[scenario_hash] = (prevalences, scenario)
    return prevalences


def main(args: argparse.Namespace):
    """Function to run the risk prediction routine."""
    if args.samples is None and args.priors is None:
        raise ValueError("Either samples or priors must be provided.")

    params = utils.load_yaml_params(args.params)
    defined_modalities = params["modalities"]
    lnl_names = params["graph"]["lnl"].keys()
    model = utils.create_model(params)
    side = params["model"].get("side", "ipsi")
    samples = utils.load_model_samples(args.samples) if args.samples else None
    data = utils.load_patient_data(args.data)

    if args.modality is not None:
        try:
            modality = {args.modality: defined_modalities[args.modality]}
        except KeyError:
            modality = {args.modality: {
                "spec": args.spec,
                "sens": args.sens,
                "kind": args.kind,
            }}

    if args.scenarios is None:
        scenario = {
            "mode": args.mode,
            "t_stage": args.t_stage,
            "t_stage_dist": args.t_stage_dist,
            "midext": args.midext,
            "involvement": {
                "ipsi": utils.make_pattern(args.ipsi_involvement, lnl_names),
                "contra": utils.make_pattern(args.contra_involvement, lnl_names),
            },
        }
        scenarios = [scenario]
        num_scens = len(scenarios)
    else:
        scenarios = utils.load_yaml_params(args.scenarios)["scenarios"]
        num_scens = len(scenarios)
        logger.info(f"Loaded {num_scens} scenarios. May ignore some arguments.")

    if args.priors is None:
        logger.warning("No persistent priors cache provided.")
        priors_cache = {}
    else:
        priors_cache = HDF5FileCache(args.priors)
    prevalences_cache = HDF5FileCache(args.prevalences)

    for i, scenario in enumerate(scenarios):
        if (scen_mod_name := scenario.get("modality")) is not None:
            scen_mod = {scen_mod_name: defined_modalities[scen_mod_name]}
        else:
            scen_mod = modality
        utils.assign_modalities(model=model, config=scen_mod, clear=True)

        _predicted_prevalences = compute_prevalences_using_cache(
            model=model,
            scenario=scenario,
            side=side,
            samples=samples,
            priors_cache=priors_cache,
            prevalences_cache=prevalences_cache,
            progress_desc=f"Computing prevalences for scenario {i+1}/{num_scens}",
        )
        expected_keys = ["t_stage", "midext", "involvement"]
        obs_scenario = {k: scenario.get(k) for k in expected_keys}
        logger.info(f"Computing observed prevalence for scenario {i+1}/{num_scens}.")
        num_match, num_total = compute_observed_prevalence(
            data=data,
            lnls=lnl_names,
            modality=list(scenario.get("modality", modality).keys())[0],
            **obs_scenario,
        )
        scenario_hash = hashlib.md5(str(scenario).encode()).hexdigest()
        with h5py.File(prevalences_cache.file_path, "a") as file:
            file[scenario_hash].attrs["num_match"] = num_match
            file[scenario_hash].attrs["num_total"] = num_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
