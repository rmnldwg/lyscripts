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

Warning:
    The command skips the computation of the priors if it finds them in the cache. But
    this cache only accounts for the scenario, *NOT* the samples. So, if the samples
    change, you need to force a recomputation of the priors (e.g., by deleting them).
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from lymph import models, types, utils
from rich.progress import track

from lyscripts import utils
from lyscripts.precompute.priors import compute_priors_using_cache
from lyscripts.precompute.utils import HDF5FileCache, get_modality_subset
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
    invert: bool = False,
) -> pd.Series:
    """Get indices of rows in the ``data`` where the diagnosis matches the ``pattern``.

    This uses the ``match_idx`` as a starting point and updates it according to the
    ``pattern``. If ``invert`` is set to ``True``, the function returns the inverted
    indices.

    >>> pattern = {"II": True, "III": None}
    >>> data = pd.DataFrame.from_dict({
    ...     "II":  [True, False],
    ...     "III": [False, False],
    ... })
    >>> get_match_idx(True, pattern, data)
    0     True
    1    False
    Name: II, dtype: bool
    """
    for lnl, involvement in data.items():
        if lnl not in pattern or pattern[lnl] is None:
            continue
        if invert:
            match_idx |= involvement != pattern[lnl]
        else:
            match_idx &= involvement == pattern[lnl]

    return match_idx


def does_t_stage_match(data: pd.DataFrame, t_stages: list[str] | None) -> pd.Series:
    """Return indices of the ``data`` where ``t_stage`` of the patients matches."""
    t_stage_col = data["tumor", "1", "t_stage"]
    if t_stages is None:
        return pd.Series([True] * len(data))

    return t_stage_col.isin(t_stages)


def does_midext_match(
    data: pd.DataFrame,
    midext: bool | None = None
) -> pd.Index:
    """Return indices of ``data`` where ``midline_ext`` of the patients matches."""
    midext_col = data["tumor", "1", "extension"]
    if midext is None:
        return pd.Series([True] * len(data))

    return midext_col == midext


def get_midext_prob(data: pd.DataFrame, t_stage: str) -> float:
    """Get the prevalence of midline extension from ``data`` for ``t_stage``."""
    if data.columns.nlevels == 2:
        return None

    has_matching_t_stage = does_t_stage_match(data, [t_stage])
    eligible_data = data[has_matching_t_stage]
    has_matching_midline_ext = does_midext_match(eligible_data, midext=True)
    matching_data = eligible_data[has_matching_midline_ext]
    return len(matching_data) / len(eligible_data)


def compute_observed_prevalence(
    data: pd.DataFrame,
    scenario: Scenario,
    mapping: dict[int, str] | Callable[[int], str],
):
    """Extract prevalence of a ``diagnosis`` from the ``data``.

    This also considers the ``t_stage`` of the patients. If the ``data`` contains
    bilateral information, one can choose to factor in whether or not the patient's
    ``midext`` should be considered as well.

    By giving a list of ``lnls``, one can restrict the matching algorithm to only those
    lymph node levels that are provided via this list.
    """
    modality = get_modality_subset(scenario.diagnosis).pop()
    diagnosis_pattern = {
        "ipsi": scenario.diagnosis["ipsi"][modality],
        "contra": scenario.diagnosis["contra"][modality],
    }

    data["tumor", "1", "t_stage"] = data["tumor", "1", "t_stage"].map(mapping)
    has_t_stage = does_t_stage_match(data, scenario.t_stages)
    eligible_data = data.loc[has_t_stage].reset_index()

    # filter the data by the involvement, which includes the involvement pattern itself
    # and the midline extension status
    has_midext = does_midext_match(eligible_data, scenario.midext)
    do_lnls_match = pd.Series([True] * len(eligible_data))

    for side in ["ipsi", "contra"]:
        do_lnls_match = get_match_idx(
            do_lnls_match,
            diagnosis_pattern[side],
            eligible_data[modality, side],
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
    priors_cache: HDF5FileCache,
    prevalences_cache: HDF5FileCache,
    cache_hit_msg = "Prevalences already computed. Skipping.",
    progress_desc: str = "Computing prevalences from priors",
) -> np.ndarray:
    """Compute prevalences from ``priors``."""
    if len(model.get_all_modalities()) != 1:
        raise ValueError("Exactly one modality must be set in the model.")

    modality = get_modality_subset(scenario.diagnosis).pop()

    if "ipsi" in scenario.diagnosis and "contra" in scenario.diagnosis:
        diagnosis_pattern = {
            "ipsi": scenario.diagnosis["ipsi"][modality],
            "contra": scenario.diagnosis["contra"][modality],
        }
    else:
        diagnosis_pattern = scenario.diagnosis[modality]

    prevalences_hash = scenario.md5_hash("prevalences")

    if prevalences_hash in prevalences_cache:
        logger.info(cache_hit_msg)
        prevalences, _ = prevalences_cache[prevalences_hash]
        return prevalences

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
    prevalences = []

    for prior in track(
        sequence=priors,
        description="[blue]INFO     [/blue]" + progress_desc,
        total=len(priors),
    ):
        obs_dist = model.obs_dist(given_state_dist=prior)
        # marginalizing the distribution over observational states is not quite the
        # intended use of the method. But as long as exactly one modality is set in
        # the model, this should work as expected, because then the observation matrix
        # is square and the `obs_dist` has the same shape as the `state_dist`.
        prevalences.append(model.marginalize(
            involvement=diagnosis_pattern,
            given_state_dist=obs_dist,
            **kwargs,
        ))

    prevalences = np.stack(prevalences)
    prevalences_cache[prevalences_hash] = (prevalences, scenario.as_dict("prevalences"))
    return prevalences


def main(args: argparse.Namespace):
    """Function to run the risk prediction routine."""
    params = utils.load_yaml_params(args.params)
    model = utils.create_model(params)
    is_uni = isinstance(model, models.Unilateral)
    side = params["model"].get("side", "ipsi")
    lnls = list(params["graph"]["lnl"].keys())
    data = utils.load_patient_data(args.data)

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
    prevalences_cache = HDF5FileCache(args.prevalences)

    for i, scenario in enumerate(scenarios):
        utils.assign_modalities(
            model=model,
            config=params["modalities"],
            subset=get_modality_subset(scenario.diagnosis),
            clear=True,
        )
        if len(model.get_all_modalities()) != 1:
            raise ValueError("Exactly one modality necessary for computing prevalences.")

        _predicted_prevalences = compute_prevalences_using_cache(
            model=model,
            scenario=scenario.for_side(side) if is_uni else scenario,
            priors_cache=priors_cache,
            prevalences_cache=prevalences_cache,
            progress_desc=f"Computing prevalences for scenario {i+1}/{num_scens}",
        )

        logger.info(f"Computing observed prevalence for scenario {i+1}/{num_scens}.")
        num_match, num_total = compute_observed_prevalence(
            data=data.copy(),
            scenario=scenario,
            mapping=params["model"].get("mapping", None),
        )
        scenario_hash = scenario.md5_hash("prevalences")
        with h5py.File(prevalences_cache.file_path, "a") as file:
            file[scenario_hash].attrs["num_match"] = num_match
            file[scenario_hash].attrs["num_total"] = num_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
