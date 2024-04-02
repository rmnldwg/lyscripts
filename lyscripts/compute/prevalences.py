"""
Predict prevalences of observed involvement pattern using the samples or prior state
distributions that were previously inferred or computed. These computed prevalences can
be compared to the prevalence of the respective pattern in the data, if provided.

This may make use of :py:mod:`.compute.priors` to compute and cache the prior state
distributions in an HDF5 file. The prevalences themselves are also stored in an HDF5
file.

Formally, the prevalence is the likelihood of the observed involvement pattern that we
are interested in, given the model and samples. We compute this by calling the model's
:py:meth:`~lymph.types.Model.state_dist` method for each of the samples and mutiply
it with the :py:func:`lymph.matrix.generate_observation` to get the likelihood of the
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
from lymph import models, types
from rich.progress import track

from lyscripts import utils
from lyscripts.compute.priors import compute_priors_using_cache
from lyscripts.compute.utils import HDF5FileCache, get_modality_subset
from lyscripts.data import accessor  # nopycln: import
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
        "--priors", type=Path, required=True,
        help=(
            "Path to the prior state distributions (HDF5 file). If samples are "
            "provided, this will be used as output to store the computed posteriors. "
            "If no samples are provided, this will be used as input to load the priors."
        )
    )
    parser.add_argument(
        "--prevalences", type=Path, required=True,
        help="Path to the HDF5 file for storing the computed prevalences."
    )
    parser.add_argument(
        "--data", type=Path, required=False,
        help="Path to the patient data (CSV file)."
    )
    parser.add_argument(
        "--params", default="./params.yaml", type=Path,
        help="Path to parameter file defining the model (YAML)."
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


def does_midext_match(
    data: pd.DataFrame,
    midext: bool | None = None
) -> pd.Index:
    """Return indices of ``data`` where ``midline_ext`` of the patients matches."""
    midext_col = data["tumor", "1", "extension"]
    if midext is None:
        return pd.Series([True] * len(data))

    return midext_col == midext


def compute_observed_prevalence(
    data: pd.DataFrame,
    scenario: Scenario,
    mapping: dict[int, str] | Callable[[int], str],
) -> np.ndarray:
    """Extract prevalence defined in a ``scenario`` from the ``data``.

    ``mapping`` defines how the T-stages in the data are supposed to be mapped to the
    T-stages defined in the ``scenario``.

    Warning:
        When computing prevalences for unilateral models, the contralateral diagnosis
        will still be considered for computing the prevalence in the *data*.
    """
    modality = get_modality_subset(scenario.diagnosis).pop()
    diagnosis_pattern = scenario.get_pattern(get_from="diagnosis", modality=modality)

    data.ly.map_t_stage(mapping)
    has_t_stage = data.ly.t_stage.isin(scenario.t_stages)
    eligible_data = data.loc[has_t_stage].reset_index()

    # filter the data by the involvement, which includes the involvement pattern itself
    # and the midline extension status
    has_midext = eligible_data.ly.is_midext(scenario.midext)
    does_pattern_match = eligible_data.ly.match(diagnosis_pattern, modality)

    try:
        matching_data = eligible_data.loc[does_pattern_match & has_midext]
        len_matching_data = len(matching_data)
    except KeyError:
        # return X, X if no actual pattern was selected
        len_matching_data = len(eligible_data)

    return len_matching_data, len(eligible_data)


def observe_prevalence_using_cache(
    data: pd.DataFrame,
    scenario: Scenario,
    cache: HDF5FileCache,
    mapping: dict[int, Any] | Callable[[int], Any] = None
):
    """Compute and cache the observed prevalence for a given ``scenario``."""
    num_match, num_total = compute_observed_prevalence(
        data=data.copy(),
        scenario=scenario,
        mapping=mapping,
    )
    scenario_hash = scenario.md5_hash("prevalences")
    with h5py.File(cache.file_path, "a") as file:
        file[scenario_hash].attrs["num_match"] = num_match
        file[scenario_hash].attrs["num_total"] = num_total


def compute_prevalences_using_cache(
    model: types.Model,
    scenario: dict[str, Any],
    priors_cache: HDF5FileCache,
    prevalences_cache: HDF5FileCache,
    cache_hit_msg: str = "Prevalences already computed. Skipping.",
    progress_desc: str = "Computing prevalences from priors",
) -> np.ndarray:
    """Compute prevalences from ``priors``."""
    if len(model.get_all_modalities()) != 1:
        raise ValueError(
            "Exactly one modality must be set in the model for computing prevalences."
        )
    modality = next(iter(model.get_all_modalities()))
    diagnosis_pattern = scenario.get_pattern(get_from="diagnosis", modality=modality)

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
            cache_hit_msg="Loaded computed priors.",
        )
    except ValueError as val_err:
        msg = "No computed priors found for the given scenario."
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
    lnls = list(params["graph"]["lnl"].keys())
    data = utils.load_patient_data(args.data) if args.data is not None else None

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
    prevalences_cache = HDF5FileCache(args.prevalences)

    for i, scenario in enumerate(scenarios):
        utils.assign_modalities(
            model=model,
            config=params["modalities"],
            subset=get_modality_subset(scenario.diagnosis),
            clear=True,
        )

        _predicted_prevalences = compute_prevalences_using_cache(
            model=model,
            scenario=scenario,
            priors_cache=priors_cache,
            prevalences_cache=prevalences_cache,
            progress_desc=f"Computing prevalences for scenario {i+1}/{num_scens}",
        )

        if data is None:
            continue

        logger.info(f"Compute observed prevalence for scenario {i+1}/{num_scens}.")
        observe_prevalence_using_cache(
            data=data,
            scenario=scenario,
            cache=prevalences_cache,
            mapping=params["model"].get("mapping", None),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
