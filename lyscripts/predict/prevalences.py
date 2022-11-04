"""
Predict prevalences of diagnostic patterns using the samples that were inferred using
the model via MCMC sampling and compare them to the prevalence in the data.

This essentially amounts to computing the data likelihood under the model and comparing
it to the empirical likelihood of a given pattern of lymphatic progression.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import lymph
import numpy as np
import pandas as pd

from lyscripts.predict.utils import clean_pattern, rich_enumerate
from lyscripts.utils import (
    cli_load_model_samples,
    cli_load_yaml_params,
    flatten,
    get_lnls,
    model_from_config,
    report,
)


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """
    Add an `ArgumentParser` to the subparsers action.
    """
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """
    Add arguments needed to run this script to a `subparsers` instance
    and run the respective main function when chosen.
    """
    parser.add_argument(
        "model", type=Path,
        help="Path to drawn samples (HDF5)"
    )
    parser.add_argument(
        "data", type=Path,
        help="Path to the data file to compare prediction and data prevalence"
    )
    parser.add_argument(
        "output", type=Path,
        help="Output path for predicted prevalences (HDF5 file)"
    )
    parser.add_argument(
        "--thin", default=1, type=int,
        help="Take only every n-th sample"
    )
    parser.add_argument(
        "--params", default="./params.yaml", type=Path,
        help="Path to parameter file"
    )

    parser.set_defaults(run_main=main)


def get_match_idx(
    match_idx,
    pattern: Dict[str, Optional[bool]],
    data: pd.DataFrame,
    lnls: List[str],
    invert: bool = False,
) -> pd.Series:
    """Get the indices of the rows in the `data` where the diagnose matches the
    `pattern` of interest for every lymph node level in the `lnls`. An example:
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

def does_t_stage_match(data: pd.DataFrame, t_stage: str) -> pd.Index:
    """Return the indices of the `data` where the `t_stage` of the patients matches."""
    if data.columns.nlevels == 2:
        return data["info", "t_stage"] == t_stage
    elif data.columns.nlevels == 3:
        return data["info", "tumor", "t_stage"] == t_stage
    else:
        raise ValueError("Data has neither 2 nor 3 header rows")

def does_midline_ext_match(
    data: pd.DataFrame,
    midline_ext: Optional[bool] = None
) -> pd.Index:
    """
    Return the indices of the `data` where the `midline_ext` of the patients matches.
    """
    if midline_ext is None or data.columns.nlevels == 2:
        return True

    try:
        return data["info", "tumor", "midline_extension"] == midline_ext
    except KeyError as key_err:
        raise KeyError(
            "Data does not seem to have midline extension information"
        ) from key_err

def get_midline_ext_prob(data: pd.DataFrame, t_stage: str) -> float:
    """Get the prevalence of midline extension from `data` for `t_stage`."""
    if data.columns.nlevels == 2:
        return None

    has_matching_t_stage = does_t_stage_match(data, t_stage)
    eligible_data = data[has_matching_t_stage]
    has_matching_midline_ext = does_midline_ext_match(eligible_data, midline_ext=True)
    matching_data = eligible_data[has_matching_midline_ext]
    return len(matching_data) / len(eligible_data)

def create_patient_row(
    pattern: Dict[str, Dict[str, bool]],
    t_stage: str,
    midline_ext: Optional[bool] = None,
    make_unilateral: bool = False,
) -> pd.DataFrame:
    """
    Create a pandas `DataFrame` representing a single patient from the specified
    involvement `pattern`, along with their `t_stage` and `midline_ext` (if provided).
    If `midline_ext` is not provided, the function creates two patient rows. One of a
    patient _with_ and one of a patient _without_ a midline extention. And the returned
    `patient_row` will only contain the `ipsi` part of the pattern when one tells the
    function to `make_unilateral`.
    """
    if make_unilateral:
        flat_pattern = flatten({"prev": pattern["ipsi"]})
        patient_row = pd.DataFrame(flat_pattern, index=[0])
        patient_row["info", "t_stage"] = t_stage
        return patient_row

    flat_pattern = flatten({"prev": pattern})
    patient_row = pd.DataFrame(flat_pattern, index=[0])
    patient_row["info", "tumor", "t_stage"] = t_stage
    if midline_ext is not None:
        patient_row["info", "tumor", "midline_extension"] = midline_ext
        return patient_row

    with_midline_ext = patient_row.copy()
    with_midline_ext["info", "tumor", "midline_extension"] = True
    without_midline_ext = patient_row.copy()
    without_midline_ext["info", "tumor", "midline_extension"] = False

    return with_midline_ext.append(without_midline_ext).reset_index()

def observed_prevalence(
    pattern: Dict[str, Dict[str, bool]],
    data: pd.DataFrame,
    t_stage: str,
    lnls: List[str],
    modality: str = "max_llh",
    midline_ext: Optional[bool] = None,
    invert: bool = False,
    **_kwargs,
):
    """Extract the prevalence of a lymphatic `pattern` of progression for a given
    `t_stage` from the `data` as reported by the given `modality`.

    If the `data` contains bilateral information, one can choose to factor in whether
    or not the patient's `midline_ext` should be considered as well.

    By giving a list of `lnls`, one can restrict the matching algorithm to only those
    lymph node levels that are provided via this list.

    When `invert` is set to `True`, the function returns 1 minus the prevalence.
    """
    pattern = clean_pattern(pattern, lnls)

    has_matching_t_stage = does_t_stage_match(data, t_stage)
    has_matching_midline_ext = does_midline_ext_match(data, midline_ext)

    eligible_data = data.loc[has_matching_t_stage & has_matching_midline_ext, modality]
    eligible_data = eligible_data.dropna(axis="index", how="all")

    # filter the data by the LNL pattern they report
    do_lnls_match = False if invert else True
    if data.columns.nlevels == 2:
        do_lnls_match = get_match_idx(
            do_lnls_match,
            pattern["ipsi"],
            eligible_data,
            lnls=lnls,
            invert=invert,
        )
    else:
        for side in ["ipsi", "contra"]:
            do_lnls_match = get_match_idx(
                do_lnls_match,
                pattern[side],
                eligible_data[side],
                lnls=lnls,
                invert=invert
            )

    matching_data = eligible_data.loc[do_lnls_match]
    return len(matching_data), len(eligible_data)

def predicted_prevalence(
    pattern: Dict[str, Dict[str, bool]],
    model: Union[lymph.Unilateral, lymph.Bilateral, lymph.MidlineBilateral],
    samples: np.ndarray,
    t_stage: str,
    midline_ext: Optional[bool] = None,
    midline_ext_prob: float = 0.3,
    modality_spsn: Optional[List[float]] = None,
    invert: bool = False,
    description: Optional[str] = None,
    **_kwargs,
) -> np.ndarray:
    """Compute the prevalence of a given `pattern` of lymphatic progression using a
    `model` and trained `samples`.

    Do this computation for the specified `t_stage` and whether or not the tumor has
    a `midline_ext`. `modality_spsn` defines the values for specificity & sensitivity
    of the diagnostic modality for which the prevalence is to be computed. Default is
    a value of 1 for both.

    Use `invert` to compute 1 - p.
    """
    lnls = get_lnls(model)
    pattern = clean_pattern(pattern, lnls)

    if modality_spsn is None:
        model.modalities = {"prev": [1., 1.]}
    else:
        model.modalities = {"prev": modality_spsn}

    prevalences = np.zeros(shape=len(samples), dtype=float)
    is_unilateral = isinstance(model, lymph.Unilateral)
    patient_row = create_patient_row(
        pattern, t_stage, midline_ext, make_unilateral=is_unilateral
    )
    model.patient_data = patient_row

    # compute prevalence as likelihood of diagnose `prev`, which was defined above
    for i,sample in rich_enumerate(samples, description):
        if isinstance(model, lymph.MidlineBilateral):
            model.check_and_assign(sample)
            if midline_ext is None:
                # marginalize over patients with and without midline extension
                prevalences[i] = (
                    midline_ext_prob * model.ext.likelihood(log=False) +
                    (1. - midline_ext_prob) * model.noext.likelihood(log=False)
                )
            elif midline_ext:
                prevalences[i] = model.ext.likelihood(log=False)
            else:
                prevalences[i] = model.noext.likelihood(log=False)
        else:
            prevalences[i] = model.likelihood(
                given_params=sample,
                log=False,
            )
    return 1. - prevalences if invert else prevalences


def main(args: argparse.Namespace):
    """
    This subprogram's call signature can be obtained via `lyscripts predict
    prevalences --help` and shows this:

    ```
    USAGE: lyscripts predict prevalences [-h] [--thin THIN] [--params PARAMS]
                                         model data output

    Predict prevalences of diagnostic patterns using the samples that were inferred
    using the model via MCMC sampling and compare them to the prevalence in the data.

    This essentially amounts to computing the data likelihood under the model and
    comparing it to the empirical likelihood of a given pattern of lymphatic
    progression.

    POSITIONAL ARGUMENTS:
    model            Path to drawn samples (HDF5)
    data             Path to the data file to compare prediction and data prevalence
    output           Output path for predicted prevalences (HDF5 file)

    OPTIONAL ARGUMENTS:
    -h, --help       show this help message and exit
    --thin THIN      Take only every n-th sample (default: 1)
    --params PARAMS  Path to parameter file (default: ./params.yaml)
    ```
    """
    params = cli_load_yaml_params(args.params)
    samples = cli_load_model_samples(args.model)

    with report.status("Read in training data..."):
        # Only read in two header rows when using the Unilateral model
        is_unilateral = params["model"]["class"] == "Unilateral"
        header = [0, 1] if is_unilateral else [0, 1, 2]
        DATA = pd.read_csv(args.data, header=header)
        report.success(f"Read in training data from {args.data}")

    with report.status("Set up model..."):
        MODEL = model_from_config(
            graph_params=params["graph"],
            model_params=params["model"],
        )
        ndim = len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric
        report.success(
            f"Set up {type(MODEL)} model with {ndim} parameters"
        )

    args.output.parent.mkdir(exist_ok=True)
    num_prevalences = len(params["prevalences"])
    with h5py.File(args.output, mode="w") as prevalences_storage:
        for i,scenario in enumerate(params["prevalences"]):
            prevalences = predicted_prevalence(
                model=MODEL,
                samples=samples[::args.thin],
                description=f"Compute prevalences for scenario {i+1}/{num_prevalences}...",
                midline_ext_prob=get_midline_ext_prob(DATA, scenario["t_stage"]),
                **scenario
            )
            prevalences_dset = prevalences_storage.create_dataset(
                name=scenario["name"],
                data=prevalences,
            )
            num_match, num_total = observed_prevalence(
                data=DATA,
                lnls=get_lnls(MODEL),
                **scenario,
            )
            for key,val in scenario.items():
                try:
                    prevalences_dset.attrs[key] = val
                except TypeError:
                    pass

            prevalences_dset.attrs["num_match"] = num_match
            prevalences_dset.attrs["num_total"] = num_total

        report.success(
            f"Computed prevalences of {num_prevalences} scenarios stored at "
            f"{args.output}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
