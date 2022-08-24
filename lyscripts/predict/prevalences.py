"""
Predict prevalences of diagnostic patterns using the samples that were inferred using
the model via MCMC sampling and compare them to the prevalence in the data.

This essentially amounts to computing the data likelihood under the model and comparing
it to the empirical likelihood of a given pattern of lymphatic progression.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import emcee
import h5py
import lymph
import numpy as np
import pandas as pd
import yaml
from rich.progress import track

from ..helpers import (
    clean_docstring,
    get_lnls,
    model_from_config,
    nested_to_pandas,
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
        description=clean_docstring(__doc__),
        help=clean_docstring(__doc__),
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
    `pattern` of interest for every lymph node level in the `lnls`.
    """
    for lnl in lnls:
        if lnl not in pattern or pattern[lnl] is None:
            continue
        if invert:
            match_idx |= data[lnl] != pattern[lnl]
        else:
            match_idx &= data[lnl] == pattern[lnl]

    return match_idx

def get_midline_ext_prob(data: pd.DataFrame, t_stage: str) -> float:
    """Get the prevalence of midline extension from `data` for `t_stage`."""
    if data.columns.nlevels == 2:
        raise ValueError(
            "Unilateral data contains no information about midline extension"
        )
    is_t_stage = data["info", "tumor", "t_stage"] == t_stage
    eligible_data = data[is_t_stage]
    has_midline_ext = eligible_data["info", "tumor", "midline_extension"] == True
    matching_data = eligible_data[has_midline_ext]
    return len(matching_data) / len(eligible_data)

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
    # make sure the pattern has the right form
    if pattern is None:
        pattern = {}
    for side in ["ipsi", "contra"]:
        if side not in pattern:
            pattern[side] = {}
        pattern[side] = {lnl: pattern[side].get(lnl, None) for lnl in lnls}

    # get the data we care about
    has_midline_ext = True
    if data.columns.nlevels == 3:
        is_bilateral = True
        is_t_stage = data["info", "tumor", "t_stage"] == t_stage
        if midline_ext is not None:
            has_midline_ext = data["info", "tumor", "midline_extension"] == midline_ext
    elif data.columns.nlevels == 2:
        is_bilateral = False
        is_t_stage = data["info", "t_stage"] == t_stage
    else:
        raise ValueError("Data must contain either 2 or 3 levels.")

    eligible_data = data.loc[is_t_stage & has_midline_ext, modality]
    eligible_data = eligible_data.dropna(axis="index", how="all")

    # filter the data by the LNL pattern they report
    do_lnls_match = False if invert else True
    if is_bilateral:
        do_lnls_match = get_match_idx(
            do_lnls_match,
            pattern["ipsi"],
            eligible_data["ipsi"],
            lnls=lnls,
            invert=invert
        )
        do_lnls_match = get_match_idx(
            do_lnls_match,
            pattern["contra"],
            eligible_data["contra"],
            lnls=lnls,
            invert=invert
        )
    else:
        do_lnls_match = get_match_idx(
            do_lnls_match,
            pattern["ipsi"],
            eligible_data,
            lnls=lnls,
            invert=invert,
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
    if modality_spsn is None:
        modality_spsn = [1., 1.]

    model.modalities = {"prev": modality_spsn}

    # wrap the iteration over samples in a rich progressbar if `verbose`
    enumerate_samples = enumerate(samples)
    if description is not None:
        enumerate_samples = track(
            enumerate_samples,
            description=description,
            total=len(samples),
            console=report,
            transient=True
        )

    prevalences = np.zeros(shape=len(samples), dtype=float)

    # ensure the `pattern` is complete
    lnls = get_lnls(model)
    for side in ["ipsi", "contra"]:
        if side not in pattern:
            pattern[side] = {}
        pattern[side] = {lnl: pattern[side].get(lnl, None) for lnl in lnls}

    if isinstance(model, lymph.Unilateral):
        # make DataFrame with one row from `pattern`
        pattern_df = nested_to_pandas({"prev": pattern["ipsi"]})
        pattern_df["info", "t_stage"] = t_stage

    elif isinstance(model, (lymph.Bilateral, lymph.MidlineBilateral)):
        # make DataFrame with one row from `pattern`
        mi = pd.MultiIndex.from_product([["prev"], ["ipsi", "contra"], lnls])
        pattern_df = pd.DataFrame(columns=mi)
        pattern_df["prev"] = nested_to_pandas(pattern)
        pattern_df["info", "tumor", "t_stage"] = t_stage

        if isinstance(model, lymph.MidlineBilateral) and midline_ext is None:
            # if midline_ext is None, provide the MidlineBilateral model with two
            # patients: One with and one without midline extension. Then, marginalize
            # over the two cases using the empirical probability of a midline extension
            # (see below)
            pattern_ext = pattern_df.copy()
            pattern_ext["info", "tumor", "midline_extension"] = True
            pattern_noext = pattern_df.copy()
            pattern_noext["info", "tumor", "midline_extension"] = False
            pattern_df = pd.concat([pattern_ext, pattern_noext], ignore_index=True)
        else:
            pattern_df["info", "tumor", "midline_extension"] = midline_ext

    else:
        raise TypeError(f"{type(model)} is not a supported model")

    model.patient_data = pattern_df

    # compute prevalence as likelihood of diagnose `prev`, which was defined above
    for i,sample in enumerate_samples:
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
    This subprogram's call signature can be obtained via `python -m lyscripts predict
    prevalences --help` and shows this:

    ```
    usage: lyscripts predict prevalences [-h] [--params PARAMS] model data output

    Predict prevalences of diagnostic patterns using the samples that were inferred
    using the model via MCMC sampling and compare them to the prevalence in the data.

    This essentially amounts to computing the data likelihood under the model and
    comparing it to the empirical likelihood of a given pattern of lymphatic
    progression.


    POSITIONAL ARGUMENTS
    model            Path to drawn samples (HDF5)
    data             Path to the data file to compare prediction and data prevalence
    output           Output path for predicted prevalences (HDF5 file)

    OPTIONAL ARGUMENTS
    -h, --help       show this help message and exit
    --params PARAMS  Path to parameter file (default: ./params.yaml)
    ```
    """
    with report.status("Read in parameters..."):
        with open(args.params, mode='r') as params_file:
            params = yaml.safe_load(params_file)
        report.success(f"Read in params from {args.params}")

    with report.status("Read in training data..."):
        # Only read in two header rows when using the Unilateral model
        is_unilateral = params["model"]["class"] == "Unilateral"
        header = [0, 1] if is_unilateral else [0, 1, 2]
        DATA = pd.read_csv(args.data, header=header)
        report.success(f"Read in training data from {args.data}")

    with report.status("Loading samples..."):
        reader = emcee.backends.HDFBackend(args.model, read_only=True)
        SAMPLES = reader.get_chain(flat=True)
        report.success(f"Loaded samples with shape {SAMPLES.shape} from {args.model}")

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
                samples=SAMPLES,
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
            prevalences_dset.attrs["num_match"] = float(num_match)
            prevalences_dset.attrs["num_total"] = float(num_total)
        report.success(
            f"Computed prevalences of {num_prevalences} scenarios stored at "
            f"{args.output}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
