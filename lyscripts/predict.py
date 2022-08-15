"""
Predict risks of involvements and prevalences of diagnostic patterns using the model
used for inference and the drawn samples.
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

from .helpers import get_lnls, model_from_config, nested_to_pandas, report

# def set_size(width="single", unit="cm", ratio="golden"):
#     """
#     Get optimal figure size for a range of scenarios.
#     """
#     if width == "single":
#         width = 10
#     elif width == "full":
#         width = 16

#     ratio = 1.618 if ratio == "golden" else ratio
#     width = width / 2.54 if unit == "cm" else width
#     height = width / ratio
#     return (width, height)

# def prepare_figure(title: str):
#     """Return figure and axes to plot risk histograms into."""
#     fig, ax = plt.subplots(figsize=set_size(width="full"))
#     fig.suptitle(risk_plot["title"])
#     histogram_cycler = (
#         cycler(histtype=["stepfilled", "step"])
#         * cycler(color=USZ_COLOR_LIST)
#     )
#     vline_cycler = (
#         cycler(linestyle=["-", "--"])
#         * cycler(color=USZ_COLOR_LIST)
#     )
#     return fig, ax, histogram_cycler, vline_cycler


def get_match_idx(
    pattern: Dict[str, Optional[bool]],
    data: pd.DataFrame,
    lnls: List[str],
    invert: bool = False,
) -> pd.Series:
    """Get the indices of the rows in the `data` where the diagnose matches the
    `pattern` of interest for every lymph node level in the `lnls`.
    """
    match_idx = False if invert else True
    for lnl in lnls:
        if lnl not in pattern or pattern[lnl] is None:
            continue
        if invert:
            match_idx |= data[lnl] != pattern[lnl]
        else:
            match_idx &= data[lnl] == pattern[lnl]

    return match_idx

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
    if is_bilateral:
        do_lnls_match = get_match_idx(
            pattern["ipsi"],
            eligible_data["ipsi"],
            lnls=lnls,
            invert=invert
        )
        do_lnls_match &= get_match_idx(
            pattern["contra"],
            eligible_data["contra"],
            lnls=lnls,
            invert=invert
        )
    else:
        do_lnls_match = get_match_idx(
            pattern["ipsi"],
            eligible_data,
            lnls=lnls,
            invert=invert,
        )
    matching_data = eligible_data[do_lnls_match]
    return len(matching_data) / len(eligible_data)

def predicted_prevalence(
    pattern: Dict[str, Dict[str, bool]],
    model: Union[lymph.Unilateral, lymph.Bilateral, lymph.MidlineBilateral],
    samples: np.ndarray,
    t_stage: str,
    midline_ext: bool = False,
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

        if isinstance(model, lymph.MidlineBilateral):
            if midline_ext:
                model = model.ext
            else:
                model = model.noext

    model.patient_data = pattern_df

    # compute prevalence as likelihood of diagnose `prev`, which was defined above
    for i,sample in enumerate_samples:
        prevalences[i] = model.likelihood(
            given_params=sample,
            log=False,
        )
    return 1. - prevalences if invert else prevalences


def predicted_risk(
    involvement: Dict[str, Dict[str, bool]],
    model: Union[lymph.Unilateral, lymph.Bilateral, lymph.MidlineBilateral],
    samples: np.ndarray,
    t_stage: str,
    midline_ext: bool = False,
    given_diagnosis: Optional[Dict[str, Dict[str, bool]]] = None,
    given_diagnosis_spsn: Optional[List[float]] = None,
    invert: bool = False,
    description: Optional[str] = None,
    **_kwargs,
) -> np.ndarray:
    """Compute the probability of arriving in a particular `involvement` in a given
    `t_stage` using a `model` with pretrained `samples`. This probability can be
    computed for a `given_diagnosis` that was obtained using a modality with
    specificity & sensitivity provided via `given_diagnosis_spsn`. If the model is an
    instance of `lymph.MidlineBilateral`, one can specify whether or not the primary
    tumor has a `midline_ext`.

    Both the `involvement` and the `given_diagnosis` should be dictionaries like this:

    ```python
    involvement = {
        "ipsi":  {"I": False, "II": True , "III": None , "IV": None},
        "contra: {"I": None , "II": False, "III": False, "IV": None},
    }
    ```

    The returned probability can be `invert`ed.

    Set `verbose` to `True` for a visualization of the progress.
    """
    model.modalities = {"risk": given_diagnosis_spsn}

    # wrap the iteration over samples in a rich progressbar if `verbose`
    enumerate_samples = enumerate(samples)
    if description is not None:
        enumerate_samples = track(
            enumerate_samples,
            description=description,
            total=len(samples),
            console=report,
            transient=True,
        )

    risks = np.zeros(shape=len(samples), dtype=float)

    if isinstance(model, lymph.Unilateral):
        given_diagnosis = {"risk": given_diagnosis["ipsi"]}

        for i,sample in enumerate_samples:
            risks[i] = model.risk(
                involvement=involvement["ipsi"],
                given_params=sample,
                given_diagnoses=given_diagnosis,
                t_stage=t_stage
            )
        return 1. - risks if invert else risks

    elif isinstance(model, lymph.MidlineBilateral):
        if midline_ext:
            model = model.ext
        else:
            model = model.noext

    elif not isinstance(model, lymph.Bilateral):
        raise TypeError("Model is not a known type.")

    given_diagnosis = {"risk": given_diagnosis}

    for i,sample in enumerate_samples:
        risks[i] = model.risk(
            involvement=involvement,
            given_params=sample,
            given_diagnoses=given_diagnosis,
            t_stage=t_stage,
        )
    return 1. - risks if invert else risks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True,
        help="Path to drawn samples (HDF5)"
    )
    parser.add_argument(
        "--data", default=None,
        help="Path to the data file if risk is to be compared to prevalence"
    )
    parser.add_argument(
        "--params", default="params.yaml",
        help="Path to parameter file (YAML)"
    )
    parser.add_argument(
        "--output-risks", default="models/risks.hdf5",
        help="Output path for predicted risks (HDF5 file)"
    )
    parser.add_argument(
        "--output-prevalences", default="models/prevalences.hdf5",
        help="Output path for predicted prevalences (HDF5 file)"
    )
    args = parser.parse_args()

    with report.status("Read in parameters..."):
        params_path = Path(args.params)
        with open(params_path, mode='r') as params_file:
            params = yaml.safe_load(params_file)
        report.success(f"Read in params from {params_path}")

    if args.data is not None:
        with report.status("Read in training data..."):
            data_path = Path(args.data)
            # Only read in two header rows when using the Unilateral model
            is_unilateral = params["model"]["class"] == "Unilateral"
            header = [0, 1] if is_unilateral else [0, 1, 2]
            DATA = pd.read_csv(data_path, header=header)
            report.success(f"Read in training data from {data_path}")

    with report.status("Loading samples..."):
        model_path = Path(args.model)
        reader = emcee.backends.HDFBackend(model_path, read_only=True)
        SAMPLES = reader.get_chain(flat=True)
        report.success(f"Loaded samples with shape {SAMPLES.shape} from {model_path}")

    with report.status("Set up model..."):
        MODEL = model_from_config(
            graph_params=params["graph"],
            model_params=params["model"],
        )
        ndim = len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric
        report.success(
            f"Set up {type(MODEL)} model with {ndim} parameters"
        )

    risks_path = Path(args.output_risks)
    risks_path.parent.mkdir(exist_ok=True)
    num_risks = len(params["risks"])
    with h5py.File(risks_path, mode="w") as risks_storage:
        for i,scenario in enumerate(params["risks"]):
            risks = predicted_risk(
                model=MODEL,
                samples=SAMPLES,
                description=f"Compute risks for scenario {i+1}/{num_risks}...",
                **scenario
            )
            risks_dset = risks_storage.create_dataset(
                name=scenario["name"],
                data=risks,
            )
            for key,val in scenario.items():
                try:
                    risks_dset.attrs[key] = val
                except TypeError:
                    pass
        report.success(f"Computed risks of {num_risks} scenarios stored at {risks_path}")

    prevalences_path = Path(args.output_prevalences)
    prevalences_path.parent.mkdir(exist_ok=True)
    num_prevalences = len(params["prevalences"])
    with h5py.File(prevalences_path, mode="w") as prevalences_storage:
        for i,scenario in enumerate(params["prevalences"]):
            prevalences = predicted_prevalence(
                model=MODEL,
                samples=SAMPLES,
                description=f"Compute prevalences for scenario {i+1}/{num_prevalences}...",
                **scenario
            )
            prevalences_dset = prevalences_storage.create_dataset(
                name=scenario["name"],
                data=prevalences,
            )
            data_prevalence = observed_prevalence(
                data=DATA,
                lnls=get_lnls(MODEL),
                **scenario,
            )
            for key,val in scenario.items():
                try:
                    prevalences_dset.attrs[key] = val
                except TypeError:
                    pass
            prevalences_dset.attrs["observed"] = data_prevalence
        report.success(
            f"Computed prevalences of {num_prevalences} scenarios stored at "
            f"{prevalences_path}"
        )
