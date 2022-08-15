"""
Compute and plot risks for a sampled model.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import emcee
import lymph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from cycler import cycler
from rich.progress import track

from .helpers import model_from_config, report

# define colors
USZ_BLUE = '#005ea8'
USZ_GREEN = '#00afa5'
USZ_RED = '#ae0060'
USZ_ORANGE = '#f17900'
USZ_GRAY = '#c5d5db'
USZ_COLOR_LIST = [USZ_BLUE, USZ_ORANGE, USZ_GREEN, USZ_RED, USZ_GRAY]
HATCH_LIST = ["////", r"\\\\", "||||", "----", "oooo"]


def set_size(width="single", unit="cm", ratio="golden"):
    """
    Get optimal figure size for a range of scenarios.
    """
    if width == "single":
        width = 10
    elif width == "full":
        width = 16

    ratio = 1.618 if ratio == "golden" else ratio
    width = width / 2.54 if unit == "cm" else width
    height = width / ratio
    return (width, height)

def prepare_figure(title: str):
    """Return figure and axes to plot risk histograms into."""
    fig, ax = plt.subplots(figsize=set_size(width="full"))
    fig.suptitle(risk_plot["title"])
    histogram_cycler = (
        cycler(histtype=["stepfilled", "step"])
        * cycler(color=USZ_COLOR_LIST)
    )
    vline_cycler = (
        cycler(linestyle=["-", "--"])
        * cycler(color=USZ_COLOR_LIST)
    )
    return fig, ax, histogram_cycler, vline_cycler


def get_match_idx(
    pattern: Dict[str, Optional[bool]],
    data: pd.DataFrame,
    lnls: Optional[List[str]] = None,
    invert: bool = False,
) -> pd.Series:
    """Get the indices of the rows in the `data` where the diagnose matches the
    `pattern` of interest for every lymph node level in the `lnls`.
    """
    if lnls is None:
        lnls = list(pattern.keys())

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
    lnls: Optional[List[str]] = None,
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
    if "ipsi" not in pattern:
        pattern["ipsi"] = {}
    if "contra" not in pattern:
        pattern["contra"] = {}

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
    verbose: bool = True,
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
    if verbose:
        enumerate_samples = track(
            enumerate_samples,
            description="Computing predicted prevalence...",
            console=report,
        )

    prevalences = np.zeros(shape=len(samples), dtype=float)

    # ensure the `pattern` is complete
    lnls = model.lnls
    for side in ["ipsi", "contra"]:
        if side not in pattern:
            pattern[side] = {}
        pattern[side] = {lnl: pattern[side].get(lnl, None) for lnl in lnls}

    if isinstance(model, lymph.Unilateral):
        # make DataFrame with one row from `pattern`
        mi = pd.MultiIndex.from_product([["prev"], lnls])
        pattern_df = pd.DataFrame(columns=mi)
        pattern_df["prev"] = pd.DataFrame(pattern["ipsi"], index=[0])
        pattern["info", "t_stage"] = t_stage

    elif isinstance(model, (lymph.Bilateral, lymph.MidlineBilateral)):
        # make DataFrame with one row from `pattern`
        mi = pd.MultiIndex.from_product([["prev"], ["ipsi", "contra"], lnls])
        pattern_df = pd.DataFrame(columns=mi)
        for side in ["ipsi", "contra"]:
            pattern_df["prev"][side] = pd.DataFrame(pattern[side], index=[0])
        pattern["info", "tumor", "t_stage"] = t_stage

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


def compute_risk(
    involvement: Dict[str, Dict[str, bool]],
    model: Union[lymph.Unilateral, lymph.Bilateral, lymph.MidlineBilateral],
    samples: np.ndarray,
    t_stage: str,
    midline_ext: bool = False,
    given_diagnosis: Optional[Dict[str, Dict[str, bool]]] = None,
    given_diagnosis_spsn: Optional[List[float]] = None,
    invert: bool = False,
    verbose: bool = True,
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
    if verbose:
        enumerate_samples = track(
            enumerate_samples,
            description="Computing risk...",
            console=report,
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

    given_diagnose = {"risk": given_diagnose}

    for i,sample in enumerate_samples:
        risks[i] = model.risk(
            involvement=involvement,
            given_params=sample,
            given_diagnoses=given_diagnose,
            t_stage=t_stage,
        )
    return 1. - risks if invert else risks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m", "--model", required=True,
        help="Path to drawn samples (HDF5)"
    )
    parser.add_argument(
        "-d", "--data", default=None,
        help="Path to the data file if risk is to be compared to prevalence"
    )
    parser.add_argument(
        "-p", "--params", default="params.yaml",
        help="Path to parameter file (YAML)"
    )
    parser.add_argument(
        "-o", "--output", default="plots",
        help="Output directory for results (plots)"
    )
    parser.add_argument(
        "--mplstyle", default=".mplstyle",
        help="Matplotlib style file"
    )
    args = parser.parse_args()

    plt.style.use(args.mplstyle)

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
        walkers_per_dim = params["sampling"]["walkers_per_dim"]
        all_samples = reader.get_chain(flat=True)
        num = len(all_samples)
        idx = np.random.choice(num, size=(num // walkers_per_dim))
        samples = all_samples[idx]
        report.success(f"Loaded samples with shape {samples.shape} from {model_path}")

    with report.status("Set up model & load data..."):
        MODEL = model_from_config(
            graph_params=params["graph"],
            model_params=params["model"],
            modalities_params=params["modalities"],
        )
        MODEL.patient_data = DATA
        ndim = len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric
        nwalkers = ndim * params["sampling"]["walkers_per_dim"]
        report.success(
            f"Set up {type(MODEL)} model with {ndim} parameters and loaded "
            f"{len(DATA)} patients"
        )

    plt.style.use(args.mplstyle)

    num_risk_plots = len(params["risk_plots"])
    with report.status(f"Computing & drawing 0/{num_risk_plots} risks...") as s:
        plot_path = Path(args.output)
        plot_path.mkdir(exist_ok=True)
        plot_path = plot_path

        # loop through the individual risk plots to compute
        for k, risk_plot in enumerate(params["risk_plots"]):
            s.update(f"Computing & drawing {k+1}/{num_risk_plots} risks...")
            fig, ax, hist_cyc, vline_cyc = prepare_figure(risk_plot["title"])

            risks, prevalences = [], []
            for i, scenario in enumerate(risk_plot["scenarios"]):
                if args.data is not None and "comp_modality" in scenario:
                    prevalences.append(100. * observed_prevalence(
                        data=DATA, **scenario
                    ))
                MODEL.modalities = {"risk": scenario["given_diagnosis_spsn"]}
                try:
                    given_diagnosis = {"risk": scenario["given_diagnosis"]}
                except KeyError:
                    given_diagnosis = None
                risks.append(100. * np.array([
                    MODEL.risk(
                        involvement=scenario["pattern"],
                        given_params=sample,
                        given_diagnoses=given_diagnosis,
                        t_stage=scenario["t_stage"],
                        midline_extension=scenario["midline_ext"],
                    ) for sample in samples
                ]))
                if scenario["invert"]:
                    risks[-1] = 100. - risks[-1]

            bins = np.linspace(
                start=np.minimum(np.min(risks), np.min(prevalences)),
                stop=np.maximum(np.max(risks), np.max(prevalences)),
                num=risk_plot["num_bins"],
            )
            # cycle through the scenarios to compare in one figure
            for i, tmp in enumerate(zip(risk_plot["scenarios"], hist_cyc, vline_cyc)):
                scenario, hist_style, vline_style = tmp
                ax.hist(
                    risks[i],
                    bins=bins,
                    density=True,
                    label=scenario["label"],
                    alpha=0.7,
                    linewidth=2,
                    **hist_style
                )
                if args.data is not None and "comp_modality" in scenario:
                    ax.axvline(
                        prevalences[i],
                        label=f"prevalence {scenario['label']}",
                        **vline_style
                    )

            ax.set_ylabel("$p(R)$")
            ax.set_xlabel("Risk $R$ [%]")
            ax.legend()
            plot_name = risk_plot["title"].lower().replace(" ", "_")
            fig.savefig(plot_path / f"{plot_name}.png", dpi=300)
            fig.savefig(plot_path / f"{plot_name}.svg")

        report.success(f"Computed & drawn {num_risk_plots} risks to {plot_path}")
