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

def compute_prevalence(
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
    if data.columns.nlevels == 3:
        is_bilateral = True
        is_t_stage = data["info", "tumor", "t_stage"] == t_stage
        if midline_ext is not None:
            has_midline_ext = data["info", "tumor", "midline_extension"] == midline_ext
        else:
            has_midline_ext = True
    elif data.columns.nlevels == 2:
        is_bilateral = False
        is_t_stage = data["info", "t_stage"] == t_stage
        has_midline_ext = True
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

def compute_risk(
    model: Union[lymph.Unilateral, lymph.Bilateral, lymph.MidlineBilateral],
    pattern: Dict[str, Dict[str, bool]],
    given_params: np.ndarray,
    given_diagnose: Optional[Dict[str, Dict[str, bool]]] = None,
    given_diagnose_spsn: Optional[List[float]] = None,
    prediction_spsn: Optional[List[float]] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Compute the probability of seeing a `pattern` of involvement using a `model`
    with pretrained `given_params`. This pribability can be computed for a
    `given_diagnose` that was obtained using a modality with specificity & sensitivity
    specified by `given_diagnose_spsn`.

    With the `prediction_spsn` one can set the specificity & sensitivity for the
    prediction. E.g., if both are set to `1.0`, this will output the risk for
    occult disease, while if it is anything lower, it will return the probability
    of seeing a diagnose that matches the `pattern`.

    Set `verbose` to `True` for a visualization of the progress.
    """


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
                    prevalences.append(100. * compute_prevalence(
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
