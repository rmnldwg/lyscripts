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
from lymph.utils import fast_binomial_pmf

from .helpers import get_graph_from_, report

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


def make_complete_(pat_or_diag: Dict[str, Dict[str, bool]]) -> Dict[str, Dict[str, bool]]:
    """
    Make sure both sides in the pattern or diagnose are specified.
    """
    if not any([side in pat_or_diag for side in ["ipsi", "contra"]]):
        raise ValueError("pattern/diagnose must contain at least one side.")

    if "ipsi" in pat_or_diag:
        lnls = [lnl for lnl in pat_or_diag["ipsi"].keys()]
    else:
        lnls = [lnl for lnl in pat_or_diag["contra"].keys()]

    for side in ["ipsi", "contra"]:
        if side not in pat_or_diag:
            pat_or_diag[side] = {lnl: None for lnl in lnls}

    return pat_or_diag


def compute_prevalence(
    data: pd.DataFrame,
    t_stage: str,
    pattern: Dict[str, Dict[str, bool]],
    comp_modality: str = "max_llh",
    midline_ext: Optional[bool] = None,
    invert: bool = False,
    **_kwargs,
):
    """
    Compute the prevalence of a `pattern` of interest in the data. Do this by calling
    the appropriate function, depending on whether the data contains uni- or bilateral
    information.
    """
    pattern = make_complete_(pattern)

    if data.columns.nlevels == 3:
        return _compute_prevalence_bilateral(
            data=data,
            t_stage=t_stage,
            midline_ext=midline_ext,
            pattern=pattern,
            comp_modality=comp_modality,
            invert=invert,
        )
    elif data.columns.nlevels == 2:
        return _compute_prevalence_unilateral(
            data=data,
            t_stage=t_stage,
            pattern=pattern["ipsi"],
            comp_modality=comp_modality,
            invert=invert,
        )
    else:
        raise ValueError("Data must contain either 2 or 3 levels.")

def _compute_prevalence_unilateral(
    data: pd.DataFrame,
    t_stage: str,
    pattern: Dict[str, bool],
    comp_modality: str = "max_llh",
    invert: bool = False,
):
    """
    Compute the prevalence of a given involvement `pattern` in the data for the
    unilateral case.
    """
    is_t_stage = data["info", "t_stage"] == t_stage
    prev_data = data.loc[is_t_stage, comp_modality]

    lnl_match = False if invert else True
    for lnl, state in pattern.items():
        if state is None:
            continue
        if invert:
            lnl_match |= prev_data[lnl] != state
        else:
            lnl_match &= prev_data[lnl] == state

    match_data = prev_data.loc[lnl_match]
    prevalence = len(match_data) / len(prev_data)
    return prevalence

def _compute_prevalence_bilateral(
    data: pd.DataFrame,
    t_stage: str,
    pattern: Dict[str, Dict[str, bool]],
    comp_modality: str = "max_llh",
    midline_ext: Optional[bool] = None,
    invert: bool = False,
) -> float:
    """
    Compute the prevalence of a given involvement `pattern` in the data for the
    bilateral case (with or without `midline_ext`).
    """
    is_t_stage = data["info", "tumor", "t_stage"] == t_stage

    if midline_ext is not None:
        has_midline_ext = data["info", "tumor", "midline_extension"] == midline_ext
        prev_data = data.loc[is_t_stage & has_midline_ext, comp_modality]
    else:
        prev_data = data.loc[is_t_stage, comp_modality]

    lnl_match = False if invert else True
    for side in ["ipsi", "contra"]:
        for lnl, state in pattern[side].items():
            if state is None:
                continue
            if invert:
                lnl_match |= prev_data[side, lnl] != state
            else:
                lnl_match &= prev_data[side, lnl] == state

    match_data = prev_data.loc[lnl_match]
    prevalence = len(match_data) / len(prev_data)
    return prevalence


def split_samples(
    samples: np.ndarray,
    selected_t_stage: str,
    all_t_stages: List[str],
    first_binom_prob: float = 0.3,
):
    """
    Split the samples into spread probs and parameters for the binomial distribution
    over diagnose times.
    """
    ndim = samples.shape[1]
    num_spread_probs = ndim - len(all_t_stages) + 1
    spread_probs = samples[:, :num_spread_probs]

    t_stage_idx = all_t_stages.index(selected_t_stage)
    if t_stage_idx == 0:
        binom_probs = np.ones(shape=samples.shape[0]) * first_binom_prob
    else:
        binom_probs = samples[:, num_spread_probs + t_stage_idx - 1]

    return spread_probs, binom_probs


def compute_risk(
    model: Union[lymph.Unilateral, lymph.Bilateral, lymph.MidlineBilateral],
    spread_probs: np.ndarray,
    binom_probs: np.ndarray,
    pattern: Dict[str, Dict[str, bool]],
    diagnosis: Optional[Dict[str, Dict[str, bool]]] = None,
    diagnosis_spsn: Optional[List[float]] = None,
    midline_ext: Optional[bool] = None,
    max_t: int = 10,
    invert: Optional[bool] = None,
    **_kwargs,
):
    """
    Compute the risk of a `pattern` of involvement given a `diagnose`. Do this by
    calling the appropriate function for the respective `model` class.
    """
    if isinstance(model, lymph.Unilateral):
        lnl_names = [lnl.name for lnl in model.lnls]
    elif isinstance(model, lymph.Bilateral):
        lnl_names = [lnl.name for lnl in model.ipsi.lnls]
    elif isinstance(model, lymph.MidlineBilateral):
        lnl_names = [lnl.name for lnl in model.ext.ipsi.lnls]
    else:
        raise ValueError(f"Unknown model type {type(model)}.")

    # Make sure all LNLs are specified for the pattern, as well as the diagnosis.
    if diagnosis is None:
        diagnosis = {
            "ipsi": {lnl: None for lnl in lnl_names},
            "contra": {lnl: None for lnl in lnl_names},
        }

    pattern = make_complete_(pattern)
    diagnosis = make_complete_(diagnosis)

    # Adapt the pattern and diagnosis to the lymph-model API.
    # This is necessary, because the risk function's implementation in
    # `lymph` isn't quite ideal yet
    lymph_api_pattern = {"ipsi": [], "contra": []}
    lymph_api_diagnosis = {"ipsi": {"risk": []}, "contra": {"risk": []}}
    for side in ["ipsi", "contra"]:
        for lnl in lnl_names:
            lymph_api_pattern[side].append(pattern[side][lnl])
            lymph_api_diagnosis[side]["risk"].append(diagnosis[side][lnl])

    # Apply the specificity and sensitivity values of the provided diagnosis
    if diagnosis_spsn is None:
        diagnosis_spsn = [1.0, 1.0]
    model.modalities = {"risk": diagnosis_spsn}

    if isinstance(model, lymph.Unilateral):
        return _compute_risk_unilateral(
            model=model,
            spread_probs=spread_probs,
            binom_probs=binom_probs,
            pattern=lymph_api_pattern["ipsi"],
            diagnosis=lymph_api_diagnosis["ipsi"],
            max_t=max_t,
            invert=invert,
        )
    elif isinstance(model, lymph.Bilateral) or isinstance(model, lymph.MidlineBilateral):
        return _compute_risk_bilateral(
            model=model,
            spread_probs=spread_probs,
            binom_probs=binom_probs,
            pattern=lymph_api_pattern,
            diagnosis=lymph_api_diagnosis,
            midline_ext=midline_ext,
            max_t=max_t,
            invert=invert,
        )

def _compute_risk_unilateral(
    model: lymph.Unilateral,
    spread_probs: np.ndarray,
    binom_probs: np.ndarray,
    pattern: Dict[str, bool],
    diagnosis: Dict[str, bool],
    max_t: int = 10,
    invert: bool = False,
):
    """
    Compute the risk of a given `pattern` for a set of parameter `samples` and
    an observed `diagnosis` in the unilateral case.
    """
    time = np.arange(max_t+1)
    risk = np.zeros(shape=len(spread_probs), dtype=float)

    for i, (sp, bp) in enumerate(zip(spread_probs, binom_probs)):
        time_dist = fast_binomial_pmf(time, max_t, bp)
        model.spread_probs = sp
        risk[i] = model.risk(
            inv=pattern,
            diagnoses=diagnosis,
            time_dist=time_dist,
        )

    return 1. - risk if invert else risk

def _compute_risk_bilateral(
    model: Union[lymph.Bilateral, lymph.MidlineBilateral],
    spread_probs: np.ndarray,
    binom_probs: np.ndarray,
    midline_ext: bool,
    pattern: Dict[str, Dict[str, bool]],
    diagnosis: Dict[str, Dict[str, bool]],
    max_t: int = 10,
    invert: bool = False,
) -> np.ndarray:
    """
    Compute the risk of a given `pattern` for a set of parameter `samples` and
    an observed `diagnosis` in the bilateral case.
    """
    time = np.arange(max_t+1)
    risk = np.zeros(shape=len(spread_probs), dtype=float)

    # Correctly call the function, depending on whether the midline extension is used
    if isinstance(model, lymph.Bilateral):
        def risk_func(inv, diagnoses, time_dist, **_kwargs):
            return model.risk(
                inv=inv,
                diagnoses=diagnoses,
                time_dist=time_dist,
            )
    elif isinstance(model, lymph.MidlineBilateral):
        def risk_func(inv, diagnoses, time_dist, midline_ext, **_kwargs):
            return model.risk(
                inv=inv,
                diagnoses=diagnoses,
                time_dist=time_dist,
                midline_extension=midline_ext,
            )
    else:
        raise TypeError(
            f"Model type must be `Bilateral` or `MidlineBilateral`, not {type(model)}."
        )

    for i, (sp, bp) in enumerate(zip(spread_probs, binom_probs)):
        time_dist = fast_binomial_pmf(time, max_t, bp)
        model.spread_probs = sp
        risk[i] = risk_func(
            inv=pattern,
            diagnoses=diagnosis,
            time_dist=time_dist,
            midline_ext=midline_ext,
        )

    return 1. - risk if invert else risk


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
        with open(params_path, 'r') as params_file:
            params = yaml.safe_load(params_file)
        report.success(f"Read in params from {params_path}")

    if args.data is not None:
        with report.status("Reading in data..."):
            data_path = Path(args.data)
            header = [0,1] if params["model"]["class"] == "Unilateral" else [0,1,2]
            data = pd.read_csv(data_path, header=header)
            report.success(f"Read in data from {data_path}")

    with report.status("Loading samples..."):
        model_path = Path(args.model)
        reader = emcee.backends.HDFBackend(model_path, read_only=True)
        burnin = reader.iteration - params["sampling"]["keep_steps"]
        walkers_per_dim = params["sampling"]["walkers_per_dim"]
        samples = reader.get_chain(flat=True, thin=walkers_per_dim, discard=burnin)
        report.success(f"Loaded samples with shape {samples.shape} from {model_path}")

    with report.status("Set up model..."):
        model_cls = getattr(lymph, params["model"]["class"])
        graph = get_graph_from_(params["model"]["graph"])
        model = model_cls(graph=graph)
        report.success(f"Set up model with length of spread probs {len(model.spread_probs)}")

    plt.style.use(args.mplstyle)

    num_risk_plots = len(params["risk_plots"])
    with report.status(f"Computing & drawing 0/{num_risk_plots} risks...") as s:
        plot_path = Path(args.output)
        plot_path.mkdir(exist_ok=True)
        plot_path = plot_path

        for k, risk_plot in enumerate(params["risk_plots"]):
            s.update(f"Computing & drawing {k+1}/{num_risk_plots} risks...")
            fig, ax = plt.subplots(figsize=set_size(width="full"))
            fig.suptitle(risk_plot["title"])
            hist_cyc = (
                cycler(histtype=["stepfilled", "step"])
                * cycler(color=USZ_COLOR_LIST)
            )
            vline_cyc = (
                cycler(linestyle=["-", "--"])
                * cycler(color=USZ_COLOR_LIST)
            )

            risks = np.zeros(
                shape=(len(risk_plot["scenarios"]), len(samples)),
                dtype=float
            )
            prevalences = np.zeros(shape=len(risk_plot["scenarios"]))
            for i, scenario in enumerate(risk_plot["scenarios"]):
                if args.data is not None and "comp_modality" in scenario:
                    prevalences[i] = 100. * compute_prevalence(
                        data=data, **scenario
                    )
                spread_probs, binom_probs = split_samples(
                    samples,
                    selected_t_stage=scenario["t_stage"],
                    all_t_stages=params["model"]["t_stages"],
                    first_binom_prob=params["model"]["first_binom_prob"],
                )
                risks[i] = 100. * compute_risk(
                    model=model,
                    spread_probs=spread_probs,
                    binom_probs=binom_probs,
                    **scenario,
                    **params["model"],
                )

            bins = np.linspace(
                start=np.minimum(np.min(risks), np.min(prevalences)),
                stop=np.maximum(np.max(risks), np.max(prevalences)),
                num=risk_plot["num_bins"],
            )
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
