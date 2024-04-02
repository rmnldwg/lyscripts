"""
A `streamlit`_ app for computing, displaying, and reproducing prevalence estimates.

The primary goal with this little GUI is that one can quickly draft some data &
prediction comparisons visually and then copy & paste the configuration in YAML format
that is necessary to reproduce this via the :py:mod:`.predict.prevalences` script.

.. _streamlit: https://streamlit.io/
"""
import argparse
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lymph import models, types

from lyscripts.compute.prevalences import (  # generate_predicted_prevalences,
    compute_observed_prevalence,
)
from lyscripts.compute.utils import complete_pattern, reduce_pattern
from lyscripts.plot.utils import COLOR_CYCLE, BetaPosterior, Histogram, draw
from lyscripts.utils import (
    create_model,
    load_model_samples,
    load_patient_data,
    load_yaml_params,
)


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """Add a parser to ``subparsers`` and call :py:func:`_add_arguments` with it."""
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """Add arguments needed to run this `streamlit`_ app.

    .. _streamlit: https://streamlit.io/
    """
    parser.add_argument(
        "--message", type=str,
        help="Print our this little message."
    )

    parser.set_defaults(run_main=launch_streamlit)


def launch_streamlit(*_args, discard_args_idx: int = 3, **_kwargs):
    """Start the `streamlit`_ app with the given arguments.

    It will discard all entries in the ``sys.argv`` that come before the
    ``discard_args_idx``, because this also usually contains e.g. the name of the
    current file that might not be relevant to the streamlit app.

    .. _streamlit: https://streamlit.io/
    """
    try:
        from streamlit.web.cli import main as st_main
    except ImportError as mnf_err:
        raise ImportError(
            "Install lyscripts with the `apps` option to install the necessary "
            "requirements for running the streamlit apps."
        ) from mnf_err

    sys.argv = ["streamlit", "run", __file__, "--", *sys.argv[discard_args_idx:]]
    st_main()


def _get_lnl_pattern_label(selected: bool | None = None) -> str:
    """Return labels for the involvement options of an LNL."""
    if selected is None:
        return "Unknown"
    elif selected:
        return "Involved"
    elif not selected:
        return "Healthy"
    else:
        raise ValueError("Selected option can only be `True`, `False` or `None`.")


def _get_midline_ext_label(selected: bool | None = None) -> str:
    """Return labels for the options of the midline extension."""
    if selected is None:
        return "Unknown"
    elif selected:
        return "Extension"
    elif not selected:
        return "Lateralized"
    else:
        raise ValueError("Selected option can only be `True`, `False` or `None`.")


def interactive_load(streamlit):
    """Load YAML parameters, CSV patient data, and HDF5 samples interactively."""
    params_file = streamlit.file_uploader(
        label="YAML params file",
        type=["yaml", "yml"],
        help="Parameter YAML file containing configurations w.r.t. the model etc.",
    )
    params = load_yaml_params(params_file)
    model = create_model(params)
    is_unilateral = isinstance(model, models.Unilateral)

    streamlit.write("---")

    data_file = streamlit.file_uploader(
        label="CSV file of patient data",
        type=["csv"],
        help="CSV spreadsheet containing lymphatic patterns of progression",
    )
    header_rows = [0,1] if is_unilateral else [0,1,2]
    patient_data = load_patient_data(data_file, header=header_rows)

    streamlit.write("---")

    samples_file = streamlit.file_uploader(
        label="HDF5 sample file",
        type=["hdf5", "hdf", "h5"],
        help="HDF5 file containing the samples."
    )
    samples = load_model_samples(samples_file)

    return model, patient_data, samples


def interactive_pattern(
    streamlit,
    is_unilateral: bool,
    lnls: list[str],
    side: str
) -> dict[str, bool]:
    """Create a `streamlit`_ panel for specifying an involvement pattern.

    Fill this panel with radio buttons for each of the ``lnls`` of the given ``side``.
    """
    streamlit.subheader(f"{side}lateral")
    side_pattern = {}

    if side == "contra" and is_unilateral:
        return side_pattern

    for lnl in lnls:
        side_pattern[lnl] = streamlit.radio(
            label=f"LNL {lnl}",
            options=[False, None, True],
            index=1,
            key=f"{side}_{lnl}",
            format_func=_get_lnl_pattern_label,
            horizontal=True,
        )

    return side_pattern


def interactive_additional_params(
    streamlit: ModuleType,
    model: types.Model,
    data: pd.DataFrame,
    samples: np.ndarray,
) -> dict[str, Any]:
    """Add other selectors to the `streamlit`_ panel.

    Allows the user to select T-category, midline extension and whether to invert the
    computed prevalence (meaning computing $1 - p$, when $p$ is the prevalence).

    The respective controls are presented next to each other in three dedicated columns.
    """
    control_cols = streamlit.columns([1,2,1,1,1])
    t_stage = control_cols[0].selectbox(
        label="T-category",
        options=model.diag_time_dists.keys(),
    )
    modalities_in_data = data.columns.get_level_values(level=0).difference(
        ["patient", "tumor", "positive_dissected", "total_dissected", "info"]
    )
    selected_modality = control_cols[1].selectbox(
        label="Modality",
        options=modalities_in_data,
        index=5,
    )
    midline_ext = control_cols[2].radio(
        label="Midline Extension",
        options=[False, None, True],
        index=0,
        format_func=_get_midline_ext_label,
    )

    invert = control_cols[3].radio(
        label="Invert?",
        options=[False, True],
        index=0,
        format_func=lambda x: "Yes" if x else "No",
    )

    thin = control_cols[4].slider(
        label="Sample thinning",
        min_value=1,
        max_value=len(samples) // 100,
        value=100,
    )

    return {
        "t_stage": t_stage,
        "modality": selected_modality,
        "midline_ext": midline_ext,
        "invert": invert,
        "thin": thin,
    }


def reset(session_state: dict[str, Any]):
    """Reset `streamlit`_ session state.

    .. _streamlit: https://streamlit.io/
    """
    for key in session_state.keys():
        del session_state[key]


def add_current_scenario(
    session_state: dict[str, Any],
    pattern: dict[str, dict[str, bool]],
    model: types.Model,
    samples: np.ndarray,
    data: pd.DataFrame,
    prevs_kwargs: dict[str, Any] | None = None,
) -> list[Histogram | BetaPosterior]:
    """Compute prevalence of ``pattern`` as seem in ``data`` and predicted by ``model``.

    This uses a set of ``samples``. The results are then stored in the ``contents``
    list ready to be plotted. The ``prevs_kwargs`` are directly passed on to
    the functions :py:func:`.predict.prevalences.compute_observed_prevalence`
    and :py:func:`.predict.prevalences.generate_predicted_prevalences`.
    """
    num_success, num_total = compute_observed_prevalence(
        diagnosis=pattern,
        data=data,
        lnls=len(model.get_params()),
        **prevs_kwargs,
    )

    prevs_gen = generate_predicted_prevalences(
        pattern=pattern,
        model=model,
        samples=samples,
        **prevs_kwargs,
    )
    computed_prevs = np.zeros(shape=len(samples))
    for i, prevalence in enumerate(prevs_gen):
        computed_prevs[i] = prevalence

    next_color = next(COLOR_CYCLE)
    beta_posterior = BetaPosterior(num_success, num_total, kwargs={"color": next_color})
    histogram = Histogram(computed_prevs, kwargs={"color": next_color})

    session_state["contents"].append(beta_posterior)
    session_state["contents"].append(histogram)

    session_state["scenarios"].append({
        "pattern": reduce_pattern(pattern), **prevs_kwargs
    })


def main(args: argparse.Namespace):
    """The main function that contains the `streamlit`_ code and functionality.

    .. _streamlit: https://streamlit.io/"""
    import streamlit as st

    st.title("Prevalence")

    with st.sidebar:
        model, patient_data, samples = interactive_load(st)

    st.write("---")

    contra_col, ipsi_col = st.columns(2)
    container = {"ipsi": ipsi_col, "contra": contra_col}

    lnls = len(model.get_params())
    is_unilateral = isinstance(model, models.Unilateral)

    pattern = {}
    for side in ["ipsi", "contra"]:
        with container[side]:
            pattern[side] = interactive_pattern(st, is_unilateral, lnls, side)

    pattern = complete_pattern(pattern, lnls)
    st.write("---")

    prevs_kwargs = interactive_additional_params(st, model, patient_data, samples)
    thin = prevs_kwargs.pop("thin")

    st.write("---")

    if "contents" not in st.session_state:
        st.session_state["contents"] = []

    if "scenarios" not in st.session_state:
        st.session_state["scenarios"] = []

    button_cols = st.columns(6)
    button_cols[0].button(
        label="Reset plot",
        on_click=reset,
        args=(st.session_state,),
        type="secondary",
    )
    button_cols[1].button(
        label="Add figures",
        on_click=add_current_scenario,
        kwargs={
            "session_state": st.session_state,
            "pattern": pattern,
            "model": model,
            "samples": samples[::thin],
            "data": patient_data,
            "prevs_kwargs": prevs_kwargs,
        },
        type="primary",
    )

    fig, ax = plt.subplots()
    draw(axes=ax, contents=st.session_state.get("contents", []), xlims=(0., 100.))
    ax.legend()
    st.pyplot(fig)

    st.write("---")

    for scenario in st.session_state["scenarios"]:
        st.code(yaml.dump(scenario))


if __name__ == "__main__":
    if "__streamlit__" in locals():
        parser = argparse.ArgumentParser(description=__doc__)
        _add_arguments(parser)

        args = parser.parse_args()
        main(args)

    else:
        launch_streamlit(discard_args_idx=1)
