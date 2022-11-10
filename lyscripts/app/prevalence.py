"""
A `streamlit` app for computing, displaying and reproducing prevalence estimates.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import lymph

from lyscripts.app.utils import st_chain_from_hdf5, st_load_yaml_params
from lyscripts.predict.utils import clean_pattern
from lyscripts.utils import get_lnls, model_from_config


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
    Add arguments needed to run this `streamlit` app.
    """
    parser.add_argument(
        "--message", type=str,
        help="Print our this little message."
    )

    parser.set_defaults(run_main=launch_streamlit)


def get_option_label(selected: Optional[bool] = None) -> str:
    """Return labels for the involvement options of an LNL."""
    if selected is None:
        return "Unknown"
    elif selected:
        return "Involved"
    elif not selected:
        return "Healthy"
    else:
        raise ValueError("Selected option can only be `True`, `False` or `None`.")


def launch_streamlit(*_args, discard_args_idx: int = 3, **_kwargs):
    """
    Regardless of the entry point into this script, this function will start
    `streamlit` and pass on the provided command line arguments.
    """
    try:
        from streamlit.web.cli import main as st_main
    except ModuleNotFoundError as mnf_err:
        raise ModuleNotFoundError(
            "Install lyscripts with the `apps` option to install the necessary "
            "requirements for running the streamlit apps."
        ) from mnf_err

    sys.argv = ["streamlit", "run", __file__, "--", *sys.argv[discard_args_idx:]]
    st_main()


def main(args: argparse.Namespace):
    """
    The main function that contains the `streamlit` code and main functionality.
    """
    import streamlit as st

    st.title("Prevalence")

    with st.sidebar:
        params_file = st.file_uploader(
            label="YAML params file",
            type=["yaml", "yml"],
            help="Parameter YAML file containing configurations w.r.t. the model etc.",
        )
        params = st_load_yaml_params(params_file)

        model = model_from_config(
            graph_params=params["graph"],
            model_params=params["model"],
        )
        lnls = get_lnls(model)

        st.write("---")

        samples_file = st.file_uploader(
            label="HDF5 sample file",
            type=["hdf5", "hdf", "h5"],
            help="HDF5 file containing the samples."
        )
        samples = st_chain_from_hdf5(samples_file)

    contra_col, ipsi_col = st.columns(2)
    container = {"ipsi": ipsi_col, "contra": contra_col}
    pattern = {"ipsi": {}, "contra": {}}

    for side in ["ipsi", "contra"]:
        with container[side]:
            st.subheader(f"{side}lateral")

            if side == "contra" and isinstance(model, lymph.Unilateral):
                continue

            for lnl in lnls:
                pattern[side][lnl] = st.radio(
                    label=f"LNL {lnl}",
                    options=[False, None, True],
                    index=1,
                    key=f"{side}_{lnl}",
                    format_func=get_option_label,
                    horizontal=True,
                )

    pattern = clean_pattern(pattern, lnls)
    st.write(pattern)



if __name__ == "__main__":
    if "__streamlit__" in locals():
        parser = argparse.ArgumentParser(description=__doc__)
        _add_arguments(parser)

        args = parser.parse_args()
        main(args)

    else:
        launch_streamlit(discard_args_idx=1)
