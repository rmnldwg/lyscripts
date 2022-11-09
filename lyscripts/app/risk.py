"""
A `streamlit` app for computing, displaying and reproducing risk estimates.
"""
import argparse
import sys
from pathlib import Path


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

    st.title(args.message)


if __name__ == "__main__":
    if "__streamlit__" in locals():
        parser = argparse.ArgumentParser(description=__doc__)
        _add_arguments(parser)

        args = parser.parse_args()
        main(args)

    else:
        launch_streamlit(discard_args_idx=1)
