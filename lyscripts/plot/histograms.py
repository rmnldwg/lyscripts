"""
Plot computed risks and prevalences into a beautiful histogram.
"""
import argparse
from pathlib import Path
from warnings import warn

import matplotlib.pyplot as plt

from lyscripts.helpers import report
from lyscripts.plot._utils import (
    COLOR_CYCLE,
    Histogram,
    Posterior,
    draw,
    get_size,
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
        "input", type=Path,
        help="File path of the computed risks or prevalences (HDF5)"
    )
    parser.add_argument(
        "output", type=Path,
        help="Output path for the plot"
    )

    parser.add_argument(
        "--names", nargs="+",
        help="List of names of computed risks/prevalences to combine into one plot"
    )
    parser.add_argument(
        "--title", type=str,
        help="Title of the plot"
    )
    parser.add_argument(
        "--bins", default=60, type=int,
        help="Number of bins to put the computed values into"
    )
    parser.add_argument(
        "--mplstyle", default="./.mplstyle", type=Path,
        help="Path to the MPL stylesheet"
    )

    parser.set_defaults(run_main=main)


def main(args: argparse.Namespace):
    """
    The CLI's signature (can be asked for via `lyscripts plot histograms
    --help`) defines the function and its arguments in the following way:

    ```
    USAGE: lyscripts plot histograms [-h] [--names NAMES [NAMES ...]] [--title TITLE]
                                     [--bins BINS] [--mplstyle MPLSTYLE]
                                     input output

    Plot computed risks and prevalences into a beautiful histogram.

    POSITIONAL ARGUMENTS:
        input                 File path of the computed risks or prevalences (HDF5)
        output                Output path for the plot

    OPTIONAL ARGUMENTS:
        -h, --help            show this help message and exit
        --names NAMES [NAMES ...]
                              List of names of computed risks/prevalences to combine
                              into one plot (default: None)
        --title TITLE         Title of the plot (default: None)
        --bins BINS           Number of bins to put the computed values into (default:
                              60)
        --mplstyle MPLSTYLE   Path to the MPL stylesheet (default: ./.mplstyle)
    ```
    """
    try:
        plt.style.use(args.mplstyle)
    except OSError:
        warn(f"Did not find MPL stylesheet at {args.mplstyle}, proceeding without.")
    else:
        report.success(f"Applied MPL stylesheet from {args.mplstyle}")

    with report.status("Add content to figure..."):
        contents = []
        for name in args.names:
            color = next(COLOR_CYCLE)
            contents.append(Histogram(
                filename=args.input,
                dataname=name,
                kwargs={"color": color},
            ))
            report.success(f"Added histogram {name} to figure")
            try:
                contents.append(Posterior(
                    filename=args.input,
                    dataname=name,
                    kwargs={"color": color},
                ))
            except KeyError:
                report.info(f"No observation data available for dataset {name}")
            else:
                report.success(f"Added posterior PDF for data {name} to figure")

    with report.status("Draw figure..."):
        _, ax = plt.subplots(figsize=get_size())
        draw(axes=ax, contents=contents, hist_kwargs={"nbins": args.bins}, percent_lims=(5., 5.))
        ax.legend()
        report.success("Drawn figure")

    with report.status("Save plots..."):
        args.output.parent.mkdir(exist_ok=True)
        plt.savefig(args.output.with_suffix(".png"), dpi=300)
        plt.savefig(args.output.with_suffix(".svg"))
        report.success(f"Stored plots at {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
