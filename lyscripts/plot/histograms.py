"""
Plot computed risks and prevalences into a beautiful histogram.
"""
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from cycler import cycler

from ..helpers import clean_docstring, report
from . import COLORS


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
        "input", type=Path,
        help="File path of the computed risks or prevalences (HDF5)"
    )
    parser.add_argument(
        "plots", type=Path,
        help="Output directory for the plot"
    )

    parser.add_argument(
        "--names", nargs="+",
        help="List of names of computed risks/prevalences to combine into one plot"
    )
    parser.add_argument(
        "--title",
        help="Title of the plot"
    )
    parser.add_argument(
        "--bins", default=50, type=int,
        help="Number of bins to put the computed values into"
    )
    parser.add_argument(
        "--mplstyle", default="./.mplstyle", type=Path,
        help="Path to the MPL stylesheet"
    )

    parser.set_defaults(run_main=main)


def get_size(width="single", unit="cm", ratio="golden"):
    """Get optimal figure size for a range of scenarios."""
    if width == "single":
        width = 10
    elif width == "full":
        width = 16

    ratio = 1.618 if ratio == "golden" else ratio
    width = width / 2.54 if unit == "cm" else width
    height = width / ratio
    return (width, height)


def get_label(attrs) -> str:
    """Extract label of a historgam from the HDF5 attrs object of the dataset."""
    label = []
    transforms = {
        "label": lambda x: x,
        "modality": lambda x: x,
        "t_stage": lambda x: x,
        "midline_ext": lambda x: "ext" if x else "noext"
    }
    for key,func in transforms.items():
        if key in attrs:
            label.append(func(attrs[key]))
    return " | ".join(label)


def main(args: argparse.Namespace):
    """
    Run main program with `args` parsed by argparse.
    """
    with report.status("Read in computed values..."):
        with h5py.File(name=args.input, mode="r") as h5_file:
            values = []
            labels = []
            num_matches = []
            num_totals = []
            lines = []
            min_value = 1.
            max_value = 0.

            for name in args.names:
                try:
                    dataset = h5_file[name]
                except KeyError as key_err:
                    raise KeyError(
                        f"No precomputed values found for name {name}"
                    ) from key_err

                values.append(100. * dataset[:])
                labels.append(get_label(dataset.attrs))
                num_matches.append(dataset.attrs.get("num_match", np.nan))
                num_totals.append(dataset.attrs.get("num_total", np.nan))
                lines.append(100. * num_matches[-1] / num_totals[-1])
                min_value = np.minimum(min_value, np.min(values))
                max_value = np.maximum(max_value, np.max(values))

            min_value = np.min(lines, where=~np.isnan(lines), initial=min_value)
            max_value = np.max(lines, where=~np.isnan(lines), initial=max_value)

        report.success(f"Read in computed values from {args.input}")

    with report.status("Apply MPL stylesheet..."):
        plt.style.use(args.mplstyle)
        report.success(f"Applied MPL stylesheet from {args.mplstyle}")

    with report.status("Set up figure..."):
        fig, ax = plt.subplots(figsize=get_size())
        fig.suptitle(args.title)
        hist_cycl = (
            cycler(histtype=["stepfilled", "step"])
            * cycler(color=list(COLORS.values()))
        )
        line_cycl = (
            cycler(linestyle=["-", "--"])
            * cycler(color=list(COLORS.values()))
        )
        report.success("Set up figure")
        hist_kwargs = {
            "bins": np.linspace(min_value, max_value, args.bins),
            "density": True,
            "alpha": 0.6,
            "linewidth": 2.,
        }

    with report.status("Plot histograms..."):
        x = np.linspace(min_value, max_value, 200)
        zipper = zip(values, labels, num_matches, num_totals, hist_cycl, line_cycl)
        for vals, label, a, n, hstyle, lstyle in zipper:
            ax.hist(
                vals,
                label=label,
                **hist_kwargs,
                **hstyle
            )
            if not np.isnan(a):
                post = sp.stats.beta.pdf(x / 100., a+1, n-a+1) / 100.
                ax.plot(x, post, label=f"{int(a)}/{int(n)}", **lstyle)
            ax.legend()
            ax.set_xlabel("probability [%]")
        report.success(f"Plotted {len(values)} histograms")

    with report.status("Save plots..."):
        args.plots.mkdir(exist_ok=True)
        plt.savefig(args.plots / f"{args.title}.png", dpi=200)
        plt.savefig(args.plots / f"{args.title}.svg")
        report.success(f"Stored plots at {args.plots}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
