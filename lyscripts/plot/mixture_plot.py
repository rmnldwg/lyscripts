import argparse
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from lyscripts.plot.utils import COLORS, save_figure, SUBSITE_COLORS

logger = logging.getLogger(__name__)

def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """Add an ``ArgumentParser`` to the subparsers action."""
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """Add arguments to the parser."""
    parser.add_argument(
        "--input", type=Path,
        help="File path of mixture coefficients"
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output path for the plot"
    )

    parser.set_defaults(run_main=main)

def main(args: argparse.Namespace):
    tmp = LinearSegmentedColormap.from_list("tmp", [COLORS['green'], COLORS['red']], N=128)
    mixture_df = pd.read_csv(args.input)
    filtered_keys = [key for key in SUBSITE_COLORS if key in mixture_df.columns]
    mixture_df = mixture_df[filtered_keys]

    # Transpose the matrix to rotate by 90°
    matrix_rotated = mixture_df.T

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 10))

    # Display the rotated matrix using imshow
    cax = ax.imshow(matrix_rotated.values, cmap=tmp, origin='upper')

    # Loop over the data and create text annotations
    for i in range(matrix_rotated.shape[0]):  # Rows (previously columns)
        for j in range(matrix_rotated.shape[1]):  # Columns (previously rows)
            value = matrix_rotated.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", 
                    color="white", fontsize=12)


    # Optional: Set axis labels and title
    ax.set_xticks(range(matrix_rotated.shape[1]))
    ax.set_xticklabels(mixture_df.index, fontsize = 12)  # Original row labels
    ax.set_yticks(range(matrix_rotated.shape[0]))
    ax.set_yticklabels(mixture_df.columns, fontsize = 12)  # Original column labels
    ax.set_title("Mixture Coefficients per subsite", fontsize = 16)
    save_figure(args.output, fig, formats=["png", "svg"])
    logger.info(f"Mixture parameter matrix saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)

