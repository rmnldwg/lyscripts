import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lymph import models

from lyscripts.plot.utils import COLORS, save_figure
from lyscripts.utils import create_mixture, load_patient_data, load_yaml_params

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
        "--input", type=Path, help="File path of resampled optimal parameters"
    )
    parser.add_argument(
        "--optimal_params",
        type=Path,
        help="File path of the initial optimal parameters",
    )
    parser.add_argument("--output", type=Path, help="Output path for the plot")
    parser.add_argument(
        "-p",
        "--params",
        default="./params.yaml",
        type=Path,
        help="Path to parameter file.",
    )

    parser.set_defaults(run_main=main)


def get_stats(keys, best_params_list, optima=None, percentile_range=0.68):
    means = []
    lower_errors = []
    upper_errors = []
    for index, key in enumerate(keys):
        values = np.array([bp[key] for bp in best_params_list])
        values = sorted(values)
        n_values = len(values)
        n_within_range = int(np.ceil(percentile_range * n_values))
        if optima is not None:
            mean = optima[index]
        else:
            mean = np.mean(values)
        distances = [abs(v - mean) for v in values]
        sorted_indices = np.argsort(distances)
        closest_indices = sorted_indices[:n_within_range]
        # Extract the 68% range
        percentile_values = [values[i] for i in closest_indices]
        percentile_values.sort()
        low_err = max(mean - min(percentile_values), 0)
        high_err = max(max(percentile_values) - mean, 0)
        means.append(mean)
        lower_errors.append(low_err)
        upper_errors.append(high_err)
    return means, [lower_errors, upper_errors]


def plot_mixture_comp_uncertainty(initial_params_path, resampling_path):
    all_keys = MIXTURE.get_params().keys()
    initial_params = pd.read_csv(initial_params_path, index_col=0)["value"].to_dict()
    # List all files in the optimal_params_path directory
    optimal_params_files = os.listdir(resampling_path)

    # Open each file and load as a dictionary
    best_params_list = []
    for fname in optimal_params_files:
        fpath = os.path.join(resampling_path, fname)
        params = pd.read_csv(fpath, index_col=0)["value"].to_dict()
        best_params_list.append(params)

    num_components = len(MIXTURE.components)
    mixture_keys = [key for key in all_keys if "coef" in key]
    # Dynamically generate keys for each component based on num_components
    keys_per_component = [
        [key for key in mixture_keys if f"{i}_C" in key] for i in range(num_components)
    ]

    # Get the optimal parameters as means
    MIXTURE.set_params(**initial_params)
    optima_list = []
    mixture_coefs = MIXTURE.get_mixture_coefs()
    for i in range(num_components):
        optima_list.append(mixture_coefs.loc[i].to_list())

    # Compute stats for each component
    means_list = []
    yerr_list = []
    for keys, optima in zip(keys_per_component, optima_list, strict=False):
        means, yerr = get_stats(keys, best_params_list, optima)
        means_list.append(means)
        yerr_list.append(yerr)

    color_list = []
    for lister in optima_list:
        extreme_subsite = list(MIXTURE.subgroups.keys())[np.argmax(lister)]
        if extreme_subsite in ["C02", "C03", "C04", "C05", "C06"]:
            color_list.append(COLORS["blue"])
        elif extreme_subsite in ["C01", "C09", "C10"]:
            color_list.append(COLORS["green"])
        elif extreme_subsite in ["C12", "C13"]:
            color_list.append(COLORS["red"])
        elif extreme_subsite in ["C32.0", "C32.1", "C32.2"]:
            color_list.append(COLORS["orange"])
        # Unpack for plotting for a dynamic number of components
    plt.figure(figsize=(10, 6))
    x = np.arange(len(keys_per_component[0]))
    for idx, (means, yerr, keys, color) in enumerate(
        zip(means_list, yerr_list, keys_per_component, color_list, strict=False)
    ):
        plt.errorbar(
            x,
            means,
            yerr=yerr,
            fmt="o",
            color=color,
            ecolor=color,
            capsize=5,
            markersize=8,
            label=f"component {idx}",
        )

    plt.xticks(x, MIXTURE.subgroups.keys(), rotation=0, ha="right", fontsize=10)
    plt.legend(fontsize=10)
    plt.ylabel("Values", fontsize=12)
    plt.title("mixture Values Assignment", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    return plt


def main(args: argparse.Namespace):
    params = load_yaml_params(args.params)
    global MIXTURE
    MIXTURE = create_mixture(params)
    inference_data = load_patient_data(params["general"]["data"])

    mapping = params["model"].get("mapping", None)
    if isinstance(MIXTURE.components[0], models.Unilateral):
        MIXTURE.load_patient_data(
            inference_data,
            split_by=params["model"].get("split_by", ("tumor", "1", "subsite")),
            mapping=mapping,
        )

    plot = plot_mixture_comp_uncertainty(args.optimal_params, args.input)
    save_figure(args.output, plot, formats=["png", "svg"])
    logger.info(f"Mixture component uncertainty plot saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
