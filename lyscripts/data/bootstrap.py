"""
Learn the spread probabilities of the HMM for lymphatic tumor progression using
the preprocessed data as input and the mixture model.
"""
# pylint: disable=logging-fstring-interpolation
import argparse
import logging
import os
import numpy as np
from sklearn.utils import resample
from pathlib import Path
import pandas as pd
from lyscripts.utils import (
    load_patient_data,
    load_yaml_params,
)

logger = logging.getLogger(__name__)


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """Add a parser to the ``subparsers`` action."""
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
        help="Path to a LyProX-style CSV file"
    )
    parser.add_argument(
        "--output", type=Path,
        help="Folder destination to save LyProX-style CSV files"
    )
    parser.add_argument(
        "-p", "--params", default="params.yaml", type=Path,
        help="Path to parameter file"
    )

    parser.set_defaults(run_main=main)

def proportional_bootstrap(df, group_col, n_bootstraps, folder_path = None):
    """Produce n_bootstraps bootstrapped datasets from the original DataFrame.
    this keeps the number of patients per subsite constant.

    Args:
        df (pd.DataFrame): The original DataFrame to bootstrap from.
        group_col (str): The name of the column to group by.
        n_bootstraps (int): The number of bootstrapped datasets to create.

    Returns:
        list[pd.DataFrame]: A list of bootstrapped DataFrames.
    """
    datasets = []
    group_sizes = df[group_col].value_counts(normalize=True)
    total_n = len(df)
    os.makedirs(folder_path, exist_ok=True)
    for _ in range(n_bootstraps):
        samples = []
        for group, proportion in group_sizes.items():
            n_samples = int(np.round(proportion * total_n))
            group_df = df[df[group_col] == group]
            boot_group = resample(group_df, replace=True, n_samples=n_samples)
            samples.append(boot_group)
        boot_df = pd.concat(samples).sample(frac=1).reset_index(drop=True)  # optional shuffle
        datasets.append(boot_df)
    if folder_path is not None:
        for i, dataset in enumerate(datasets):
            file_path = os.path.join(folder_path, f'dataset_resample_{i}.csv')
            dataset.to_csv(file_path, index=False)
    else:
        return datasets

def main(args: argparse.Namespace) -> None:
    """Main function to sample parameters for a mixture model"""
    input_table = load_patient_data(args.input)
    params = load_yaml_params(args.params)
    if ('tumor', 'core', 'subsite') in input_table.columns:
        group_col = ('tumor', 'core', 'subsite')
    elif ('tumor', '1', 'subsite') in input_table.columns:
        group_col = ('tumor', '1', 'subsite')
    else:
        logger.error("No 'subsite' column found in the input data.")
    proportional_bootstrap(input_table, group_col=group_col, n_bootstraps=params["sampling"]["n_bootstraps"], folder_path=args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    args = parser.parse_args()
    args.run_main(args)
