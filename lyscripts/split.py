"""
Split the full dataset into cross-validation folds according to the
content of the params.yaml file.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .helpers import report

warnings.simplefilter(action="ignore", category=FutureWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input", default="data/joined.csv",
        help="The path to the full dataset to split."
    )
    parser.add_argument(
        "-p", "--params", default="params.yaml",
        help="Path to parameter YAML file."
    )
    parser.add_argument(
        "-o", "--output", default="data/folds",
        help="Folder to store the split CSV files in."
    )

    # Parse arguments and prepare paths
    args = parser.parse_args()
    input_path = Path(args.input)
    params_path = Path(args.params)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    with report.status("Read in parameters..."):
        with open(params_path, mode='r') as params_file:
            params = yaml.safe_load(params_file)
        report.success(f"Read in params from {params_path}")


    with report.status("Reading in concatenated CSV file..."):
        header = [0,1] if params["model"]["class"] == "Unilateral" else [0,1,2]
        concatenated_df = pd.read_csv(input_path, header=header)
        report.success(f"Read in concatenated CSV file from {input_path}")

    with report.status("Split data into sets for training & "):
        shuffled_df = concatenated_df.sample(
            frac=1.,
            replace=False,
            random_state=params["cross-validation"]["seed"]
        ).reset_index(drop=True)

        split_dfs = np.array_split(
            shuffled_df,
            len(params["cross-validation"]["folds"])
        )
        for fold_id, split_pattern in params["cross-validation"]["folds"].items():
            # Concatenate training and evaluation DataFrames from the split DataFrames
            # using the split pattern defined in the params file.
            eval_df = pd.concat(
                [split_dfs[k] for k,sym in enumerate(split_pattern) if sym == "e"],
                ignore_index=True
            )
            train_df = pd.concat(
                [split_dfs[k] for k,sym in enumerate(split_pattern) if sym == "t"],
                ignore_index=True
            )

            eval_df.to_csv(output_dir / f"{fold_id}_eval.csv", index=None)
            train_df.to_csv(output_dir / f"{fold_id}_train.csv", index=None)
            report.print(f"+ split data into train & eval sets for round {fold_id}")

        report.success(
            "Split data into train & eval sets for all "
            f"{len(params['cross-validation']['folds'])} folds"
        )
