"""
Join datasets from different sources (but of the same format) into one.
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd

from .helpers import report

warnings.simplefilter(action="ignore", category=FutureWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--inputs", nargs='+',
        help="List of paths to inference-ready CSV datasets to concatente."
    )
    parser.add_argument(
        "-o", "--output", default="data/joined.csv",
        help="Location to store the concatenated CSV file."
    )

    # Parse arguments and prepare paths
    args = parser.parse_args()
    input_paths = [Path(p) for p in args.inputs]


    with report.status("Reading & concatenating CSV files..."):
        concatenated_df = pd.DataFrame()
        for input_path in input_paths:
            df = pd.read_csv(input_path, header=[0,1,2])
            concatenated_df = pd.concat(
                [concatenated_df, df],
                ignore_index=True
            )
            report.print(f"+ concatenated data from {input_path}")
        report.success(f"Read & concatenated all {len(args.inputs)} CSV files")

    with report.status("Saving concatenated dataset..."):
        # Make sure the output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Write the concatenated dataset to disk
        concatenated_df.to_csv(output_path, index=None)
        report.success(f"Saved concatenated dataset to {output_path}")
