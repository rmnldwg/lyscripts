"""
Transform the enhanced lyDATA CSV files into a format that can be used by the
lymph model using this package's utilities.
"""
import argparse
import warnings
from pathlib import Path
import yaml

import pandas as pd
import lymph

from .helpers import report

warnings.simplefilter(action="ignore", category=FutureWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input",
        help="Path to the enhanced lyDATA CSV file to transform."
    )
    parser.add_argument(
        "-p", "--params", default="params.yaml",
        help="Path to the params file to use for the transformation."
    )
    parser.add_argument(
        "-o", "--output", default="data/cleaned.csv",
        help="Path to the cleand CSV file ready for inference."
    )

    # Parse arguments and prepare paths
    args = parser.parse_args()
    input_path = Path(args.input)
    params_path = Path(args.params)
    output_path = Path(args.output)

    with report.status("Read in parameters..."):
        with open(params_path, mode='r') as params_file:
            params = yaml.safe_load(params_file)
        report.success(f"Read in params from {params_path}")

    with report.status("Reading in CSV file..."):
        enhanced_df = pd.read_csv(input_path, header=[0,1,2])
        method = {
            "Unilateral": "unilateral",
            "Bilateral": "midline",
            "MidlineBilateral": "midline",
        }[params["model"]["class"]]
        cleaned_df = lymph.utils.lyprox_to_lymph(enhanced_df, method=method)

    with report.status("Saving cleaned dataset..."):
        cleaned_df.to_csv(output_path, index=None)
        report.success(f"Saved cleaned dataset to {output_path}")
