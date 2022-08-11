"""
Generate synthetic patient data for testing purposes.
"""
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params", default="params.yaml",
        help="Parameter file containing model specifications (YAML)"
    )
    parser.add_argument(
        "--num", type=int,
        help="Number of synthetic patient records to generate",
    )
    parser.add_argument(
        "--output", default="data/synthetic.csv",
        help="Path where to store the generated synthetic data",
    )
