"""
Generate synthetic patient data for testing purposes.
"""
import argparse
from pathlib import Path

import emcee
import numpy as np
import yaml

from .helpers import model_from_config, report

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
        "--set-theta", nargs="+",
        help="Set the spread probs and parameters for time marginalization by hand"
    )
    parser.add_argument(
        "--load-theta", choices=["mean", "max_llh"],
        help="Use either the mean or the maximum likelihood estimate from drawn samples"
    )
    parser.add_argument(
        "--samples", default="models/samples.hdf5",
        help="Path to the samples (HDF5 file) if a method to load them was chosen"
    )
    parser.add_argument(
        "--output", default="data/synthetic.csv",
        help="Path where to store the generated synthetic data",
    )

    args = parser.parse_args()

    with report.status("Read in parameters..."):
        params_path = Path(args.params)
        with open(params_path, mode='r') as params_file:
            params = yaml.safe_load(params_file)
        report.success(f"Read in params from {params_path}")

    with report.status("Create model..."):
        MODEL = model_from_config(
            graph_params=params["graph"],
            model_params=params["model"],
            modalities_params=params["modalities"],
        )
        ndim = len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric
        report.success(f"Created {type(MODEL)} model")

    if args.set_theta is not None:
        with report.status("Assign given parameters to model..."):
            if len(args.set_theta) != ndim:
                raise ValueError(
                    f"Model takes {ndim} parameters, but{len(args.set_theta)} were provided"
                )
            THETA = np.array(args.set_theta)
            MODEL.check_and_assign(THETA)
            report.print(THETA)
            report.success("Assigned given parameters to model")
    else:
        with report.status(f"Load samples and choose their {args.load_theta} value..."):
            samples_path = Path(args.samples)
            backend = emcee.backends.HDFBackend(
                samples_path,
                read_only=True,
                name="mcmc"
            )
            chain = backend.get_chain(flat=True)
            log_probs = backend.get_blobs(flat=True)

            if args.load_theta == "mean":
                THETA = np.mean(chain, axis=0)
            elif args.load_theta == "max_llh":
                max_llh_idx = np.argmax(log_probs)
                THETA = chain[max_llh_idx]
            else:
                raise ValueError("Only 'mean' and 'max_llh' are supported")

            MODEL.check_and_assign(THETA)
            report.print(THETA)
            report.success(f"Loaded samples and assigned their {args.load_theta} value")

    with report.status(f"Generate synthetic data of {args.num} patients..."):
        synthetic_data = MODEL.generate_dataset(
            num_patients=args.num,
            stage_dist=params["synthetic"]["t_stages_dist"],
            ext_prob=params["synthetic"]["midline_ext_prob"],
        )
        if len(synthetic_data) != args.num:
            raise RuntimeError(
                f"Length of generated data ({len(synthetic_data)}) does not match "
                f"target length ({args.num})"
            )
        report.success(f"Created synthetic data of {args.num} patients.")

    with report.status("Save generated dataset..."):
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True)
        synthetic_data.to_csv(output_path, index=None)
        report.success(f"Saved generated dataset to {output_path}")
