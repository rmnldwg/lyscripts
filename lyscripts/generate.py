"""
Generate synthetic patient data for testing and validation purposes.
"""
import argparse
from pathlib import Path

import emcee
import numpy as np

from lyscripts.utils import cli_load_yaml_params, model_from_config, report


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
        "num", type=int,
        help="Number of synthetic patient records to generate",
    )
    parser.add_argument(
        "output", type=Path,
        help="Path where to store the generated synthetic data",
    )

    parser.add_argument(
        "--params", default="./params.yaml", type=Path,
        help="Parameter file containing model specifications"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--set-theta", nargs="+", type=float,
        help="Set the spread probs and parameters for time marginalization by hand"
    )
    group.add_argument(
        "--load-theta", choices=["mean", "max_llh"], default="mean",
        help="Use either the mean or the maximum likelihood estimate from drawn samples"
    )

    parser.add_argument(
        "--samples", default="./models/samples.hdf5", type=Path,
        help="Path to the samples if a method to load them was chosen"
    )

    parser.set_defaults(run_main=main)


def main(args: argparse.Namespace):
    """
    The CLI's help for this subcommand (`lyscripts generate --help`) shows:

    ```
    usage: lyscripts generate [-h] [--params PARAMS]
                            [--set-theta SET_THETA [SET_THETA ...] | --load-theta
                            {mean,max_llh}] [--samples SAMPLES]
                            num output

    Generate synthetic patient data for testing purposes.


    POSITIONAL ARGUMENTS
    num                                   Number of synthetic patient records to
                                            generate
    output                                Path where to store the generated synthetic
                                            data

    OPTIONAL ARGUMENTS
    -h, --help                            show this help message and exit
    --params PARAMS                       Parameter file containing model specifications
                                            (default: ./params.yaml)
    --set-theta SET_THETA [SET_THETA      Set the spread probs and parameters for time
    ...]                                  marginalization by hand (default: None)
    --load-theta {mean,max_llh}           Use either the mean or the maximum likelihood
                                            estimate from drawn samples (default: mean)
    --samples SAMPLES                     Path to the samples if a method to load them
                                            was chosen (default: ./models/samples.hdf5)
    ```
    """
    params = cli_load_yaml_params(args.params)

    with report.status("Create model..."):
        model = model_from_config(
            graph_params=params["graph"],
            model_params=params["model"],
            modalities_params=params["modalities"],
        )
        ndim = len(model.spread_probs) + model.diag_time_dists.num_parametric
        report.success(f"Created {type(model)} model")

    if args.set_theta is not None:
        with report.status("Assign given parameters to model..."):
            if len(args.set_theta) != ndim:
                raise ValueError(
                    f"Model takes {ndim} parameters, but{len(args.set_theta)} were provided"
                )
            THETA = np.array(args.set_theta)
            model.check_and_assign(THETA)
            report.print(THETA)
            report.success("Assigned given parameters to model")
    else:
        with report.status(f"Load samples and choose their {args.load_theta} value..."):
            backend = emcee.backends.HDFBackend(
                args.samples,
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

            model.check_and_assign(THETA)
            report.print(THETA)
            report.success(f"Loaded samples and assigned their {args.load_theta} value")

    with report.status(f"Generate synthetic data of {args.num} patients..."):
        synthetic_data = model.generate_dataset(
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
        args.output.parent.mkdir(exist_ok=True)
        synthetic_data.to_csv(args.output, index=None)
        report.success(f"Saved generated dataset to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
