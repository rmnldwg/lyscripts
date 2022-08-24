"""
Generate a corner plot of the drawn samples.

A corner plot is a combination of 1D and 2D marginals of probability distributions.
The library I use for this is built on `matplotlib` and is called
[`corner`](https://github.com/dfm/corner.py).
"""
import argparse
from pathlib import Path

import corner
import emcee
import lymph
import yaml

from ..helpers import clean_docstring, graph_from_config, report


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
        "model", type=Path,
        help="Path to model output files (HDF5)."
    )
    parser.add_argument(
        "output", type=Path,
        help="Path to output corner plot (SVG)."
    )
    parser.add_argument(
        "-p", "--params", default="./params.yaml", type=Path,
        help="Path to parameter file"
    )

    parser.set_defaults(run_main=main)


def main(args: argparse.Namespace):
    """
    This (sub)subrogram shows the following help message when asking for it
    via `python -m lyscripts plot corner --help`

    ```
    usage: lyscripts plot corner [-h] [-p PARAMS] model output

    Generate a corner plot of the drawn samples.


    POSITIONAL ARGUMENTS
    model                Path to model output files (HDF5).
    output               Path to output corner plot (SVG).

    OPTIONAL ARGUMENTS
    -h, --help           show this help message and exit
    -p, --params PARAMS  Path to parameter file (default: ./params.yaml)
    ```
    """
    with report.status("Read in parameters..."):
        with open(args.params, mode='r') as params_file:
            params = yaml.safe_load(params_file)
        report.success(f"Read in params from {args.params}")

    with report.status("Open model as emcee backend..."):
        backend = emcee.backends.HDFBackend(args.model, read_only=True)
        report.success(f"Opened model as emcee backend from {args.model}")

    with report.status("Plot corner plot..."):
        graph = graph_from_config(params["graph"])
        model_cls = getattr(lymph, params["model"]["class"])
        model = model_cls(graph=graph, **params["model"]["kwargs"])

        if isinstance(model, lymph.Unilateral):
            base_labels = [f"{e.start}➜{e.end}" for e in model.base_edges]
            trans_labels = [f"{e.start}➜{e.end}" for e in model.trans_edges]
            binom_labels = [f"p of {t}" for t in params["model"]["t_stages"][1:]]
            labels = [*base_labels, *trans_labels, *binom_labels]

        elif isinstance(model, lymph.Bilateral):
            base_ipsi_labels = [f"i {e.start}➜{e.end}" for e in model.ipsi.base_edges]
            base_contra_labels = [f"c {e.start}➜{e.end}" for e in model.contra.base_edges]
            trans_labels = [f"{e.start}➜{e.end}" for e in model.ipsi.trans_edges]
            binom_labels = [f"p of {t}" for t in params["model"]["t_stages"][1:]]
            labels = [
                *base_ipsi_labels,
                *base_contra_labels,
                *trans_labels,
                *binom_labels
            ]

        elif isinstance(model, lymph.MidlineBilateral):
            base_ipsi = [f"i {e.start}➜{e.end}" for e in model.ext.ipsi.base_edges]
            base_contra_ext = [f"ce {e.start}➜{e.end}" for e in model.ext.contra.base_edges]
            base_contra_noext = [f"cn {e.start}➜{e.end}" for e in model.noext.contra.base_edges]
            trans = [f"{e.start}➜{e.end}" for e in model.ext.ipsi.trans_edges]
            binom = [f"p of {t}" for t in params["model"]["t_stages"][1:]]
            if model.use_mixing:
                labels = [
                    *base_ipsi,
                    *base_contra_noext,
                    "mixing $\\alpha$",
                    *trans,
                    *binom
                ]
            else:
                labels = [
                    *base_ipsi,
                    *base_contra_ext,
                    *base_contra_noext,
                    *trans,
                    *binom
                ]

        chain = backend.get_chain(flat=True)
        if len(labels) != chain.shape[1]:
            raise RuntimeError(f"length labels: {len(labels)}, shape chain: {chain.shape}")
        fig = corner.corner(
            chain,
            labels=labels,
            show_titles=True,
        )
        # make sure directory exists
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output.with_suffix(".svg"))
        fig.savefig(args.output.with_suffix(".png"), dpi=300)
        report.success(f"Saved corner plot to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
