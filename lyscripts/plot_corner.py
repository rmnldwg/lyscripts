"""
Generate a corner plot of the drawn samples.
"""
import argparse
from pathlib import Path

import corner
import emcee
import lymph
import yaml

from .helpers import graph_from_config, report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m", "--model", required=True,
        help="Path to model output files (HDF5)."
    )
    parser.add_argument(
        "-p", "--params", default="params.yaml",
        help="Path to parameter file (YAML)."
    )
    parser.add_argument(
        "--plots", default="plots/corner",
        help="Path to output corner plot (SVG)."
    )

    args = parser.parse_args()

    with report.status("Read in parameters..."):
        with open(args.params, mode='r') as params_file:
            params = yaml.safe_load(params_file)
        report.success(f"Read in params from {args.params}")

    with report.status("Open model as emcee backend..."):
        model_path = Path(args.model)
        backend = emcee.backends.HDFBackend(model_path, read_only=True)
        report.success(f"Opened model as emcee backend from {model_path}")

    with report.status("Plot corner plot..."):
        plot_path = Path(args.plots)
        plot_path.mkdir(parents=True, exist_ok=True)

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
        fig.savefig(plot_path / "corner.svg")
        fig.savefig(plot_path / "corner.png", dpi=300)
        report.success(f"Saved corner plot to {plot_path}")
