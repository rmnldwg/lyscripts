"""
Predict risks of involvements using the samples that were drawn during the inference
process.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import emcee
import h5py
import lymph
import numpy as np
import yaml
from rich.progress import track

from ..helpers import clean_docstring, model_from_config, report


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """
    Add an `ArgumentParser` to the subparsers action.
    """
    parser = subparsers._add_parser(
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
        help="Path to drawn samples (HDF5)"
    )
    parser.add_argument(
        "output", default="./models/risks.hdf5", type=Path,
        help="Output path for predicted risks (HDF5 file)"
    )
    parser.add_argument(
        "--params", default="./params.yaml", type=Path,
        help="Path to parameter file"
    )

    parser.set_defaults(run_main=main)


def predicted_risk(
    involvement: Dict[str, Dict[str, bool]],
    model: Union[lymph.Unilateral, lymph.Bilateral, lymph.MidlineBilateral],
    samples: np.ndarray,
    t_stage: str,
    midline_ext: bool = False,
    given_diagnosis: Optional[Dict[str, Dict[str, bool]]] = None,
    given_diagnosis_spsn: Optional[List[float]] = None,
    invert: bool = False,
    description: Optional[str] = None,
    **_kwargs,
) -> np.ndarray:
    """Compute the probability of arriving in a particular `involvement` in a given
    `t_stage` using a `model` with pretrained `samples`. This probability can be
    computed for a `given_diagnosis` that was obtained using a modality with
    specificity & sensitivity provided via `given_diagnosis_spsn`. If the model is an
    instance of `lymph.MidlineBilateral`, one can specify whether or not the primary
    tumor has a `midline_ext`.

    Both the `involvement` and the `given_diagnosis` should be dictionaries like this:

    ```python
    involvement = {
        "ipsi":  {"I": False, "II": True , "III": None , "IV": None},
        "contra: {"I": None , "II": False, "III": False, "IV": None},
    }
    ```

    The returned probability can be `invert`ed.

    Set `verbose` to `True` for a visualization of the progress.
    """
    if given_diagnosis is None:
        given_diagnosis = {"ipsi": {}, "contra": {}}

    if given_diagnosis_spsn is not None:
        model.modalities = {"risk": given_diagnosis_spsn}
    else:
        model.modalities = {"risk": [1., 1.]}

    # wrap the iteration over samples in a rich progressbar if `verbose`
    enumerate_samples = enumerate(samples)
    if description is not None:
        enumerate_samples = track(
            enumerate_samples,
            description=description,
            total=len(samples),
            console=report,
            transient=True,
        )

    risks = np.zeros(shape=len(samples), dtype=float)

    if isinstance(model, lymph.Unilateral):
        given_diagnosis = {"risk": given_diagnosis["ipsi"]}

        for i,sample in enumerate_samples:
            risks[i] = model.risk(
                involvement=involvement["ipsi"],
                given_params=sample,
                given_diagnoses=given_diagnosis,
                t_stage=t_stage
            )
        return 1. - risks if invert else risks

    elif not isinstance(model, (lymph.Bilateral, lymph.MidlineBilateral)):
        raise TypeError("Model is not a known type.")

    given_diagnosis = {"risk": given_diagnosis}

    for i,sample in enumerate_samples:
        risks[i] = model.risk(
            involvement=involvement,
            given_params=sample,
            given_diagnoses=given_diagnosis,
            t_stage=t_stage,
            midline_extension=midline_ext,
        )
    return 1. - risks if invert else risks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    args = parser.parse_args()


def main(args: argparse.Namespace):
    """
    Run main program with `args` parsed by argparse.
    """
    with report.status("Read in parameters..."):
        with open(args.params, mode='r') as params_file:
            params = yaml.safe_load(params_file)
        report.success(f"Read in params from {args.params}")

    with report.status("Loading samples..."):
        reader = emcee.backends.HDFBackend(args.model, read_only=True)
        SAMPLES = reader.get_chain(flat=True)
        report.success(f"Loaded samples with shape {SAMPLES.shape} from {args.model}")

    with report.status("Set up model..."):
        MODEL = model_from_config(
            graph_params=params["graph"],
            model_params=params["model"],
        )
        ndim = len(MODEL.spread_probs) + MODEL.diag_time_dists.num_parametric
        report.success(
            f"Set up {type(MODEL)} model with {ndim} parameters"
        )

    args.output.parent.mkdir(exist_ok=True)
    num_risks = len(params["risks"])
    with h5py.File(args.output, mode="w") as risks_storage:
        for i,scenario in enumerate(params["risks"]):
            risks = predicted_risk(
                model=MODEL,
                samples=SAMPLES,
                description=f"Compute risks for scenario {i+1}/{num_risks}...",
                **scenario
            )
            risks_dset = risks_storage.create_dataset(
                name=scenario["name"],
                data=risks,
            )
            for key,val in scenario.items():
                try:
                    risks_dset.attrs[key] = val
                except TypeError:
                    pass
        report.success(
            f"Computed risks of {num_risks} scenarios stored at {args.output}"
        )
