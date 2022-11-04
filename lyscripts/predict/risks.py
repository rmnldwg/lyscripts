"""
Predict risks of involvements using the samples that were drawn during the inference
process and scenarios as defined in a YAML file. The structure of these scenarios can
be seen in an actual `params.yaml` file over in the
[lynference](https://github.com/rmnldwg/lynference) repository.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import emcee
import h5py
import lymph
import numpy as np
import yaml

from lyscripts.predict._utils import clean_pattern, rich_enumerate
from lyscripts.utils import get_lnls, model_from_config, report


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
        "model", type=Path,
        help="Path to drawn samples (HDF5)"
    )
    parser.add_argument(
        "output", default="./models/risks.hdf5", type=Path,
        help="Output path for predicted risks (HDF5 file)"
    )
    parser.add_argument(
        "--thin", default=1, type=int,
        help="Take only every n-th sample"
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
    lnls = get_lnls(model)
    involvement = clean_pattern(involvement, lnls)
    given_diagnosis = clean_pattern(given_diagnosis, lnls)

    if given_diagnosis_spsn is not None:
        model.modalities = {"risk": given_diagnosis_spsn}
    else:
        model.modalities = {"risk": [1., 1.]}

    risks = np.zeros(shape=len(samples), dtype=float)

    if isinstance(model, lymph.Unilateral):
        given_diagnosis = {"risk": given_diagnosis["ipsi"]}

        for i,sample in rich_enumerate(samples, description):
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

    for i,sample in rich_enumerate(samples, description):
        risks[i] = model.risk(
            involvement=involvement,
            given_params=sample,
            given_diagnoses=given_diagnosis,
            t_stage=t_stage,
            midline_extension=midline_ext,
        )
    return 1. - risks if invert else risks


def main(args: argparse.Namespace):
    """
    The call signature to this script looks is shown below and can be generated by
    typng `lyscripts predict risks --help`.

    ```
    USAGE: lyscripts predict risks [-h] [--thin THIN] [--params PARAMS] model output

    Predict risks of involvements using the samples that were drawn during the
    inference process and scenarios as defined in a YAML file. The structure of these
    scenarios can be seen in an actual `params.yaml` file over in the
    (https://github.com/rmnldwg/lynference) repository.

    POSITIONAL ARGUMENTS:
    model            Path to drawn samples (HDF5)
    output           Output path for predicted risks (HDF5 file)

    OPTIONAL ARGUMENTS:
    -h, --help       show this help message and exit
    --thin THIN      Take only every n-th sample (default: 1)
    --params PARAMS  Path to parameter file (default: ./params.yaml)
    ```
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
                samples=SAMPLES[::args.thin],
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
