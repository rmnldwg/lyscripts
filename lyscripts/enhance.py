"""
Enhance a LyProX-style CSV dataset in two ways:

1. Add consensus diagnoses based on all available modalities using on of two methods:
`max_llh` infers the most likely true state of involvement given only the available
diagnoses. `rank` uses the available diagnositc modalities and ranks them based on
their respective sensitivity and specificity.

2. Complete sub- & super-level fields. This means that if a dataset reports LNLs IIa
and IIb separately, this script will add the column for LNL II and fill it with the
correct values. Conversely, if e.g. LNL II is reported to be healthy, we can assume
the sublevels IIa and IIb would have been reported as healthy, too.
"""
import argparse
import warnings
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml

from .helpers import clean_docstring, report

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
# pylint: disable=singleton-comparison


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
        "input", type=Path,
        help="Path to a LyProX-style CSV file"
    )
    parser.add_argument(
        "output", type=Path,
        help="Destination for LyProX-style output file including the consensus"
    )
    parser.add_argument(
        "-c", "--consensus", nargs="+", default=["max_llh"],
        choices=CONSENSUS_FUNCS.keys(),
        help="Choose consensus method(s)"
    )
    parser.add_argument(
        "-p", "--params", default="params.yaml", type=Path,
        help="Path to parameter file"
    )
    parser.add_argument(
        "-s", "--sublevel", action="store_true",
        help="Fill information about sub- & super levels where possible"
    )
    parser.add_argument(
        "--sublvls", nargs="+", default=["a", "b"],
        help="Indicate what kinds of sublevels exist"
    )
    parser.add_argument(
        "--lnls-with-sub", nargs="+", default=["I", "II", "V"],
        help="List of LNLs where sublevel reporting has been performed or is common"
    )

    parser.set_defaults(run_main=main)


def get_sublvl_values(
    data_frame: pd.DataFrame,
    lnl: str,
    sub_ids: List[str],
):
    """
    Get the values of sublevels (e.g. 'IIa' and 'IIb') for a given LNL and a
    dataframe.
    """
    has_sublvls = all([lnl+sub in data_frame for sub in sub_ids])
    if not has_sublvls:
        return None
    return data_frame[[lnl+sub for sub in sub_ids]].values

@lru_cache
def has_all_none(obs_tuple: Tuple[np.ndarray]):
    """
    Check if all entries in the observation tuple are ``None``.
    """
    return all(obs is None for obs in obs_tuple)

@lru_cache
def or_consensus(obs_tuple: Tuple[np.ndarray]):
    """
    Compute the consensus of different diagnostic modalities by computing the
    logical OR.
    """
    if has_all_none(obs_tuple):
        return None

    return any(obs_tuple)

@lru_cache
def and_consensus(obs_tuple: Tuple[np.ndarray]):
    """
    Compute the consensus of different diagnostic modalities by computing the
    logical AND.
    """
    if has_all_none(obs_tuple):
        return None

    return not(
        any(not(obs) if obs is not None else None for obs in obs_tuple)
    )

@lru_cache
def maxllh_consensus(
    obs_tuple: Tuple[np.ndarray],
    modalities_spsn: Tuple[List[float]]
):
    """
    Compute the consensus of different diagnostic modalities using their
    respective specificity & sensitivity.

    Args:
        obs_tuple: Tuple with the involvement (``True``, ``False`` or
            ``None``).
        modalities_spsn: Tuple with 2-element lists of the specificity &
            sensitivity of the modalities corresponding to the diagnoses in the
            parameter ``obs_tuple``.

    Returns:
        The most likely true state according to the consensus from the
        diagnoses provided.
    """
    if has_all_none(obs_tuple):
        return None

    healthy_llh = 1.
    involved_llh = 1.
    for obs, spsn in zip(obs_tuple, modalities_spsn):
        if obs is None:
            continue
        spsn = np.array(spsn)
        obs = int(obs)
        spsn2x2 = np.diag(spsn) + np.diag(1. - spsn)[::-1]
        healthy_llh *= spsn2x2[obs,0]
        involved_llh *= spsn2x2[obs,1]

    healthy_vs_involved = np.array([healthy_llh, involved_llh])
    return bool(np.argmax(healthy_vs_involved))

@lru_cache
def rank_consensus(
    obs_tuple: Tuple[np.ndarray],
    modalities_spsn: Tuple[List[float]]
):
    """
    Compute the consensus of different diagnostic modalities using a ranking
    based on sensitivity & specificity.

    Args:
        obs_tuple: Tuple with the involvement (``True``, ``False`` or
            ``None``).
        modalities_spsn: Tuple with 2-element lists of the specificity &
            sensitivity of the modalities corresponding to the diagnoses in the
            parameter ``obs_tuple``.

    Returns:
        The most likely true state based on the ranking.
    """
    if has_all_none(obs_tuple):
        return None

    modalities_spsn = list(modalities_spsn)

    healthy_sens = [
        modalities_spsn[i][1] for i,obs in enumerate(obs_tuple) if obs == False
    ]
    involved_spec = [
        modalities_spsn[i][0] for i,obs in enumerate(obs_tuple) if obs == True
    ]
    if np.max([*healthy_sens, 0.]) > np.max([*involved_spec, 0.]):
        return False

    return True


CONSENSUS_FUNCS = {
    "max_llh": maxllh_consensus,
    "rank": rank_consensus,
    "logic_or": lambda obs, *_args, **_kwargs: and_consensus(obs),
    "logic_and": lambda obs, *_args, **_kwargs: or_consensus(obs),
}


def main(args: argparse.Namespace):
    """
    Below is the help output (call with `python -m lyscripts enhance --help`)

    ```
    usage: lyscripts enhance [-h]
                            [-c {max_llh,rank,logic_or,logic_and}
    [{max_llh,rank,logic_or,logic_and} ...]]
                            [-p PARAMS] [-s] [--sublvls SUBLVLS [SUBLVLS ...]]
                            [--lnls-with-sub LNLS_WITH_SUB [LNLS_WITH_SUB ...]]
                            input output

    Enhance a LyProX-style CSV dataset in two ways:

    1. Add consensus diagnoses based on all available modalities using on of two
    methods: `max_llh` infers the most likely true state of involvement given only the
    available diagnoses. `rank` uses the available diagnositc modalities and ranks them
    based on their respective sensitivity and specificity.

    2. Complete sub- & super-level fields. This means that if a dataset reports LNLs IIa
    and IIb separately, this script will add the column for LNL II and fill it with the
    correct values. Conversely, if e.g. LNL II is reported to be healthy, we can assume
    the sublevels IIa and IIb would have been reported as healthy, too.


    POSITIONAL ARGUMENTS
    input                                 Path to a LyProX-style CSV file
    output                                Destination for LyProX-style output file
                                            including the consensus

    OPTIONAL ARGUMENTS
    -h, --help                            show this help message and exit
    -c, --consensus                       Choose consensus method(s) (default:
    {max_llh,rank,logic_or,logic_and}     ['max_llh'])
    [{max_llh,rank,logic_or,logic_and}
    ...]
    -p, --params PARAMS                   Path to parameter file (default:
                                            params.yaml)
    -s, --sublevel                        Fill information about sub- & super levels
                                            where possible (default: False)
    --sublvls SUBLVLS [SUBLVLS ...]       Indicate what kinds of sublevels exist
                                            (default: ['a', 'b'])
    --lnls-with-sub LNLS_WITH_SUB         List of LNLs where sublevel reporting has
    [LNLS_WITH_SUB ...]                   been performed or is common (default: ['I',
                                            'II', 'V'])
    ```
    """
    with report.status("Read CSV file..."):
        data = pd.read_csv(args.input, header=[0,1,2])
        report.success(f"Read CSV file from {args.input}")

    with report.status("Read in parameters..."):
        with open(args.params, 'r') as params_file:
            params = yaml.safe_load(params_file)
        modalities = params["modalities"]
        report.success(f"Read in parameters from {args.params}")

    with report.status("Compute consensus of modalities..."):
        # pylint: disable=too-many-function-args
        available_mod_keys = set(
            data.columns.get_level_values(0)
        ).intersection(
            modalities.keys()
        )
        available_mods = {key: modalities[key] for key in available_mod_keys}
        num_mods = len(available_mods)

        first_mod = list(available_mods)[0]
        lnls = data[first_mod, "ipsi"].columns
        num_lnls = len(lnls)
        num_patients = len(data)

        for side in ["ipsi", "contra"]:
            # stack observations from different modalities on top of each other
            observation_stack = np.empty(
                shape=(num_patients, num_lnls, num_mods)
            )
            for i, mod in enumerate(available_mods.keys()):
                observation_stack[:,:,i] = data[mod, side].values

            # replace NaNs with Nones
            observation_stack = np.where(
                pd.isna(observation_stack),
                None, observation_stack
            )

            # initialize empty DataFrame for one consensus method and side
            consensus_data = {}
            for cons in args.consensus:
                consensus_multiidx = pd.MultiIndex.from_product(
                    [[cons], [side], lnls]
                )
                consensus_data[cons] = pd.DataFrame(
                    index=data.index,
                    columns=consensus_multiidx
                )

            # go through patients and LNLs and compute consensus for each
            for p in range(num_patients):
                for l in range(num_lnls):
                    observations = observation_stack[p,l]
                    for cons in args.consensus:
                        consensus_data[cons].iloc[p,l] = (
                            CONSENSUS_FUNCS[cons](
                                tuple(observations),
                                available_mods.values()
                            )
                        )

            for cons in args.consensus:
                data = data.join(consensus_data[cons])

        report.success(
            "Computed consensus of observations according to "
            f"the methods {args.consensus}"
        )

    if args.sublevel:
        with report.status("Fixing sub- & super level fields..."):
            data_modalities = set(
                data.columns.get_level_values(0)
            ).intersection(
                [*modalities.keys(), *args.consensus]
            )
            for mod in data_modalities:
                for side in ["ipsi", "contra"]:
                    for lnl in args.lnls_with_sub:
                        sublvl_values = get_sublvl_values(
                            data[mod,side], lnl, args.sublvls
                        )
                        if sublvl_values is None:
                            continue
                        sublvl_involved = np.any(sublvl_values==True, axis=1)
                        sublvls_healthy = np.all(sublvl_values==False, axis=1)
                        data.loc[sublvl_involved, (mod,side,lnl)] = True
                        data.loc[sublvls_healthy, (mod,side,lnl)] = False
            report.success("Fixed sub- & super level fields.")

    with report.status("Saving enhanced file..."):
        args.output.parent.mkdir(exist_ok=True)
        data.to_csv(args.output, index=None)
        report.success(f"Saved enhanced file to disk at {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
