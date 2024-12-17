import argparse
import logging
from pathlib import Path

from cycler import cycler
import scipy as sp
import numpy as np
import emcee
from lymph import models
import matplotlib.pyplot as plt
import pandas as pd

from lyscripts.plot.utils import COLORS, save_figure
from lymixture.em import _set_params, expectation

from lyscripts.utils import (
    create_mixture,
    load_patient_data,
    load_yaml_params,
    assign_modalities
)

logger = logging.getLogger(__name__)

def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """Add an ``ArgumentParser`` to the subparsers action."""
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """Add arguments to the parser."""
    parser.add_argument(
        "--input", type=Path,
        help="File path with emcee backend of samples"
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output path for the plot"
    )
    parser.add_argument(
        "-m", "--mode", type=str, default = "fixed_mixture",
        help = "Mode of sampling. Use either 'fixed_mixture' or 'fixed_latent'"
    )
    parser.add_argument(
        "-s", "--size", type=int, default = 200,
        help = "Number of samples to be used for plotting"
    )
    parser.add_argument(
        "-p", "--params", default="./params.yaml", type=Path,
        help="Path to parameter file."
    )
    parser.add_argument(
        "-d", "--data", type=Path, required=True,
        help="Path to patient data."
    )
    parser.add_argument(
        "-mp", "--model_params", type=Path, required = True,
        help="File path of mixture coefficients"
    )

    parser.set_defaults(run_main=main)
    
def multiple_plotter(dataset, risk_dictionary_extended, subsite, stage = ''):
    hist_cycl = (
        cycler(histtype=["stepfilled", "step"])
        * cycler(color=list(COLORS.values()))
    )
    line_cycl = (
        cycler(linestyle=["-", "--"])
        * cycler(color=list(COLORS.values()))
    )
    dataset_staging = dataset.copy()
    dataset_staging['tumor','1','t_stage'] = dataset_staging['tumor','1','t_stage'].replace([0,1,2], 'early')
    dataset_staging['tumor','1','t_stage'] = dataset_staging['tumor','1','t_stage'].replace([3,4], 'late')
    if stage == 'early' or stage == 'late':
        data_selected = dataset_staging.loc[(dataset_staging['tumor']['1']['subsite'] == subsite) & (dataset_staging['tumor']['1']['t_stage'] == stage)]  
    else:
        data_selected = dataset_staging.loc[dataset_staging['tumor']['1']['subsite'] == subsite]
    min_value = 0
    max_value = 100
    prevalence = {} 
    number_of_patients = {}
    risks = {}
    risk_dictionary = risk_dictionary_extended[subsite]
    for key in risk_dictionary.keys():
        prevalence[key] = (data_selected['max_llh']['ipsi'][key] == True).sum()
        number_of_patients[key] = len(data_selected)
        risks[key] = np.array(risk_dictionary[key])*100
    
    num_matches = [prevalence[key] for key in risk_dictionary.keys()]
    num_totals = [number_of_patients[key] for key in risk_dictionary.keys()]
    values = [risks[key] for key in risk_dictionary.keys()]
    hist_kwargs = {
                "bins": np.linspace(min_value, max_value, 80),
                "density": True,
                "alpha": 0.6,
                "linewidth": 2.,
            }
    fig, ax = plt.subplots(figsize=(12,4))

    x = np.linspace(min_value, max_value, 200)
    zipper = zip(values, risk_dictionary.keys(), num_matches, num_totals, hist_cycl, line_cycl)
    for vals, label, a, n, hstyle, lstyle in zipper:
        ax.hist(
            vals,
            label=label,
            **hist_kwargs,
            **hstyle
        )
        if not np.isnan(a):
            post = sp.stats.beta.pdf(x / 100., a+1, n-a+1) / 100.
            ax.plot(x, post, label=f"{int(a)}/{int(n)}", **lstyle)
        ax.legend()
        ax.set_xlabel("probability [%]")
    fig.suptitle(f"Risk distributions {stage} for subsite {subsite}",fontsize = 16)
    return fig
    
def multiple_plotter_component(risk_dictionary_extended, component, stage = ''):
    hist_cycl = (
        cycler(histtype=["stepfilled", "step"])
        * cycler(color=list(COLORS.values()))
    )
    line_cycl = (
        cycler(linestyle=["-", "--"])
        * cycler(color=list(COLORS.values()))
    )
    min_value = 0
    max_value = 100
    prevalence = {}
    number_of_patients = {}
    risks = {}
    risk_dictionary = risk_dictionary_extended[component]
    for key in risk_dictionary.keys():
        risks[key] = np.array(risk_dictionary[key])*100
    
    values = [risks[key] for key in risk_dictionary.keys()]
    hist_kwargs = {
                "bins": np.linspace(min_value, max_value, 80),
                "density": True,
                "alpha": 0.6,
                "linewidth": 2.,
            }
    fig, ax = plt.subplots(figsize=(12,4))

    x = np.linspace(min_value, max_value, 200)
    zipper = zip(values, risk_dictionary.keys(), hist_cycl)
    for vals, label, hstyle in zipper:
        ax.hist(
            vals,
            label=label,
            **hist_kwargs,
            **hstyle
        )
        ax.legend()
        ax.set_xlabel("probability [%]")
    fig.suptitle(f"Risk distributions {stage} for component {component}",fontsize = 16)
    return fig
    

def main(args: argparse.Namespace):
    params = load_yaml_params(args.params)
    inference_data = load_patient_data(args.data)
    backend = emcee.backends.HDFBackend(args.input)
    samples = backend.get_chain(flat = True)
    model_params = pd.read_csv(args.model_params,header = [0])

    # ugly, but necessary for pickling
    global MIXTURE
    MIXTURE = create_mixture(params)

    mapping = params["model"].get("mapping", None)
    if isinstance(MIXTURE.components[0], models.Unilateral):
        side = params["model"].get("side", "ipsi")
        MIXTURE.load_patient_data(inference_data, split_by= params["model"].get("split_by", ("tumor", "1", "subsite")), mapping=mapping)
        assign_modalities(model=MIXTURE, config=params.get("inference_modalities", {}))

    else:
        raise "Only Unilateral has been implemented so far"

    param_dict = dict(model_params.iloc[-1])
    MIXTURE.set_params(**param_dict)
    MIXTURE.set_resps(expectation(MIXTURE, param_dict))

    print(MIXTURE.get_params())
    lnls = list(MIXTURE.components[0].graph.lnls.keys())
    component_list = list(range(len(MIXTURE.components)))
    component_dictionary_early_full_sampling = {}
    component_dictionary_late_full_sampling = {}

    for component in component_list:
        component_dictionary_early_full_sampling[str(component)] = {
            lnl: [] for lnl in lnls  
        }
        component_dictionary_late_full_sampling[str(component)] = {
            lnl: [] for lnl in lnls
        }
    
    subsite_dictionary_early_full_sampling = {}
    subsite_dictionary_late_full_sampling = {}
    subsite_list = list(MIXTURE.subgroups.keys())

    for subsite in subsite_list:
        subsite_dictionary_early_full_sampling[subsite] = {
            lnl: [] for lnl in lnls  # Dynamically generate keys from the lnls list
        }
        subsite_dictionary_late_full_sampling[subsite] = {
            lnl: [] for lnl in lnls
        }

    involvement_dict = {lnl: {lnl: True} for lnl in lnls}
    samples_thinned = samples[::int(np.round(len(samples)/args.size,0))]
    for round, sample in enumerate(samples_thinned):
        if args.mode == "fixed_latent":
            MIXTURE.set_params(*sample)
        elif args.mode == "fixed_mixture":
            _set_params(MIXTURE, sample)
        else:
            raise ValueError("Invalid mode")
        component_dictionary_early_full_sampling['0']['II'].append(MIXTURE.components[0].risk(involvement = involvement_dict['II'],t_stage = 'early'))
        for component in component_list:
            for lnl in lnls:
                component_dictionary_early_full_sampling[str(component)][lnl].append(MIXTURE.components[component].risk(involvement = involvement_dict[lnl],t_stage = 'early'))  
                component_dictionary_late_full_sampling[str(component)][lnl].append(MIXTURE.components[component].risk(involvement = involvement_dict[lnl],t_stage = 'late'))
        
        for subsite in subsite_list:
            for lnl in lnls:
                subsite_dictionary_early_full_sampling[subsite][lnl].append(MIXTURE.risk(subgroup = subsite, involvement = involvement_dict[lnl],t_stage = 'early'))
                subsite_dictionary_late_full_sampling[subsite][lnl].append(MIXTURE.risk(subgroup = subsite, involvement = involvement_dict[lnl],t_stage = 'late'))
        print(round, ' done')
    print(inference_data)
    for component in component_list:
        fig = multiple_plotter_component(component_dictionary_early_full_sampling, str(component), stage = 'early')
        save_figure(args.output/f"component_{component}_early", fig, formats = ['png','svg'])
        fig = multiple_plotter_component(component_dictionary_late_full_sampling, str(component), stage = 'late')
        save_figure(args.output/f"component_{component}_late", fig, formats = ['png','svg'])
    for subsite in subsite_list:
        fig = multiple_plotter(inference_data, subsite_dictionary_early_full_sampling, subsite, stage = 'early')
        save_figure(args.output/f"{subsite}_early", fig, formats = ['png','svg'])
        fig = multiple_plotter(inference_data,subsite_dictionary_late_full_sampling, subsite, stage = 'late')
        save_figure(args.output/f"{subsite}_late", fig, formats = ['png','svg'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)

