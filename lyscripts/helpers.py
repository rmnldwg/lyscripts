"""
This module contains frequently used functions as well as instructions on how
to parse and process the raw data from different institutions
"""
from typing import Union, Dict, Any, List, Optional, Callable

import numpy as np
from scipy.special import factorial
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

import lymph


CROSS = "[bold red]✗[/bold red]"
CIRCL = "[bold yellow]∘[/bold yellow]"
CHECK = "[bold green]✓[/bold green]"

class ConsoleReport(Console):
    """
    Small extension to the `Console` class of the rich package.
    """
    def success(self, *objects, **kwargs) -> None:
        """Prefix a bold green check mark to any output."""
        objects = [CHECK, *objects]
        return super().print(*objects, **kwargs)

    def info(self, *objects, **kwargs) -> None:
        """Prefix a bold yellow circle to any output."""
        objects = [CIRCL, *objects]
        return super().print(*objects, **kwargs)

    def failure(self, *objects, **kwargs) -> None:
        """Prefix a bold red cross to anything printed."""
        objects = [CROSS, *objects]
        return super().print(*objects, **kwargs)

report = ConsoleReport()

class CustomProgress(Progress):
    """Small wrapper around rich's `Progress` initializing my custom columns."""
    def __init__( self, **kwargs: dict):
        columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
        ]
        super().__init__(*columns, **kwargs)

report_progress = CustomProgress()


def binom_pmf(k: Union[List[int], np.ndarray], n: int, p: float):
    """Binomial PMF"""
    if p > 1. or p < 0.:
        raise ValueError("Binomial prob must be btw. 0 and 1")
    q = (1. - p)
    binom_coeff = factorial(n) / (factorial(k) * factorial(n - k))
    return binom_coeff * p**k * q**(n - k)

def parametric_binom_pmf(n: int) -> Callable:
    """Return a parametric binomial PMF"""
    def inner(t, p):
        """Parametric binomial PMF"""
        return binom_pmf(t, n, p)
    return inner

def graph_from_config(graph_params: dict):
    """
    Build the graph for the `lymph` model from the graph in the params file. I cannot
    simply write the graph in the params file as I like because YAML does not support
    tuples as keys in a dictionary.
    """
    lymph_graph = {}

    if not "tumor" in graph_params and "lnl" in graph_params:
        raise KeyError("Parameters must define tumors and LNLs")

    for node_type, node_dict in graph_params.items():
        for node_name, out_connections in node_dict.items():
            lymph_graph[(node_type, node_name)] = out_connections

    return lymph_graph

def add_tstage_marg(
    model: Union[lymph.Unilateral, lymph.Bilateral, lymph.MidlineBilateral],
    t_stages: List[str],
    first_binom_prob: float,
    max_t: int,
):
    """Add margializors over diagnose times to `model`."""
    for i,stage in enumerate(t_stages):
        if i == 0:
            model.diag_time_dists[stage] = binom_pmf(
                k=np.arange(max_t + 1),
                n=max_t,
                p=first_binom_prob
            )
        else:
            model.diag_time_dists[stage] = parametric_binom_pmf(n=max_t)

def model_from_config(
    graph_params: Dict[str, Any],
    model_params: Dict[str, Any],
    modalities_params: Optional[Dict[str, Any]] = None,
) -> Union[lymph.Unilateral, lymph.Bilateral, lymph.MidlineBilateral]:
    """Create a model instance as defined by some YAML params."""
    graph = graph_from_config(graph_params)

    model_cls = getattr(lymph, model_params["class"])
    model = model_cls(graph, **model_params["kwargs"])

    if modalities_params is not None:
        model.modalities = modalities_params

    add_tstage_marg(
        model,
        t_stages=model_params["t_stages"],
        first_binom_prob=model_params["first_binom_prob"],
        max_t=model_params["max_t"],
    )

    return model
