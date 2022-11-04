"""
This module contains frequently used functions as well as instructions on how
to parse and process the raw data from different institutions
"""
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lymph
import numpy as np
import yaml
from emcee.backends import HDFBackend
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from scipy.special import factorial

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


def get_lnls(model) -> List[str]:
    """Extract the list of LNLs from a model instance. E.g.:
    >>> graph = {
    ...     ("tumor", "T"): ["II", "III"],
    ...     ("lnl", "II"): ["III"],
    ...     ("lnl", "III"): [],
    ... }
    >>> model = lymph.Unilateral(graph)
    >>> get_lnls(model)
    ['II', 'III']
    """
    if isinstance(model, lymph.Unilateral):
        return [lnl.name for lnl in model.lnls]
    elif isinstance(model, lymph.Bilateral):
        return [lnl.name for lnl in model.ipsi.lnls]
    elif isinstance(model, lymph.MidlineBilateral):
        return [lnl.name for lnl in model.ext.ipsi.lnls]
    else:
        raise TypeError(f"Model cannot be of type {type(model)}")


def flatten(
    nested: dict,
    prev_key: tuple = (),
    flattened: Optional[dict] = None,
) -> dict:
    """
    Flatten a `nested` dictionary recursivel by extending the `prev_key` tuple. For
    example:
    >>> nested = {"hi": {"there": "buddy"}, "how": {"are": "you?"}}
    >>> flatten(nested)
    {('hi', 'there'): 'buddy', ('how', 'are'): 'you?'}
    """
    if flattened is None:
        flattened = {}

    for key,val in nested.items():
        if isinstance(val, dict):
            flatten(val, (*prev_key, key), flattened)
        else:
            flattened[(*prev_key, key)] = val

    return flattened


def get_modalities_subset(
    defined_modalities: Dict[str, List[float]],
    selection: List[str],
) -> Dict[str, List[float]]:
    """
    Of the `defined_modalities` return only those mentioned in the `selection`. For
    instance:
    >>> modalities = {"CT": [0.76, 0.81], "MRI": [0.63, 0.86]}
    >>> get_modalities_subset(modalities, ["CT"])
    {'CT': [0.76, 0.81]}
    """
    selected_modalities = {}
    for mod in selection:
        try:
            selected_modalities[mod] = defined_modalities[mod]
        except KeyError as key_err:
            raise KeyError(f"Modality {mod} has not been defined yet") from key_err
    return selected_modalities


def report_func_state(
    status_msg: str,
    success_msg: str,
    actions: Optional[Dict[type, Tuple[bool, Callable, str]]] = None,
) -> Callable:
    """
    Decorator to report the state of a function. E.g., inform the user whether it
    succeeded or failed to execute the desired action.

    The `status_msg` will be shown during the function's execution and the `success_msg`
    when the function was executed without any exceptions. The `actions` dictionary
    defines what to report and what to do for each error type that might occur. Its
    keys are exception types (e.g. `FileNotFoundError`). Its values are tuples of
    `(do_stop, report.func, message)` where the boolean `do_stop` indicates whether the
    execution of the entire program should stop, the `report.func` defines which
    function should be called with the `message` as an argument to report what happens.

    This should be the outermost decorator.
    """
    actions = {} if actions is None else actions

    dflt_action = (
        True,
        report.failure,
        "Unexpected exception, stopping."
    )

    def assembled_decorator(func: Callable) -> Callable:
        """
        This is the decorator that gets assembled, when providing the outer function
        is called with its arguments.
        """
        with report.status(status_msg):
            def inner(*args, **kwargs) -> Any:
                """The returned, wrapped function."""
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    do_stop, report_func, message = actions.get(type(exc), dflt_action)
                    report_func(message)
                    if do_stop:
                        sys.exit()
                else:
                    report.success(success_msg)
                    return result

                return None

        return inner
    return assembled_decorator


def check_file_exists(loading_func: Callable) -> Callable:
    """
    Decorator that checks if the file path provided to the `loading_func` exists and
    throws a `FileNotFoundError` if it does not.

    The purpose of this deorator is to provide a consistent error message to the
    `report_func_state`, since some libraries throw other errors when a file is not
    found.
    """
    def inner(file_path, *args, **kwargs) -> Any:
        """Wrapped function."""
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"No file found at {file_path}")

        return loading_func(file_path, *args, **kwargs)

    return inner


@check_file_exists
def load_yaml_params(file_path: Path) -> dict:
    """Load parameters from a YAML file at `file_path`."""
    with open(file_path, mode="r", encoding="utf-8") as file_content:
        params = yaml.safe_load(file_content)
        return params


cli_load_yaml_params = report_func_state(
    status_msg="Load YAML params...",
    success_msg="Loaded YAML params.",
    actions={
        FileNotFoundError: (True, report.failure, "YAML file not found, stopping."),
        yaml.parser.ParserError: (True, report.failure, "Invalid YAML file, stopping"),
    }
)(load_yaml_params)
"""
The `load_yaml_params` function wrapped by the `report_func_state` such that error
messages are directed to a `rich` console.
"""


@check_file_exists
def load_model_samples(file_path: Path) -> np.ndarray:
    """
    Load samples produced by an MCMC sampling process that are stored at
    `file_path` in an HDF5 format.
    """
    backend = HDFBackend(file_path, read_only=True)
    return backend.get_chain(flat=True)


cli_load_model_samples = report_func_state(
    status_msg="Load HDF5 samples from MCMC run...",
    success_msg="Loaded HDF5 samples from MCMC run.",
    actions={
        FileNotFoundError: (True, report.failure, "HDF5 file not found, stopping."),
        AttributeError: (True, report.failure, "No HDF5 file or no MCMC data present.")
    }
)(load_model_samples)
"""
The `load_model_samples` function wrapped by the `report_func_state` such that error
messages are directed to a `rich` console.
"""
