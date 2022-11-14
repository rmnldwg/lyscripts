"""
This module contains frequently used functions as well as instructions on how
to parse and process the raw data from different institutions
"""
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

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

try:
    import streamlit
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    streamlit.status = streamlit.spinner
except ImportError:
    def get_script_run_ctx() -> bool:
        """A mock for the `get_script_run_ctx` function of `streamlit`."""
        return None


CROSS = "[bold red]✗[/bold red]"
CIRCL = "[bold blue]∘[/bold blue]"
WARN = "[bold yellow]Δ[/bold yellow]"
CHECK = "[bold green]✓[/bold green]"


def is_streamlit_running() -> bool:
    """Checks if code is running inside a `streamlit` app."""
    return get_script_run_ctx() is not None


def redirect_to_streamlit(func: Callable) -> Callable:
    """
    If this method detects that it is called from within a `streamlit`
    application, the output is redirected to the appropriate `streamlit` function.
    """
    func_name = func.__name__

    def inner(self, *objects, **kwargs) -> Any:
        """Wrapper function."""
        if is_streamlit_running():
            return getattr(streamlit, func_name)(" ".join(objects))

        return func(self, *objects, **kwargs)

    return inner


class LyScriptsReport(Console):
    """
    Small extension to the `Console` class of the rich package.
    """

    @redirect_to_streamlit
    def status(self, *objects, **kwargs):
        """Re-implement `status` method to allow decoration."""
        return super().status(*objects, **kwargs)

    @redirect_to_streamlit
    def success(self, *objects, **kwargs) -> None:
        """Prefix a bold green check mark to any output."""
        objects = [CHECK, *objects]
        return super().print(*objects, **kwargs)

    @redirect_to_streamlit
    def info(self, *objects, **kwargs) -> None:
        """Prefix a bold yellow circle to any output."""
        objects = [CIRCL, *objects]
        return super().print(*objects, **kwargs)

    @redirect_to_streamlit
    def warning(self, *objects, **kwargs) -> None:
        """Prefix a bold yellow triangle to any output."""
        objects = [WARN, *objects]
        return super().print(*objects, **kwargs)

    @redirect_to_streamlit
    def error(self, *objects, **kwargs) -> None:
        """Prefix a bold red cross to any output."""
        objects = [CROSS, *objects]
        return super().print(*objects, **kwargs)

    def exception(self, exception, **kwargs) -> None:
        """Display a traceback either via `streamlit` or in the console."""
        if is_streamlit_running():
            return streamlit.exception(exception)
        else:
            return super().print_exception(extra_lines=4, show_locals=True, **kwargs)

report = LyScriptsReport()


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


class LyScriptsWarning(Warning):
    """
    Exception that can be raised by methods if they want the `LyScriptsReport` instance
    to not stop and print a traceback, but display some message appropriately.

    Essentially, this is a way for decorated functions to propagate messages through
    the `report_state` decorator.
    """
    def __init__(self, *args: object, level: str = "info") -> None:
        """Extract the `level` of the message (can be _info_, _warning_ or _error_)."""
        self.level = level
        self.message = args[0]
        super().__init__(*args)


def report_state(
    status_msg: str,
    success_msg: str,
) -> Callable:
    """
    Outermost decorator that catches and gracefully reports exceptions that occur.

    During the execution of the decorated function, it will display the `status_msg`.
    When successful, the `success_msg` will finally be printed. And if the decorated
    function raises a `LyScriptsError`, then that exception's message will be passed on
    to the methods of the reporting class/module.
    """
    def assembled_decorator(func: Callable) -> Callable:
        """The decorator that gets returned by `report_state`."""
        with report.status(status_msg):
            def inner(*args, **kwargs):
                """The wrapped function."""
                try:
                    result = func(*args, **kwargs)
                except LyScriptsWarning as ly_err:
                    msg = getattr(ly_err, "message", repr(ly_err))
                    level = getattr(ly_err, "level", "info")
                    getattr(report, level)(msg)
                except Exception as exc:
                    report.exception(exc)
                    sys.exit()
                else:
                    report.success(success_msg)
                    return result

                return None

        return inner

    return assembled_decorator


def check_input_file_exists(loading_func: Callable) -> Callable:
    """
    Decorator that checks if the file path provided to the `loading_func` exists and
    throws a `FileNotFoundError` if it does not.

    The purpose of this deorator is to provide a consistent error message to the
    `report_func_state`, since some libraries throw other errors when a file is not
    found.
    """
    def inner(file_path: str, *args, **kwargs) -> Any:
        """Wrapped loading function."""
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"No file found at {file_path}")

        return loading_func(file_path, *args, **kwargs)

    return inner


def check_output_dir_exists(saving_func: Callable) -> Callable:
    """
    Decorator to make sure the parent directory of the file that is supposed to be
    saved does exist.
    """
    def inner(file_path: str, *args, **kwargs) -> Any:
        """Wrapped saving function."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        return saving_func(file_path, *args, **kwargs)

    return inner


@report_state(
    status_msg="Load YAML params...",
    success_msg="Loaded YAML params.",
)
@check_input_file_exists
def load_yaml_params(file_path: Path) -> dict:
    """Load parameters from a YAML file at `file_path`."""
    with open(file_path, mode="r", encoding="utf-8") as file_content:
        params = yaml.safe_load(file_content)
        return params


@report_state(
    status_msg="Load HDF5 samples from MCMC run...",
    success_msg="Loaded HDF5 samples from MCMC run.",
)
@check_input_file_exists
def load_model_samples(file_path: Path) -> np.ndarray:
    """
    Load samples produced by an MCMC sampling process that are stored at
    `file_path` in an HDF5 format.
    """
    backend = HDFBackend(file_path, read_only=True)
    return backend.get_chain(flat=True)
