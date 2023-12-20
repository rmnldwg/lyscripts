"""
This module contains frequently used functions and decorators that are used throughout
the subcommands to load e.g. YAML specifications or model definitions.

It also contains helpers for reporting the script's progress via a slightly customized
`rich` console and a custom `Exception` called `LyScriptsWarning` that can propagate
occuring issues to the right place.
"""
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import h5py
import numpy as np
import pandas as pd
import yaml
from emcee.backends import HDFBackend
from lymph import models
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from scipy.special import factorial

from lyscripts.decorators import (
    check_input_file_exists,
    log_state,
    provide_file,
)

try:
    import streamlit
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    streamlit.status = streamlit.spinner
except ImportError:
    def get_script_run_ctx() -> bool:
        """A mock for the `get_script_run_ctx` function of `streamlit`."""
        return None


LymphModel = models.Unilateral | models.Bilateral # | MidlineBilateral


class LyScriptsWarning(Warning):
    """
    Exception that can be raised by methods if they want the `LyScriptsReport` instance
    to not stop and print a traceback, but display some message appropriately.

    Essentially, this is a way for decorated functions to propagate messages through
    the `report_state` decorator.
    """
    def __init__(self, *args: object, level: str = "info") -> None:
        """Extract the `level` of the message (can be "info", "warning" or "error")."""
        self.level = level
        self.message = args[0]
        super().__init__(*args)


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


def inject_lvl_and_symbol(objects, level = "INFO", symbol = None, width = 8):
    """Nicely format the `objects` to be printed with a `level` and `symbol`."""
    prefix = "[blue]" + level.ljust(width) + "[/blue]"
    if symbol is not None:
        objects = [prefix, symbol, *objects]
    else:
        objects = [prefix, *objects]
    return objects


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
        objects = inject_lvl_and_symbol(objects, symbol=CHECK)
        return super().print(*objects, **kwargs)

    @redirect_to_streamlit
    def info(self, *objects, **kwargs) -> None:
        """Prefix a bold yellow circle to any output."""
        objects = inject_lvl_and_symbol(objects, symbol=CIRCL)
        return super().print(*objects, **kwargs)

    @redirect_to_streamlit
    def add(self, *objects, **kwargs) -> None:
        """Prefix a bold yellow circle to any output."""
        objects = inject_lvl_and_symbol(objects, symbol="+")
        return super().print(*objects, **kwargs)

    @redirect_to_streamlit
    def warning(self, *objects, **kwargs) -> None:
        """Prefix a bold yellow triangle to any output."""
        objects = inject_lvl_and_symbol(objects, symbol=WARN)
        return super().print(*objects, **kwargs)

    @redirect_to_streamlit
    def error(self, *objects, **kwargs) -> None:
        """Prefix a bold red cross to any output."""
        objects = inject_lvl_and_symbol(objects, symbol=CROSS)
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
        prefix = " ".join(inject_lvl_and_symbol([]))
        columns = [
            TextColumn(prefix),
            SpinnerColumn(finished_text=CHECK),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
        ]
        super().__init__(*columns, **kwargs)


def binom_pmf(k: list[int] | np.ndarray, n: int, p: float):
    """Binomial PMF"""
    if p > 1. or p < 0.:
        raise ValueError("Binomial prob must be btw. 0 and 1")
    q = 1. - p
    binom_coeff = factorial(n) / (factorial(k) * factorial(n - k))
    return binom_coeff * p**k * q**(n - k)


def parametric_binom_pmf(support: np.ndarray, p: float = 0.5) -> Callable:
    """Return a parametric binomial PMF"""
    return binom_pmf(k=support, n=support[-1], p=p)


def graph_from_config(graph_params: dict):
    """Build graph dictionary for the `lymph` models from the YAML params."""
    lymph_graph = {}

    if not "tumor" in graph_params and "lnl" in graph_params:
        raise KeyError("Parameters must define tumors and LNLs")

    for node_type, node_dict in graph_params.items():
        for node_name, out_connections in node_dict.items():
            lymph_graph[(node_type, node_name)] = out_connections

    return lymph_graph


def add_tstage_marg(
    model: LymphModel,
    t_stages: list[str],
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
            model.diag_time_dists[stage] = parametric_binom_pmf


def model_from_config(
    graph_params: dict[str, Any],
    model_params: dict[str, Any],
    modalities_params: dict[str, Any] | None = None,
) -> LymphModel:
    """Create a model instance as defined by some YAML params."""
    graph = graph_from_config(graph_params)

    model_cls = getattr(models, model_params["class"])
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


@log_state()
def create_model_from_config(params: dict[str, Any]) -> LymphModel:
    """Create a model instance as defined by some YAML params."""
    if "graph" in params:
        graph = graph_from_config(params["graph"])
    else:
        raise LyScriptsWarning("No graph definition found in YAML file", level="error")

    if "model" in params:
        model_cls = getattr(models, params["model"]["class"])
        if not "is_symmetric" in params["model"]["kwargs"]:
            warnings.warn(
                "The keywords `base_symmetric`, `trans_symmetric`, and `use_mixing` "
                "have been deprecated. Please use `is_symmetric` instead.",
                DeprecationWarning,
            )
            params["model"]["kwargs"]["is_symmetric"] = {
                "tumor_spread": params["model"]["kwargs"].pop("base_symmetric", False),
                "lnl_spread": params["model"]["kwargs"].pop("trans_symmetric", True),
            }
        model = model_cls(graph, **params["model"]["kwargs"])

        add_tstage_marg(
            model,
            t_stages=params["model"]["t_stages"],
            first_binom_prob=params["model"]["first_binom_prob"],
            max_t=params["model"]["max_t"],
        )
    else:
        raise LyScriptsWarning(
            "No model class and constructor params found in YAML file",
            level="error",
        )

    if "modalities" in params:
        model.modalities = params["modalities"]

    return model


def get_dict_depth(nested: dict) -> int:
    """
    Get the depth of a nested dictionary.

    For example:
    >>> get_dict_depth({"a": {"b": 1}})
    2
    >>> varying_depth = {"a": {"b": 1}, "c": {"d": {"e": 2}}}
    >>> get_dict_depth(varying_depth)
    3
    """
    if isinstance(nested, dict):
        max_depth = None
        for _, value in nested.items():
            value_depth = get_dict_depth(value)
            max_depth = max(max_depth or value_depth, value_depth)

        return 1 + (max_depth or 0)

    return 0


def delete_private_keys(nested: dict) -> dict:
    """
    Delete private keys from a nested dictionary.

    A 'private' key is a key whose name starts with an underscore. For example:
    >>> delete_private_keys({"patient": {"__doc__": "some patient info", "age": 61}})
    {'patient': {'age': 61}}
    >>> delete_private_keys({"patient": {"age": 61}})
    {'patient': {'age': 61}}
    """
    cleaned = {}

    if isinstance(nested, dict):
        for key, value in nested.items():
            if not (isinstance(key, str) and key.startswith("_")):
                cleaned[key] = delete_private_keys(value)
    else:
        cleaned = nested

    return cleaned


def flatten(
    nested: dict,
    prev_key: tuple = (),
    max_depth: int | None = None,
) -> dict:
    """
    Flatten a `nested` dictionary by creating key tuples for each value at `max_depth`.

    For example:
    >>> nested = {"tumor": {"1": {"t_stage": 1, "size": 12.3}}}
    >>> flatten(nested)
    {('tumor', '1', 't_stage'): 1, ('tumor', '1', 'size'): 12.3}
    >>> mapping = {"patient": {"#": {"age": {"func": int, "columns": ["age"]}}}}
    >>> flatten(mapping, max_depth=3)
    {('patient', '#', 'age'): {'func': <class 'int'>, 'columns': ['age']}}

    Note that flattening an already flat dictionary will yield some weird results.
    """
    result = {}

    for key, value in nested.items():
        is_dict = isinstance(value, dict)
        has_reached_max_depth = max_depth is not None and len(prev_key) >= max_depth - 1

        if is_dict and not has_reached_max_depth:
            result.update(flatten(value, (*prev_key, key), max_depth))
        else:
            result[(*prev_key, key)] = value

    return result


def unflatten(flat: dict) -> dict:
    """
    Take a flat dictionary with tuples of keys and create nested dict from it.

    Like so:
    >>> flat = {('tumor', '1', 't_stage'): 1, ('tumor', '1', 'size'): 12.3}
    >>> unflatten(flat)
    {'tumor': {'1': {'t_stage': 1, 'size': 12.3}}}
    >>> mapping = {('patient', '#', 'age'): {'func': int, 'columns': ['age']}}
    >>> unflatten(mapping)
    {'patient': {'#': {'age': {'func': <class 'int'>, 'columns': ['age']}}}}
    """
    result = {}

    for keys, value in flat.items():
        current = result
        for key in keys[:-1]:
            current = current.setdefault(key, {})

        current[keys[-1]] = value

    return result


def get_modalities_subset(
    defined_modalities: dict[str, list[float]],
    selection: list[str],
) -> dict[str, list[float]]:
    """
    Of the `defined_modalities` return only those mentioned in the `selection`.

    For instance:
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


@log_state()
@provide_file(is_binary=False)
def load_patient_data(
    file: TextIO,
    header_rows: list[int] | None = None,
) -> pd.DataFrame:
    """Load patient data from a CSV file stored at `file`."""
    if header_rows is None:
        header_rows = [0,1,2]
    return pd.read_csv(file, header=header_rows)


@log_state()
@provide_file(is_binary=False)
def load_yaml_params(file: Path) -> dict:
    """Load parameters from a YAML `file`."""
    return yaml.safe_load(file)


@log_state()
@check_input_file_exists
def load_model_samples(file_path: Path) -> np.ndarray:
    """
    Load samples produced by an MCMC sampling process that are stored at
    `file_path` in an HDF5 format.
    """
    backend = HDFBackend(file_path, read_only=True)
    return backend.get_chain(flat=True)


@log_state()
@provide_file(is_binary=True)
def load_hdf5_samples(file: BinaryIO, name: str = "mcmc/chain") -> np.ndarray:
    """
    Load a chain of samples from an uploaded HDF5 `file` that is stored in the dataset
    called `name`.
    """
    with h5py.File(file, mode="r") as h5file:
        try:
            samples = h5file[name][:]
        except KeyError as key_err:
            raise KeyError("Dataset `mcmc` not in the HDF5 file.") from key_err

        new_shape = (samples.shape[0] * samples.shape[1], samples.shape[2])
        flattened_samples = samples.reshape(new_shape)
        return flattened_samples
