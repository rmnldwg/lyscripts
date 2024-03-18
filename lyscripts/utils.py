"""
This module contains frequently used functions and decorators that are used throughout
the subcommands to load e.g. YAML specifications or model definitions.

It also contains helpers for reporting the script's progress via a slightly customized
`rich` console and a custom `Exception` called `LyScriptsWarning` that can propagate
occuring issues to the right place.
"""
import warnings
from logging import LogRecord
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from deprecated import deprecated
from emcee.backends import HDFBackend
from lymph import diagnose_times, models, types
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from scipy.special import factorial

from lyscripts.decorators import (
    check_input_file_exists,
    check_output_dir_exists,
    log_state,
)

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

LymphModel = models.Unilateral | models.Bilateral | models.Midline
console = Console()


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


def is_streamlit_running() -> bool:
    """Checks if code is running inside a `streamlit` app."""
    return get_script_run_ctx() is not None


class CustomProgress(Progress):
    """Small wrapper around rich's `Progress` initializing my custom columns."""
    def __init__( self, **kwargs: dict):
        columns = [
            SpinnerColumn(finished_text=CHECK),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
        ]
        super().__init__(*columns, **kwargs)


class CustomRichHandler(RichHandler):
    """Uses `func_filepath` from the `extra` dict to modify `pathname`."""
    def emit(self, record: LogRecord) -> None:
        """Emit a log record."""
        if (
            "func_filepath" in record.__dict__
            and "func_name" in record.__dict__
            and "module_name" in record.__dict__
        ):
            prefix = record.pathname.rsplit("lyscripts")[0]
            record.pathname = f"{prefix}lyscripts/{record.func_filepath}"
            record.filename = record.func_filepath.split("/")[-1]
            record.funcName = record.func_name
            record.module = record.module_name
            record.lineno = 0
        return super().emit(record)


def binom_pmf(support: list[int] | np.ndarray, p: float = 0.5):
    """Binomial PMF"""
    max_time = len(support) - 1
    if p > 1. or p < 0.:
        raise ValueError("Binomial prob must be btw. 0 and 1")
    q = 1. - p
    binom_coeff = factorial(max_time) / (factorial(support) * factorial(max_time - support))
    return binom_coeff * p**support * q**(max_time - support)


FUNCS = {
    "binomial": binom_pmf,
}


def graph_from_config(graph_params: dict) -> dict[tuple[str, str], list[str]]:
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
    max_time: int,
):
    """Add margializors over diagnose times to `model`."""
    support = np.arange(max_time + 1)
    for i, stage in enumerate(t_stages):
        if i == 0:
            model.set_distribution(stage, binom_pmf(support, first_binom_prob))
        else:
            model.set_distribution(stage, binom_pmf)


@deprecated(reason="Use new config file version.")
def _create_model_from_v0(params: dict[str, Any]) -> LymphModel:
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
            max_time=params["model"]["max_t"],
        )
    else:
        raise LyScriptsWarning(
            "No model class and constructor params found in YAML file",
            level="error",
        )

    if "modalities" in params:
        for name, modality in params["modalities"].items():
            model.set_modality(name, spec=modality.spec, sens=modality.sens)

    return model


def create_distribution(config: dict[str, Any]) -> diagnose_times.Distribution:
    """Create a distribution instance as defined by a ``config`` dictionary."""
    max_time = config["max_time"]
    kwargs = config.get("kwargs", {})

    if (type_ := config.get("frozen")) is not None:
        kwargs.update({"support": np.arange(max_time+1)})
        distribution = diagnose_times.Distribution(FUNCS[type_](**kwargs))
    elif (type_ := config.get("parametric")) is not None:
        distribution = diagnose_times.Distribution(FUNCS[type_], max_time, **kwargs)
    else:
        raise LyScriptsWarning("No distribution type found", level="error")

    return distribution


@log_state()
def create_model(config: dict[str, Any], config_version: int = 0) -> types.ModelT:
    """Create a model instance as defined by a ``config`` dictionary."""
    if (version := config.get("version", config_version)) == 0:
        return _create_model_from_v0(config)

    if version > 1:
        raise LyScriptsWarning(f"{version=} unsupported", level="error")

    if (graph_config := config.get("graph")) is None:
        raise LyScriptsWarning("No graph definition found in YAML file", level="error")

    if (model_config := config.get("model")) is None:
        raise LyScriptsWarning("No model definition found in YAML file", level="error")

    graph_dict = graph_from_config(graph_config)
    model_cls_name, _, cls_meth_name = model_config["class"].partition(".")
    model_cls = getattr(models, model_cls_name)
    if cls_meth_name is not None:
        model_cls = getattr(model_cls, cls_meth_name)
    model_kwargs = model_config.get("kwargs", {})
    model = model_cls(graph_dict, **model_kwargs)

    if (defined_modalities := config.get("modalities")) is not None:
        for modality in model_config["modalities"]:
            model.set_modality(modality, *defined_modalities[modality])

    for t_stage, dist_config in model_config.get("distributions", {}).items():
        distribution = create_distribution(dist_config)
        model.set_distribution(t_stage, distribution)

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
    if not isinstance(nested, dict):
        return 0

    max_depth = None
    for _, value in nested.items():
        value_depth = get_dict_depth(value)
        max_depth = max(max_depth or value_depth, value_depth)

    return 1 + (max_depth or 0)


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
@check_input_file_exists
def load_patient_data(
    file_path: Path,
    header: list[int] | None = None,
) -> pd.DataFrame:
    """Load patient data from a CSV file stored at `file`."""
    if header is None:
        header = [0,1,2]
    return pd.read_csv(file_path, header=header)


@log_state()
@check_input_file_exists
def load_yaml_params(file_path: Path) -> dict:
    """Load parameters from a YAML `file`."""
    with open(file_path, encoding="utf-8") as file:
        params = yaml.safe_load(file)
    return params


@log_state()
@check_input_file_exists
def load_model_samples(
    file_path: Path,
    name: str = "mcmc",
    flat: bool = True,
    discard: int | None = None,
    thin: int | None = None,
) -> np.ndarray:
    """
    Load samples produced by an MCMC sampling process that are stored at
    `file_path` in an HDF5 format.
    """
    backend = HDFBackend(file_path, name=name, read_only=True)
    return backend.get_chain(flat=flat, dicard=discard, thin=thin)


@log_state()
@check_output_dir_exists
def initialize_backend(
    file_path: Path,
    nwalkers: int | None = None,
    ndim: int | None = None,
    name: str = "mcmc",
    reset: bool = False,
) -> HDFBackend:
    """Open an HDF5 file at `file_path` and return a backend."""
    backend = HDFBackend(file_path, name=name)

    if reset:
        backend.reset(nwalkers, ndim)

    return backend
