"""General utility functions for the lyscripts package."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from emcee.backends import HDFBackend
from loguru import logger
from rich.console import Console
from scipy.special import factorial

from lyscripts.decorators import (
    check_input_file_exists,
    check_output_dir_exists,
)

console = Console()


def binom_pmf(support: list[int] | np.ndarray, p: float = 0.5):
    """Binomial PMF that is much faster than the one from scipy."""
    max_time = len(support) - 1
    if p > 1.0 or p < 0.0:
        raise ValueError("Binomial prob must be btw. 0 and 1")
    q = 1.0 - p
    binom_coeff = factorial(max_time) / (
        factorial(support) * factorial(max_time - support)
    )
    return binom_coeff * p**support * q ** (max_time - support)


def get_dict_depth(nested: dict) -> int:
    """Get the depth of a nested dictionary.

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
    """Delete private keys from a nested dictionary.

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
    """Flatten ``nested`` dict by creating key tuples for each value at ``max_depth``.

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
    """Take a flat dictionary with tuples of keys and create nested dict from it.

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
    """Of the ``defined_modalities`` return only those mentioned in the ``selection``.

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


def load_patient_data(
    file_path: Path,
    **read_csv_kwargs: dict,
) -> pd.DataFrame:
    """Load patient data from a CSV file stored at ``file``."""
    if "header" not in read_csv_kwargs:
        read_csv_kwargs["header"] = [0, 1, 2]

    data = pd.read_csv(file_path, **read_csv_kwargs)
    logger.info(f"Loaded {len(data)} patient records from {file_path}")
    return data


@check_input_file_exists
def load_yaml_params(file_path: Path) -> dict:
    """Load parameters from a YAML ``file``."""
    with open(file_path, encoding="utf-8") as file:
        loaded_params = yaml.safe_load(file)
        logger.info(f"Loaded YAML parameters from {file_path}")
        return loaded_params


@check_input_file_exists
def load_model_samples(
    file_path: Path,
    name: str = "mcmc",
    flat: bool = True,
    discard: int = 0,
    thin: int = 1,
) -> np.ndarray:
    """Load MCMC samples stored in HDF5 file at ``file_path`` under a key ``name``."""
    backend = HDFBackend(file_path, name=name, read_only=True)
    samples = backend.get_chain(flat=flat, discard=discard, thin=thin)
    logger.info(f"Loaded samples with shape {samples.shape} from {file_path}")
    return samples


@check_output_dir_exists
def get_hdf5_backend(
    file_path: Path,
    dataset: str = "mcmc",
    nwalkers: int | None = None,
    ndim: int | None = None,
    reset: bool = False,
) -> HDFBackend:
    """Open an HDF5 file at ``file_path`` and return a backend."""
    backend = HDFBackend(file_path, name=dataset)
    logger.info(f"Opened HDF5 file at {file_path}")

    if reset:
        logger.info(f"Resetting backend at {file_path} to {nwalkers=} and {ndim=}")
        backend.reset(nwalkers, ndim)

    return backend
