"""
Utilities for precomputing the priors and posteriors.
"""
from pathlib import Path
from typing import Any, TypeVar

import h5py
import numpy as np


def str_dict(attrs: dict[str, Any]) -> dict[str, str]:
    """Convert a dictionary of attributes to a dictionary of strings."""
    return {key: str(value) for key, value in attrs.items()}


HDF5FileCacheT = TypeVar("HDF5FileCacheT", bound="HDF5FileCache")

class HDF5FileCache:
    """HDF5 file acting as a cache for expensive arrays."""
    def __init__(self, file_path: Path) -> None:
        """Initialize the cache with the given ``file_path``."""
        self.file_path = file_path

    def __getitem__(self, key: bytes | str) -> tuple[np.ndarray, dict[str, Any]]:
        with h5py.File(self.file_path, "a") as file:
            array = file[key][()]
            attrs = dict(file[key].attrs)
        return array, attrs

    def __setitem__(
        self,
        key: bytes | str,
        value: tuple[np.ndarray, dict[str, Any]],
    ) -> None:
        array, attrs = value
        with h5py.File(self.file_path, "a") as file:
            file[key] = array
            file[key].attrs.update(str_dict(attrs))

    def __contains__(self, key: bytes | str) -> bool:
        with h5py.File(self.file_path, "a") as file:
            return key in file
