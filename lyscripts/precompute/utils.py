"""
Utilities for precomputing the priors and posteriors.
"""
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def is_hdf5_compatible(value: Any) -> bool:
    """Check if the given ``value`` can be stored in an HDF5 file."""
    return isinstance(value, (bool, str, bytes, int, float, np.ndarray, list, tuple))


def hdf5_dict(attrs: dict[str, Any]) -> dict[str, str]:
    """Convert ``attrs`` to a dictionary of HDF5 compatible attributes or strings."""
    res = {}
    for key, val in attrs.items():
        if is_hdf5_compatible(val):
            res[key] = val
        else:
            res[key] = str(val)
    return res


class HDF5FileCache:
    """HDF5 file acting as a cache for expensive arrays."""
    def __init__(self, file_path: Path) -> None:
        """Initialize the cache with the given ``file_path``."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
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
            if key in file:
                del file[key]
            file[key] = array
            file[key].attrs.update(hdf5_dict(attrs))

    def __contains__(self, key: bytes | str) -> bool:
        with h5py.File(self.file_path, "a") as file:
            return key in file
