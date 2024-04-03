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


def get_modality_subset(diagnosis: dict[str, Any]) -> set[str]:
    """Get the subset of modalities used in the ``scenario``.

    >>> diagnosis = {
    ...     "ipsi": {
    ...         "MRI": {"II": True, "III": False},
    ...         "PET": {"II": False, "III": True},
    ...      },
    ...     "contra": {"MRI": {"II": False, "III": None}},
    ... }
    >>> sorted(get_modality_subset(diagnosis))
    ['MRI', 'PET']
    """
    modality_set = set()

    if "ipsi" not in diagnosis and "contra" not in diagnosis:
        return modality_set | set(diagnosis.keys())

    for side in ["ipsi", "contra"]:
        if side in diagnosis:
            modality_set |= set(diagnosis[side].keys())

    return modality_set


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


def reduce_pattern(pattern: dict[str, dict[str, bool]]) -> dict[str, dict[str, bool]]:
    """Reduce a ``pattern`` by removing all entries that are ``None``.

    This way, it should be completely recoverable by the ``complete_pattern`` function
    but be shorter to store.

    Example:

    >>> full = {
    ...     "ipsi": {"I": None, "II": True, "III": None},
    ...     "contra": {"I": None, "II": None, "III": None},
    ... }
    >>> reduce_pattern(full)
    {'ipsi': {'II': True}}
    """
    tmp_pattern = pattern.copy()
    reduced_pattern = {}
    for side in ["ipsi", "contra"]:
        if not all(v is None for v in tmp_pattern[side].values()):
            reduced_pattern[side] = {}
            for lnl, val in tmp_pattern[side].items():
                if val is not None:
                    reduced_pattern[side][lnl] = val

    return reduced_pattern


def complete_pattern(
    pattern: dict[str, dict[str, bool]] | None,
    lnls: list[str],
) -> dict[str, dict[str, bool]]:
    """Make sure the provided involvement ``pattern`` is correct.

    For each side of the neck, and for each of the ``lnls`` this should in the end
    contain ``True``, ``False`` or ``None``.

    Example:
    >>> pattern = {"ipsi": {"II": True}}
    >>> lnls = ["II", "III"]
    >>> complete_pattern(pattern, lnls)
    {'ipsi': {'II': True, 'III': None}, 'contra': {'II': None, 'III': None}}
    """
    if pattern is None:
        pattern = {}

    for side in ["ipsi", "contra"]:
        if side not in pattern:
            pattern[side] = {}

        for lnl in lnls:
            if lnl not in pattern[side]:
                pattern[side][lnl] = None
            elif pattern[side][lnl] is None:
                continue
            else:
                pattern[side][lnl] = bool(pattern[side][lnl])

    return pattern
