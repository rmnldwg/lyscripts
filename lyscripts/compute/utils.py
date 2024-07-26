"""Utilities for precomputing the priors and posteriors."""

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def is_hdf5_compatible(value: Any) -> bool:
    """Check if the given ``value`` can be stored in an HDF5 file."""
    return isinstance(
        value,
        bool | str | bytes | int | float | np.ndarray | list | tuple,
    )


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

    def __init__(self, file_path: Path, attrs: dict[str, Any] | None = None) -> None:
        """Initialize the cache with the given ``file_path``."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path = file_path
        logger.info(f"Initialized HDF5 storage at {file_path}")

        if attrs is not None:
            with h5py.File(self.file_path, "a") as file:
                file.attrs.update(hdf5_dict(attrs))

    def __getitem__(self, key: bytes | str) -> tuple[np.ndarray, dict[str, Any]]:
        """Get the array and attributes stored under the given ``key``."""
        with h5py.File(self.file_path, "r") as file:
            array = file[key][()]
            attrs = dict(file[key].attrs)

        logger.debug(f"Loaded dataset {key} from {self.file_path}")
        return array, attrs

    def __setitem__(
        self,
        key: bytes | str,
        value: tuple[np.ndarray, dict[str, Any]],
    ) -> None:
        """Store the given ``value`` under the given ``key``."""
        array, attrs = value
        with h5py.File(self.file_path, "a") as file:
            if key in file:
                del file[key]
            file[key] = array
            file[key].attrs.update(hdf5_dict(attrs))

        logger.debug(f"Stored dataset {key} in {self.file_path}")

    def __contains__(self, key: bytes | str) -> bool:
        """Check if the given ``key`` is in the cache."""
        with h5py.File(self.file_path, "a") as file:
            return key in file


class HDF5FileStorage:
    """Helper class for storing and loading data from an HDF5 file."""

    def __init__(self, file_path: Path, attrs: dict[str, Any] | None = None) -> None:
        """Initialize the storage at the given ``file_path``."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path = file_path
        logger.info(f"Initialized HDF5 storage at {file_path}")

        if attrs is not None:
            with h5py.File(self.file_path, "a") as file:
                file.attrs.update(hdf5_dict(attrs))

    def _get_dset(self, dset_name: str) -> h5py.Dataset:
        """Get the dataset with ``dset_name``."""
        with h5py.File(self.file_path, "a") as file:
            return file[dset_name]

    def load(self, dset_name: str) -> np.ndarray:
        """Load the dataset with the name ``dset_name``."""
        array = self._get_dset(dset_name)[()]
        logger.debug(f"Loaded dataset {dset_name} from {self.file_path}")
        return array

    def get_attrs(self, dset_name: str) -> dict[str, Any]:
        """Get the attributes of the dataset ``dset_name``."""
        attrs = dict(self._get_dset(dset_name).attrs)
        logger.debug(f"Loaded attributes for dataset {dset_name} from {self.file_path}")
        return attrs

    def save(
        self,
        dset_name: str,
        values: np.ndarray,
    ) -> None:
        """Set the ``values`` for the ``dset_name`` dataset."""
        with h5py.File(self.file_path, "a") as file:
            if dset_name in file:
                del file[dset_name]
            file[dset_name] = values

        logger.debug(f"Stored dataset {dset_name} in {self.file_path}")

    def set_attrs(
        self,
        dset_name: str,
        attrs: dict[str, Any],
    ) -> None:
        """Update the ``attrs`` for the ``dset_name`` dataset."""
        with h5py.File(self.file_path, "a") as file:
            file[dset_name].attrs.update(hdf5_dict(attrs))

        logger.debug(f"Stored attributes for dataset {dset_name} in {self.file_path}")


def reduce_pattern(pattern: dict[str, dict[str, bool]]) -> dict[str, dict[str, bool]]:
    """Reduce a ``pattern`` by removing all entries that are ``None``.

    This way, it should be completely recoverable by the ``complete_pattern`` function
    but be shorter to store.

    Example:
    --------
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
    --------
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
