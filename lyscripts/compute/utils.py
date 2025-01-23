"""Utilities for precomputing the priors and posteriors."""

import ast
import functools
from pathlib import Path
from typing import Annotated, Any

import h5py
import numpy as np
from joblib import Memory
from loguru import logger
from pydantic import AfterValidator, BaseModel, Field

from lyscripts.configs import (
    BaseCLI,
    DistributionConfig,
    GraphConfig,
    ModelConfig,
    SamplingConfig,
    ScenarioConfig,
)


class BaseComputeCLI(BaseCLI):
    """Common command line settings for the submodule ``compute``."""

    graph: GraphConfig
    model: ModelConfig = ModelConfig()
    distributions: dict[str, DistributionConfig] = Field(
        default={},
        description=(
            "Mapping of model T-categories to predefined distributions over "
            "diagnose times."
        ),
    )
    cache_dir: Path = Field(
        default=Path.cwd() / ".cache",
        description="Cache directory for storing function calls.",
    )
    scenarios: list[ScenarioConfig] = Field(
        default=[],
        description="List of scenarios to compute risks for.",
    )
    sampling: SamplingConfig


def is_hdf5_compatible(value: Any) -> bool:
    """Check if the given ``value`` can be stored in an HDF5 file."""
    return isinstance(
        value,
        bool | str | bytes | int | float | np.ndarray | list | tuple,
    )


def to_hdf5_attrs(mapping: dict[str, Any]) -> dict[str, str]:
    """Convert ``attrs`` to a dictionary of HDF5 compatible attributes or strings."""
    res = {}
    for key, val in mapping.items():
        if is_hdf5_compatible(val):
            res[key] = val
        else:
            res[key] = str(val)
    return res


def from_hdf5_attrs(mapping: h5py.AttributeManager) -> dict[str, Any]:
    """Convert the HDF5 attributes to a dictionary of Python objects."""
    attrs = {}
    for key, value in mapping.items():
        try:
            attrs[key] = ast.literal_eval(value)
        except ValueError:
            attrs[key] = value
    return attrs


def extract_modalities(diagnosis: dict[str, Any]) -> set[str]:
    """Get the set of modalities used in the ``diagnosis``.

    This is not used in the main apps anymore, but since it may be useful, I keep it.

    >>> diagnosis = {
    ...     "ipsi": {
    ...         "MRI": {"II": True, "III": False},
    ...         "PET": {"II": False, "III": True},
    ...      },
    ...     "contra": {"MRI": {"II": False, "III": None}},
    ... }
    >>> sorted(extract_modalities(diagnosis))
    ['MRI', 'PET']
    """
    modality_set = set()

    if "ipsi" not in diagnosis and "contra" not in diagnosis:
        return modality_set | set(diagnosis.keys())

    for side in ["ipsi", "contra"]:
        if side in diagnosis:
            modality_set |= set(diagnosis[side].keys())

    return modality_set


def ensure_parent_dir(path: Path) -> Path:
    """Create the parent directory of the given ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured parent directory of {path}")
    return path


HasParentPath = Annotated[Path, AfterValidator(ensure_parent_dir)]
"""Type hint for path whose parent dir is created if it doesn't exist."""


class HDF5FileStorage(BaseModel):
    """HDF5 file storage for in- and outputs of computations."""

    file: HasParentPath = Field(
        description="Path to the HDF5 file. Parent directories are created if needed."
    )
    dataset: str | None = Field(
        default=None,
        description=(
            "Name of the dataset in the HDF5 file. Save/load methods can override this."
        ),
    )

    def _get_dataset(self) -> str:
        """Get attribute ``dataset`` or the first dataset in the file.

        >>> from tempfile import TemporaryDirectory
        >>> tmp_path = Path(TemporaryDirectory().name) / "test.hdf5"
        >>> storage = HDF5FileStorage(file=tmp_path)
        >>> rand_data = np.random.rand(100, 100)
        >>> storage.save(values=rand_data, dataset="test")
        >>> np.all(storage.load(dataset="test") == rand_data)
        np.True_
        >>> np.all(storage.load() == rand_data)   # loads first dataset
        np.True_
        >>> some_attrs = {"key": "value"}
        >>> storage.set_attrs(attrs=some_attrs, dataset="test")
        >>> storage.get_attrs(dataset="test")
        {'key': 'value'}
        """
        if self.dataset is not None:
            return self.dataset

        with h5py.File(self.file, "r") as file:
            return next(iter(file.keys()))

    def load(self, dataset: str | None = None) -> np.ndarray:
        """Load the dataset with the name ``dataset``."""
        dataset = dataset or self._get_dataset()

        with h5py.File(self.file, "r") as file:
            array = file[dataset][()]

        logger.debug(f"Loaded dataset {dataset} from {self.file}")
        return array

    def get_attrs(self, dataset: str | None = None) -> dict[str, Any]:
        """Get the attributes of the dataset ``dataset``."""
        dataset = dataset or self._get_dataset()

        with h5py.File(self.file, "r") as file:
            attrs = from_hdf5_attrs(file[dataset].attrs)

        logger.debug(f"Loaded attrs for dataset '{dataset}' from {self.file}")
        return attrs

    def save(self, values: np.ndarray, dataset: str | None = None) -> None:
        """Set the ``values`` for the ``dataset`` dataset."""
        dataset = dataset or self._get_dataset()

        with h5py.File(self.file, "a") as file:
            if dataset in file:
                del file[dataset]
            file[dataset] = values

        logger.debug(f"Stored dataset {dataset} in {self.file}")

    def set_attrs(self, attrs: dict[str, Any], dataset: str | None = None) -> None:
        """Update the ``attrs`` for the ``dataset`` dataset."""
        dataset = dataset or self._get_dataset()

        with h5py.File(self.file, "a") as file:
            if dataset not in file:
                raise ValueError(f"Dataset '{dataset}' not found in {self.file}")
            file[dataset].attrs.update(to_hdf5_attrs(attrs))

        logger.debug(f"Stored attrs {attrs} for dataset '{dataset}' in {self.file}")


def reduce_pattern(pattern: dict[str, dict[str, bool]]) -> dict[str, dict[str, bool]]:
    """Reduce a ``pattern`` by removing all entries that are ``None``.

    This way, it should be completely recoverable by the ``complete_pattern`` function
    but be shorter to store.

    Unused but maybe useful for some cases. Keeping it in here for now.

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

    Unused but maybe useful for some cases. Keeping it in here for now.

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


def get_cached(func: callable, cache_dir: Path) -> callable:
    """Return cached ``func`` with a cache at ``cache_dir``."""
    memory = Memory(location=cache_dir, verbose=0)
    cached_func = memory.cache(func, ignore=["progress_desc"])
    logger.info(f"Initialized cache for {func.__name__} at {cache_dir}")

    @functools.wraps(func)
    def log_cache_info_wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__}({args}, {kwargs})")
        if cached_func.check_call_in_cache(*args, **kwargs):
            logger.info(f"Cache hit for {func.__name__}, returning stored result")
        else:
            logger.info(f"Cache miss for {func.__name__}, computing result")

        result = cached_func(*args, **kwargs)
        logger.debug(f"Computed {result = }")
        return result

    log_cache_info_wrapper._cached_func = cached_func
    return log_cache_info_wrapper
