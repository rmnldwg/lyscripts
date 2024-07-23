"""Define configuration via dataclasses and dacite."""

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from numbers import Number
from pathlib import Path
from typing import Any, Literal

import dacite
import numpy as np
import pandas as pd
from lymph import models
from lymph.types import GraphDictType, Model, PatternType
from scipy.special import factorial

from lyscripts.utils import load_yaml_params

logger = logging.getLogger(__name__)
FuncNameType = Literal["binomial"]


def binom_pmf(support: list[int] | np.ndarray, p: float = 0.5):
    """Binomial PMF."""
    max_time = len(support) - 1
    if p > 1.0 or p < 0.0:
        raise ValueError("Binomial prob must be btw. 0 and 1")
    q = 1.0 - p
    binom_coeff = factorial(max_time) / (
        factorial(support) * factorial(max_time - support)
    )
    return binom_coeff * p**support * q ** (max_time - support)


DIST_MAP: dict[FuncNameType, Callable] = {
    "binomial": binom_pmf,
}


@dataclass
class DistributionConfig:
    """Configuration defining a distribution over diagnose times."""

    kind: Literal["frozen", "parametric"]
    func: FuncNameType = "binomial"
    params: dict[str, Number] = field(default_factory=dict)

    @classmethod
    def dict_from_params(
        cls: type,
        path: Path,
        key: str = "distributions",
    ) -> dict[str, "DistributionConfig"]:
        """Load a distribution configuration from a YAML file."""
        params = load_yaml_params(path)
        return {key: dacite.from_dict(cls, value) for key, value in params[key].items()}


@dataclass
class ModelConfig:
    """Configuration that defines the setup of a model.

    >>> model_config = dc_from_dict(ModelConfig, {
    ...     "class_name": "Unilateral",
    ...     "constructor": "binary",
    ...     "max_time": 10,
    ... })
    >>> model_config == ModelConfig(
    ...     class_name="Unilateral",
    ...     constructor="binary",
    ...     max_time=10,
    ...     kwargs={},
    ... )
    True
    """

    class_name: Literal["Unilateral", "Bilateral", "Midline"]
    constructor: Literal["binary", "trinary"] = "binary"
    max_time: int = 10
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_params(cls, path: Path) -> "ModelConfig":
        """Load a model configuration from a YAML file."""
        params = load_yaml_params(path)
        return dacite.from_dict(cls, params["model"])


def construct_model(
    config: ModelConfig,
    graph_dict: GraphDictType,
) -> Model:
    """Construct a model from a ``model_config``.

    The ``dist_map`` should map a name to a function that will be used as distribution
    over diagnosis times.
    """
    cls = getattr(models, config.class_name)
    constructor = getattr(cls, config.constructor)
    model = constructor(graph_dict, max_time=config.max_time, **config.kwargs)
    logger.info(f"Constructed model: {model}")
    return model


def add_dists(
    model: Model,
    distributions: dict[str | int, DistributionConfig],
    dist_map: dict[FuncNameType, Callable] | None = None,
    inplace: bool = False,
) -> Model:
    """Construct and add distributions over diagnose times to a ``model``."""
    if not inplace:
        model = deepcopy(model)
        logger.debug("Created deepcopy of model.")

    dist_map = dist_map or DIST_MAP

    for t_stage, dist_config in distributions.items():
        if dist_config.kind == "frozen":
            support = np.arange(model.max_time + 1)
            dist = dist_map[dist_config.func](support, **dist_config.params)
        elif dist_config.kind == "parametric":
            dist = dist_map[dist_config.func]
        else:
            raise ValueError(f"Unknown distribution kind: {dist_config.kind}")

        model.set_distribution(t_stage, dist)
        if dist_config.kind == "parametric" and dist_config.params:
            model.get_distribution(t_stage).set_params(**dist_config.params)

        logger.debug(f"Set {dist_config.kind} distribution for '{t_stage}': {dist}")

    logger.info(f"Added {len(distributions)} distributions to model: {model}")
    return model


@dataclass
class ModalityConfig:
    """Define a diagnostic or pathological modality."""

    spec: float
    sens: float
    kind: Literal["clinical", "pathological"] = "clinical"

    @classmethod
    def dict_from_params(
        cls: type,
        path: Path,
        key: str = "modalities",
    ) -> dict[str, "ModalityConfig"]:
        """Load a modality configuration from a YAML file."""
        params = load_yaml_params(path)
        return {key: dacite.from_dict(cls, value) for key, value in params[key].items()}


def add_modalities(
    model: Model,
    modalities: dict[str, ModalityConfig],
    inplace: bool = False,
) -> Model:
    """Add ``modalities`` to a ``model``."""
    if not inplace:
        model = deepcopy(model)
        logger.debug("Created deepcopy of model.")

    for modality, modality_config in modalities.items():
        model.set_modality(modality, **asdict(modality_config))
        logger.debug(f"Added modality {modality} to model: {modality_config}")

    logger.info(f"Added {len(modalities)} modalities to model: {model}")
    return model


@dataclass
class DataConfig:
    """Config that defines which data to load and how.

    >>> data_config = dc_from_dict(DataConfig, {
    ...     "path": "data/processed.csv",
    ...     "side": "ipsi",
    ... }, config=Config(cast=[Path]))
    >>> data_config == DataConfig(
    ...     path=Path("data/processed.csv"),
    ...     side="ipsi",
    ... )
    True
    """

    path: Path
    side: Literal["ipsi", "contra"] = "ipsi"
    mapping: dict[Literal[0, 1, 2, 3, 4], int | str] = field(
        default_factory=lambda: {
            0: "early",
            1: "early",
            2: "early",
            3: "late",
            4: "late",
        }
    )


def add_data(
    model: Model,
    path: Path,
    side: Literal["ipsi", "contra"],
    mapping: dict[Literal[0, 1, 2, 3, 4], int | str] | None = None,
    inplace: bool = False,
) -> Model:
    """Add data to a ``model``."""
    data = pd.read_csv(path, header=[0, 1, 2])
    logger.debug(f"Loaded data from {path}: Shape: {data.shape}")

    kwargs = {"patient_data": data, "mapping": mapping}
    if isinstance(model, models.Unilateral):
        kwargs["side"] = side

    if not inplace:
        model = deepcopy(model)
        logger.debug("Created deepcopy of model.")

    model.load_patient_data(**kwargs)
    logger.info(f"Added data to model: {model}")
    return model


@dataclass
class InvolvementConfig:
    """Config that defines an ipsi- and contralateral involvement pattern."""

    ipsi: PatternType = field(default_factory=dict)
    contra: PatternType = field(default_factory=dict)


@dataclass
class DiagnosisConfig:
    """Defines an ipsi- and contralateral diagnosis pattern."""

    ipsi: dict[str, PatternType] = field(default_factory=dict)
    contra: dict[str, PatternType] = field(default_factory=dict)


@dataclass
class ScenarioConfig:
    """Define a scenario for which quantities may be computed.

    >>> ScenarioConfig(t_stages=["early", "late"])    # doctest: +NORMALIZE_WHITESPACE
    ScenarioConfig(t_stages=['early', 'late'],
                   t_stages_dist=[0.5, 0.5],
                   midext=None,
                   mode='HMM',
                   involvement=InvolvementConfig(ipsi={}, contra={}),
                   diagnosis=DiagnosisConfig(ipsi={}, contra={}))
    >>> scenario = dc_from_dict(ScenarioConfig, {
    ...     "t_stages": [1, 2, 3, 4],
    ...     "t_stages_dist": [4., 1.],
    ... })
    >>> scenario == ScenarioConfig(
    ...     t_stages=[1, 2, 3, 4],
    ...     t_stages_dist=[0.4, 0.3, 0.2, 0.1],
    ... )
    True
    """

    t_stages: list[int | str]
    t_stages_dist: list[float] = field(default_factory=lambda: [1.0])
    midext: bool | None = None
    mode: Literal["HMM", "BN"] = "HMM"
    involvement: InvolvementConfig = field(default_factory=InvolvementConfig)
    diagnosis: DiagnosisConfig = field(default_factory=DiagnosisConfig)

    def __post_init__(self):
        """Interpolate and normalize the distribution."""
        self.interpolate()
        self.normalize()

    def interpolate(self):
        """Interpolate the distribution to the number of ``t_stages``."""
        if len(self.t_stages) != len(self.t_stages_dist):
            new_x = np.linspace(0.0, 1.0, len(self.t_stages))
            old_x = np.linspace(0.0, 1.0, len(self.t_stages_dist))
            # cast to list to make ``__eq__`` work
            self.t_stages_dist = np.interp(new_x, old_x, self.t_stages_dist).tolist()

    def normalize(self):
        """Normalize the distribution to sum to 1."""
        if not np.isclose(np.sum(self.t_stages_dist), 1.0):
            self.t_stages_dist = (
                np.array(self.t_stages_dist) / np.sum(self.t_stages_dist)
            ).tolist()  # cast to list to make ``__eq__`` work
