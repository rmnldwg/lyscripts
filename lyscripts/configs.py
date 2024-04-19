"""Define configuration via dataclasses and dacite."""
from dataclasses import dataclass, field
from numbers import Number
from pathlib import Path
from typing import Literal

import numpy as np
from dacite import from_dict as dc_from_dict  # nopycln: import
from lymph.types import PatternType


@dataclass
class DistributionConfig:
    """Configuration defining a distribution over diagnose times."""
    kind: Literal["frozen", "parametric"]
    func: Literal["binomial"] = "binomial"
    params: dict[str | int, Number] | None = None


@dataclass
class ModelConfig:
    """Configration that defines the setup of a model.

    >>> model_config = dc_from_dict(ModelConfig, {
    ...     "constructor": "Unilateral",
    ...     "distributions": {
    ...         "early": {"kind": "frozen", "func": "binomial", "params": {"p": 0.3}},
    ...         "late": {"kind": "parametric", "func": "binomial"},
    ...     },
    ...     "max_time": 10,
    ... })
    >>> model_config == ModelConfig(
    ...     constructor="Unilateral",
    ...     distributions={
    ...         "early": DistributionConfig(kind="frozen", func="binomial", params={"p": 0.3}),
    ...         "late": DistributionConfig(kind="parametric", func="binomial"),
    ...     },
    ...     max_time=10,
    ... )
    True
    """
    constructor: Literal[
        "Unilateral", "Unilateral.binary", "Unilateral.trinary",
        "Bilateral", "Bilateral.binary", "Bilateral.trinary",
        "Midline", "Midline.binary", "Midline.trinary",
    ]
    distributions: dict[str, DistributionConfig] = field(default_factory=dict)
    max_time: int = 10


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
    side: Literal["ipsi", "contra"] | None = None
    mapping: dict[Literal[0,1,2,3,4], int | str] = field(default_factory=lambda: {
        0: "early", 1: "early", 2: "early",
        3: "late", 4: "late",
    })


@dataclass
class ModalityConfig:
    """Define a diagnostic or pathological modality."""
    spec: float
    sens: float
    kind: Literal["clinical", "pathological"] = "clinical"


@dataclass
class Scenario:
    """Define a scenario for which quantities may be computed.

    >>> scenario = dc_from_dict(Scenario, {
    ...     "t_stages": [1, 2, 3],
    ...     "t_stages_dist": [0.2, 0.5, 0.3],
    ... })
    >>> scenario == Scenario(t_stages=[1, 2, 3], t_stages_dist=[0.2, 0.5, 0.3])
    True
    """
    t_stages: list[int | str]
    t_stages_dist: list[float]
    midext: bool | None = None
    mode: Literal["HMM", "BN"] = "HMM"
    involvement: dict[Literal["ipsi", "contra"], PatternType] = field(
        default_factory=lambda: {"ipsi": {}, "contra": {}},
    )
    diagnosis: dict[Literal["ipsi", "contra"], dict[str, PatternType]] = field(
        default_factory=lambda: {"ipsi": {}, "contra": {}},
    )

    def get_t_stages_dist(self) -> list[float] | np.ndarray:
        """Safely return the distribution of T stages.

        This includes interpolating the distribution to the number of T-stages stored
        in ``t_stages`` and normalizing it to sum to 1.
        """
        if len(self.t_stages) != len(self.t_stages_dist):
            new_x = np.linspace(0., 1., len(self.t_stages))
            old_x = np.linspace(0., 1., len(self.t_stages_dist))
            self.t_stages_dist = np.interp(new_x, old_x, self.t_stages_dist)

        if not np.isclose(np.sum(self.t_stages_dist), 1.):
            self.t_stages_dist = np.array(self.t_stages_dist) / np.sum(self.t_stages_dist)

        return self.t_stages_dist
