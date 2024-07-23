"""Define configuration using pydantic."""

import logging
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from lymph import models
from lymph.types import Model, PatternType
from pydantic import BaseModel, Field
from scipy.special import factorial

from lyscripts.utils import flatten

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


class GraphConfig(BaseModel):
    """Configuration specifying the structure of the model graph.

    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam nec purus.
    """

    tumor: dict[str, list[str]] = Field(
        default={"T": []},
        description="Define the name of the tumor(s) and which LNLs it/they drain to.",
    )
    lnl: dict[str, list[str]] = Field(
        default={},
        description="Define the name of the LNL(s) and which LNLs it/they drain to.",
    )


class DistributionConfig(BaseModel):
    """Configuration defining a distribution over diagnose times."""

    kind: Literal["frozen", "parametric"] = Field(
        default="frozen", description="Parametric distributions may be updated."
    )
    func: FuncNameType = Field(
        default="binomial",
        description="Name of predefined function to use as distribution.",
    )
    params: dict[str, int | float] = Field(
        default={}, description="Parameters to pass to the predefined function."
    )


class ModalityConfig(BaseModel):
    """Define a diagnostic or pathological modality."""

    spec: float = Field(ge=0.5, le=1.0, description="Specificity of the modality.")
    sens: float = Field(ge=0.5, le=1.0, description="Sensitivity of the modality.")
    kind: Literal["clinical", "pathological"] = Field(
        default="clinical",
        description="Clinical modalities cannot detect microscopic disease.",
    )


class ModelConfig(BaseModel):
    """Configuration that defines the setup of a model."""

    class_name: Literal["Unilateral", "Bilateral", "Midline"] = Field(
        default="Unilateral", description="Name of the model class to use."
    )
    constructor: Literal["binary", "trinary"] = Field(
        default="binary",
        description="Trinary models differentiate btw. micro- and macroscopic disease.",
    )
    max_time: int = Field(
        default=10, description="Max. number of time-steps to evolve the model over."
    )
    kwargs: dict[str, Any] = Field(
        default={},
        description="Additional keyword arguments to pass to the model constructor.",
    )


class DataConfig(BaseModel):
    """Config that defines which data to load and how.

    >>> data_data = {"path": "./data.csv", "side": "ipsi"}
    >>> data_config = DataConfig(**data_data)
    >>> data_config == DataConfig(path="./data.csv", side="ipsi")
    True
    """

    path: Path = Field(
        pattern=r".*\.csv",
        description="Path to the data CSV file.",
    )
    side: Literal["ipsi", "contra"] | None = Field(
        default="ipsi",
        description="Side of the neck to load data for.",
    )
    mapping: dict[Literal[0, 1, 2, 3, 4], int | str] = Field(
        default={i: "early" if i <= 2 else "late" for i in range(5)},
        description="Optional mapping of T-stages.",
    )


class InvolvementConfig(BaseModel):
    """Config that defines an ipsi- and contralateral involvement pattern."""

    ipsi: PatternType = Field(
        default={},
        description="Involvement pattern for the ipsilateral side of the neck.",
        examples=[{"II": True, "III": False}],
    )
    contra: PatternType = Field(
        default={},
        description="Involvement pattern for the contralateral side of the neck.",
    )


class DiagnosisConfig(BaseModel):
    """Defines an ipsi- and contralateral diagnosis pattern."""

    ipsi: dict[str, PatternType] = Field(
        default={},
        description="Observed diagnoses by different modalities on the ipsi neck.",
        examples=[{"CT": {"II": True, "III": False}}],
    )
    contra: dict[str, PatternType] = Field(
        default={},
        description="Observed diagnoses by different modalities on the contra neck.",
    )


class ScenarioConfig(BaseModel):
    """Define a scenario for which e.g. prevalences and risks may be computed."""

    t_stages: list[int | str] = Field(
        description="List of T-stages to marginalize over in the scenario.",
        examples=[["early"], [3, 4]],
    )
    t_stages_dist: list[float] = Field(
        default=[1.0],
        description="Distribution over T-stages to use for marginalization.",
        examples=[[1.0], [0.6, 0.4]],
    )
    midext: bool | None = Field(
        default=None,
        description="Whether the patient's tumor extends over the midline.",
    )
    mode: Literal["HMM", "BN"] = Field(
        default="HMM",
        description="Which underlying model architecture to use.",
    )
    involvement: InvolvementConfig = InvolvementConfig()
    diagnosis: DiagnosisConfig = DiagnosisConfig()

    def model_post_init(self, __context: Any) -> None:
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


class SamplingConfig(BaseModel):
    """Configuration for the sampling procedure."""

    seed: int = Field(
        default=42,
        description="Seed for the random number generator.",
    )
    walkers_per_dim: int = Field(
        default=20,
        description="Number of walkers per parameter space dimension.",
    )
    max_burnin: int = Field(
        default=10000,
        description="Maximum number of burn-in steps.",
    )
    check_invterval: int = Field(
        default=50,
        description="Check for convergence each time after this many steps.",
    )
    trust_factor: float = Field(
        default=50.0,
        description=(
            "Trust the autocorrelation time only when it's smaller than this factor "
            "times the length of the chain."
        ),
    )
    relative_thresh: float = Field(
        default=0.05,
        description="Relative threshold for convergence.",
    )
    thin: int = Field(
        default=10, description="How many samples to draw before for saving one."
    )
    num_steps: int = Field(
        default=100,
        description="Number of samples after convergence, regardless of thinning.",
    )


def construct_model(
    model_config: ModelConfig,
    graph_config: GraphConfig,
) -> Model:
    """Construct a model from a ``model_config``.

    The ``dist_map`` should map a name to a function that will be used as distribution
    over diagnosis times.
    """
    cls = getattr(models, model_config.class_name)
    constructor = getattr(cls, model_config.constructor)
    model = constructor(
        graph_dict=flatten(graph_config.model_dump()),
        max_time=model_config.max_time,
        **model_config.kwargs,
    )
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
        model.set_modality(modality, **modality_config.model_dump())
        logger.debug(f"Added modality {modality} to model: {modality_config}")

    logger.info(f"Added {len(modalities)} modalities to model: {model}")
    return model


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
