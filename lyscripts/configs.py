"""Using `pydantic`_, we define configurations for the package.

Most importantly, these configurations are part of the CLIs that the package provides.
but they also help with programmatically validating and constructing various objects.
Maybe most importantly, the :py:class:`GraphConfig` and :py:class:`ModelConfig` may be
used to precisely and reproducibly define how the function :py:func:`construct_model`
should create lymphatic progression :py:mod:`~lymph.models`.

.. _pydantic: https://docs.pydantic.dev/latest/
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import warnings
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger
from lydata.loader import LyDataset
from lydata.utils import ModalityConfig
from lymph import models
from lymph.types import Model, PatternType
from pydantic import BaseModel, ConfigDict, Field, FilePath
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)
from pydantic_settings.sources import DEFAULT_PATH

from lyscripts.utils import binom_pmf, flatten, load_model_samples, load_patient_data

FuncNameType = Literal["binomial"]


DIST_MAP: dict[FuncNameType, Callable] = {
    "binomial": binom_pmf,
}


class CrossValidationConfig(BaseModel):
    """Configs for splitting a dataset into cross-validation folds."""

    seed: int = Field(
        default=42,
        description="Seed for the random number generator.",
    )
    folds: int = Field(
        default=5,
        description="Number of folds to split the dataset into.",
    )


class DataConfig(BaseModel):
    """Where to load lymphatic progression data from and how to feed it into a model."""

    source: FilePath | LyDataset = Field(
        description=(
            "Either a path to a CSV file or a config that specifies how and where "
            "to fetch the data from."
        )
    )
    side: Literal["ipsi", "contra"] | None = Field(
        default=None,
        description="Side of the neck to load data for. Only for Unilateral models.",
    )
    mapping: dict[Literal[0, 1, 2, 3, 4] | str, int | str] = Field(
        default_factory=lambda: {i: "early" if i <= 2 else "late" for i in range(5)},
        description="Optional mapping of numeric T-stages to model T-stages.",
    )

    def load(self, **get_dataframe_kwargs) -> pd.DataFrame:
        """Load data from path or the :py:class:`~lydata.loader.LyDataset`."""
        if isinstance(self.source, LyDataset):
            return self.source.get_dataframe(**get_dataframe_kwargs)

        return load_patient_data(self.source, **get_dataframe_kwargs)

    def get_load_kwargs(self, **read_csv_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Get kwargs for :py:meth:`~lymph.types.Model.load_patient_data`."""
        return {
            "patient_data": self.load(**(read_csv_kwargs or {})),
            **self.model_dump(exclude={"source"}, exclude_none=True),
        }


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

    def to_involvement(self, modality: str) -> InvolvementConfig:
        """Convert the diagnosis pattern to an involvement pattern for ``modality``."""
        return InvolvementConfig(
            ipsi=self.ipsi.get(modality, {}),
            contra=self.contra.get(modality, {}),
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


class GraphConfig(BaseModel):
    """Specifies how the tumor(s) and LNLs are connected in a DAG."""

    tumor: dict[str, list[str]] = Field(
        description="Define the name of the tumor(s) and which LNLs it/they drain to.",
    )
    lnl: dict[str, list[str]] = Field(
        description="Define the name of the LNL(s) and which LNLs it/they drain to.",
    )


class ModelConfig(BaseModel):
    """Define which of the ``lymph`` models to use and how to set them up."""

    external: FilePath | None = Field(
        default=None,
        description="Path to a Python file that defines a model.",
    )
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


class DeprecatedModelConfig(BaseModel):
    """Model configuration prior to ``lyscripts`` major version 1.

    This is implemented for backwards compatibility. Its sole job is to translate
    the outdated settings format into the new one. Note that the only stuff that needs
    to be translated is the model configuration itself and the distributions for
    marginalization over diagnosis times. The :py:class:`~GraphConfig` is still
    compatible.
    """

    first_binom_prob: float = Field(
        description="Fixed parameter for first binomial dist over diagnosis times.",
        ge=0.0,
        le=1.0,
    )
    max_t: int = Field(
        description="Max. number of time-steps to evolve the model over.",
        gt=0,
    )
    t_stages: list[int | str] = Field(
        description=(
            "List of T-stages to marginalize over in the scenario. The old format "
            "assumed all T-stages except the first one to be parametric. Only binomial "
            "distributions are supported."
        ),
    )
    class_: Literal["Unilateral", "Bilateral", "Midline"] = Field(
        description="Name of the model class. Only binary models are supported.",
        alias="class",
    )
    kwargs: dict[str, Any] = Field(
        default={},
        description="Additional keyword arguments to pass to the model constructor.",
    )

    def model_post_init(self, __context):
        """Issue a deprecation warning."""
        warnings.warn(
            message="The 'DeprecatedModelConfig' is deprecated.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return super().model_post_init(__context)

    def translate(self) -> tuple[ModelConfig, dict[int | str, DistributionConfig]]:
        """Translate the deprecated model config to the new format."""
        old_kwargs = self.kwargs.copy()
        new_kwargs = {}

        if (tumor_spread := old_kwargs.pop("base_symmetric")) is not None:
            new_kwargs["is_symmetric"] = new_kwargs.get("is_symmetric", {})
            new_kwargs["is_symmetric"]["tumor_spread"] = tumor_spread

        if (lnl_spread := old_kwargs.pop("trans_symmetric")) is not None:
            new_kwargs["is_symmetric"] = new_kwargs.get("is_symmetric", {})
            new_kwargs["is_symmetric"]["lnl_spread"] = lnl_spread

        new_kwargs.update(old_kwargs)

        model_config = ModelConfig(
            class_name=self.class_,
            constructor="binary",
            max_time=self.max_t,
            kwargs=new_kwargs,
        )

        distribution_configs = {}
        for i, t_stage in enumerate(self.t_stages):
            distribution_configs[t_stage] = DistributionConfig(
                kind="frozen" if i == 0 else "parametric",
                func="binomial",
                params={"p": self.first_binom_prob},
            )

        return model_config, distribution_configs


class SamplingConfig(BaseModel):
    """Settings to configure the MCMC sampling."""

    file: Path = Field(
        description="Path to HDF5 file store results or load last state."
    )
    history_file: Path | None = Field(
        default=None,
        description="Path to store the burn-in metrics (as CSV file).",
    )
    dataset: str = Field(
        default="mcmc",
        description="Name of the dataset in the HDF5 file.",
    )
    cores: int | None = Field(
        gt=0,
        default=os.cpu_count(),
        description=(
            "Number of cores to use for parallel sampling. If `None`, no parallel "
            "processing is used."
        ),
    )
    seed: int = Field(
        default=42,
        description="Seed for the random number generator.",
    )
    walkers_per_dim: int = Field(
        default=20,
        description="Number of walkers per parameter space dimension.",
    )
    check_interval: int = Field(
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
    num_steps: int | None = Field(
        default=100,
        description=("Number of steps to take in the MCMC sampling."),
    )
    thin_by: int = Field(
        default=10, description="How many samples to draw before for saving one."
    )
    inverse_temp: float = Field(
        default=1.0,
        description=(
            "Inverse temperature for thermodynamic integration. Note that this is not "
            "yet fully implemented."
        ),
    )
    param_names: list[str] = Field(
        default=None,
        description=(
            "If provided, only these parameters will be inferred during model sampling."
        ),
    )

    def load(self, thin: int = 1) -> np.ndarray:
        """Load the samples from the HDF5 file.

        Note that the ``thin`` represents another round of thinning and is usually
        not necessary if the samples were already thinned during the sampling process.
        """
        return load_model_samples(
            file_path=self.file,
            name=self.dataset,
            thin=thin,
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


def _construct_model_from_external(path: Path) -> Model:
    """Construct a model from a Python file."""
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except FileNotFoundError as fnf_err:
        logger.error(f"Could not load model from {path}: {fnf_err}")
        raise fnf_err

    logger.info(f"Loaded model from {path}. This ignores model and graph configs.")
    return module.model


def construct_model(
    model_config: ModelConfig,
    graph_config: GraphConfig,
) -> Model:
    """Construct a model from a ``model_config``.

    The default/expected use of this is to specify a model class from the
    `lymph`_ package and pass the necessary arguments to its constructor.
    However, it is also possible to load a model from an external Python file via the
    ``external`` attribute of the ``model_config`` argument. In this case, a symbol
    with name ``model`` must be defined in the file that is to be loaded.

    .. note::

        No check is performed on the model's compatibility with the command/pipeline
        it is used in. It is assumed the model complies with the
        :py:class:`model type <lymph.types.Model>` specifications of the `lymph`_
        package.

    .. _lymph: https://lymph-model.readthedocs.io/stable/
    """
    if model_config.external is not None:
        return _construct_model_from_external(model_config.external)

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


PathType = Path | str | Sequence[Path | str]


class DynamicYamlConfigSettingsSource(YamlConfigSettingsSource):
    """YAML config source that allows dynamic file path specification.

    This is heavily inspired by `this comment`_ in the discussion on a related issue
    of the `pydantic-settings`_ GitHub repository.

    Essentially, this little hack allows a user to specify a one or multiple YAML files
    from which the CLI should read configurtions. Normally, `pydanitc-settings` only
    allows hard-coding the location of these config files.

    .. _this comment: https://github.com/pydantic/pydantic-settings/issues/259#issuecomment-2549444286
    .. _pydantic-settings: https://github.com/pydantic/pydantic-settings
    """

    def __init__(
        self,
        settings_cls,
        yaml_file: PathType | None = DEFAULT_PATH,
        yaml_file_encoding: str | None = None,
        yaml_file_path_field: str = "configs",
    ) -> None:
        """Allow getting the YAML file path from any key in the current state.

        The argument ``yaml_file_path_field`` should be the :py:class:`BaseSettings`
        field that contains the path(s) to the YAML file(s).
        """
        self.yaml_file_path_field = yaml_file_path_field
        super().__init__(settings_cls, yaml_file, yaml_file_encoding)

    def __call__(self) -> dict[str, Any]:
        """Reload the config files from the paths in the current state."""
        yaml_file_to_reload = self.current_state.get(
            self.yaml_file_path_field, self.yaml_file_path
        )
        logger.debug(f"Reloading YAML files from {yaml_file_to_reload} (if it exists).")
        self.__init__(
            settings_cls=self.settings_cls,
            yaml_file=yaml_file_to_reload,
            yaml_file_encoding=self.yaml_file_encoding,
            yaml_file_path_field=self.yaml_file_path_field,
        )
        return super().__call__()

    def __repr__(self) -> str:
        """Return a string representation of the source."""
        return (
            self.__class__.__name__
            + "("
            + f"yaml_file={self.yaml_file_path!r}, "
            + f"yaml_file_encoding={self.yaml_file_encoding!r}, "
            + f"yaml_file_path_field={self.yaml_file_path_field!r}"
            + ")"
        )


class BaseCLI(BaseSettings):
    """Base settings class for all CLI scripts to inherit from."""

    model_config = ConfigDict(yaml_file="config.yaml", extra="ignore")

    configs: list[Path] = Field(
        default=["config.yaml"],
        description=(
            "Path to the YAML file(s) that contain the configuration(s). Configs from "
            "YAML files may be overwritten by command line arguments. When multiple "
            "files are specified, the configs are merged in the order they are given."
        ),
    )
    version: int = Field(
        description=(
            "Version of the configuration. Must conform to the major version of the "
            "lyscripts package (can only be 1 at the moment). This is used to avoid "
            "compatibility issues when the configuration format changes."
        ),
        ge=1,
        le=1,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Add the dynamic YAML config source to the CLI settings."""
        dynamic_yaml_config_source = DynamicYamlConfigSettingsSource(
            settings_cls=settings_cls,
            yaml_file_path_field="configs",
        )
        logger.debug(f"Created {dynamic_yaml_config_source = }")
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            dynamic_yaml_config_source,
        )
