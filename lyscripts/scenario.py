"""Helpers and classes to deal with *scenarios*.

A scenario is a set of parameters that determine how we compute priors,
posteriors, prevalences, and risks.

Depending on what we compute exactly, only subsets of the parameters that may make up
a scenario are relevant. For example, when computing priors, only the T-stage (or
distribution over T-stages), as well as the mode (Bayesian network or hidden Markov
model) are relevant. But e.g. posteriors and risks also require us to provide a
diagnosis, given which to compute the quantities of interest.
"""

import argparse
import hashlib
import inspect
from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from typing import Any, Literal, TypeVar

import numpy as np
from lymph import types

from lyscripts.utils import optional_bool


class UninitializedPropertyError(Exception):
    """Raise when a uninitialized property of a dataclass is accessed.

    If a field of a dataclass is also a property, then the dataclass will call the
    property's setter during ``__init__`` with the ``property`` object as the value
    (at least if nothing is provided to the constructor).

    Thus, I will not allow setting a ``property`` as the value and raise this exception
    in the getter when no private attribute is found.
    """


ScenarioT = TypeVar("ScenarioT", bound="Scenario")


@dataclass
class Scenario:
    """Dataclass for storing configuration of a scenario.

    This may be used by the :py:mod:`.compute` and :py:mod:`.predict` modules to
    compute priors, posteriors, prevalences, and risks.
    """

    t_stages: list[int | str] = field(default_factory=lambda: ["early"])
    t_stages_dist: list[float] | np.ndarray
    mode: Literal["BN", "HMM"] = "HMM"
    midext: bool | None = None
    diagnosis: dict[str, dict[str, types.PatternType]]
    involvement: dict[str, types.PatternType]
    is_uni: bool = False
    side: str = "ipsi"

    @staticmethod
    def _defaults(property_name: str) -> Any:
        """Return the default value for a property.

        >>> scenario = Scenario()
        >>> scenario.t_stages_dist
        array([1.])
        >>> scenario.diagnosis
        {'ipsi': {}, 'contra': {}}
        >>> scenario = Scenario(is_uni=True)
        >>> scenario.involvement
        {}
        """
        return {
            "t_stages_dist": np.array([1.0]),
            "involvement": {"ipsi": {}, "contra": {}},
            "diagnosis": {"ipsi": {}, "contra": {}},
        }[property_name]

    def __post_init__(self) -> None:
        """Declate default value of properties.

        >>> scenario = Scenario(t_stages=['a', 'b'], t_stages_dist=[0.1, 0.9])
        >>> scenario.t_stages_dist
        array([0.1, 0.9])
        """
        for field_ in fields(self):
            try:
                _ = getattr(self, field_.name)
            except UninitializedPropertyError:
                default = self._defaults(field_.name)
                setattr(self, field_.name, default)

        if not self.is_uni:
            for side in ["ipsi", "contra"]:
                if side not in self.diagnosis:
                    self.diagnosis[side] = {}
                if side not in self.involvement:
                    self.involvement[side] = {}

    @classmethod
    def fields(cls) -> dict[str, Any]:
        """Return a list of fields that may make up a scenario."""
        params = inspect.signature(cls).parameters
        res = {}
        for f, param in params.items():
            if param.default == inspect.Parameter.empty:
                res[f] = None
            else:
                res[f] = param.default
        return res

    @property
    def t_stages_dist(self) -> np.ndarray:
        """Distribution over T-stages. If not set, return uniform distribution.

        This will also interpolate the distribution if the number of T-stages has
        changed or was wrongly set:

        >>> scenario = Scenario(t_stages=[3,4])
        >>> scenario.t_stages_dist
        array([0.5, 0.5])
        >>> scenario.t_stages = [0,1,2]
        >>> scenario.t_stages_dist = [0.1, 0.2, 0.7]
        >>> scenario.t_stages_dist
        array([0.1, 0.2, 0.7])
        >>> scenario.t_stages = ["early", "late"]
        >>> scenario.t_stages_dist
        array([0.125, 0.875])
        """
        if not hasattr(self, "_t_stages_dist"):
            raise UninitializedPropertyError("t_stages_dist")

        if self._t_stages_dist is None:
            self._t_stages_dist = self._defaults("t_stages_dist")

        if len(self._t_stages_dist) != len(self.t_stages):
            new_x = np.linspace(0.0, 1.0, len(self.t_stages))
            old_x = np.linspace(0.0, 1.0, len(self._t_stages_dist))
            self._t_stages_dist = np.interp(new_x, old_x, self._t_stages_dist)

        if not np.isclose(np.sum(self._t_stages_dist), 1.0):
            self._t_stages_dist /= np.sum(self._t_stages_dist)

        return np.array(self._t_stages_dist)

    @t_stages_dist.setter
    def t_stages_dist(self, value: Iterable[float]) -> None:
        if not isinstance(value, property):
            self._t_stages_dist = value

    @classmethod
    def from_namespace(
        cls,
        namespace: argparse.Namespace,
        lnls: list[str] | None = None,
        is_uni: bool = False,
        side: str = "ipsi",
    ) -> ScenarioT:
        """Create a scenario from an ``argparse`` namespace.

        >>> parser = argparse.ArgumentParser()
        >>> add_scenario_arguments(parser, for_comp="risks")
        >>> args = parser.parse_args([
        ...     "--ipsi-diagnosis", "y", "n",
        ...     "--contra-involvement", "healthy", "involved",
        ... ])
        >>> scenario = Scenario.from_namespace(args, lnls=["II", "III"])
        >>> scenario.diagnosis    # doctest: +NORMALIZE_WHITESPACE
        {'ipsi': {'max_llh': {'II': True, 'III': False}},
         'contra': {'max_llh': {'II': None, 'III': None}}}
        >>> scenario.involvement
        {'ipsi': {'II': None, 'III': None}, 'contra': {'II': False, 'III': True}}
        """
        if lnls is None:
            lnls = []

        kwargs = {
            field: getattr(namespace, field, value)
            for field, value in cls.fields().items()
        }
        kwargs["is_uni"] = is_uni
        kwargs["side"] = side
        scenario = cls(**kwargs)

        for side in ["ipsi", "contra"]:
            pattern = getattr(namespace, f"{side}_involvement", None) or [None] * len(
                lnls
            )
            tmp = dict(zip(lnls, pattern, strict=False))
            scenario._involvement[side] = tmp

            pattern = getattr(namespace, f"{side}_diagnosis", None) or [None] * len(
                lnls
            )
            tmp = dict(zip(lnls, pattern, strict=False))
            mod_name = getattr(namespace, "modality", "max_llh")
            scenario._diagnosis[side] = {mod_name: tmp}

        return scenario

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        **kwargs,
    ) -> ScenarioT:
        """Create a scenario from a dictionary.

        >>> data = {
        ...     "t_stages": ["early"],
        ...     "mode": "BN",
        ...     "diagnosis": {"ipsi": {"max_llh": {"II": True, "III": False}}},
        ... }
        >>> scenario = Scenario.from_dict(data)
        >>> scenario.t_stages, scenario.mode
        (['early'], 'BN')
        >>> scenario.diagnosis
        {'ipsi': {'max_llh': {'II': True, 'III': False}}, 'contra': {}}
        """
        init_kwargs = {
            field: kwargs.get(field, data.get(field, value))
            for field, value in cls.fields().items()
        }
        return cls(**init_kwargs)

    @classmethod
    def list_from_params(
        cls,
        params: dict[str, Any],
        is_uni: bool = False,
        side: str = "ipsi",
    ) -> list[ScenarioT]:
        """Create scenarios from a dictionary of parameters.

        >>> params = {
        ...     "scenarios": [
        ...         {"t_stages": ["early"], "mode": "BN"},
        ...         {"t_stages": ["late"], "mode": "HMM"},
        ...     ]
        ... }
        >>> for scenario in Scenario.list_from_params(params):
        ...     print(scenario.t_stages, scenario.mode)
        ['early'] BN
        ['late'] HMM
        """
        res = []
        uni_side_kwargs = {"is_uni": is_uni, "side": side}
        scenarios = params.get("scenarios", [])
        for scenario in scenarios:
            res.append(cls.from_dict(scenario, **uni_side_kwargs))

        return res

    def as_dict(
        self,
        for_comp: Literal["priors", "posteriors", "prevalences", "risks"],
    ) -> dict[str, Any]:
        """Return dict that may be used as keyword arguments for computing priors."""
        res = {
            "mode": self.mode,
            "t_stages": self.t_stages,
            "t_stages_dist": self.t_stages_dist,
        }
        if for_comp == "priors":
            return res

        res.update(
            {
                "midext": self.midext,
                "diagnosis": self.diagnosis,
                "side": self.side,
                "is_uni": self.is_uni,
            }
        )

        if for_comp == "risks":
            res["involvement"] = self.involvement

        return res

    @property
    def diagnosis(
        self,
    ) -> dict[str, dict[str, types.PatternType]] | dict[str, types.PatternType]:
        """Get bi/uni diagnosis, depending on attrs ``side`` and ``is_uni``."""
        if not hasattr(self, "_diagnosis"):
            raise UninitializedPropertyError("diagnosis")

        if self.is_uni:
            return self._diagnosis[self.side]

        return self._diagnosis

    @diagnosis.setter
    def diagnosis(self, value: dict[str, dict[str, types.PatternType]]) -> None:
        if not isinstance(value, property):
            self._diagnosis = value

    @property
    def involvement(self) -> dict[str, types.PatternType] | types.PatternType:
        """Get bi/uni involvement, depending on attrs ``side`` and ``is_uni``."""
        if not hasattr(self, "_involvement"):
            raise UninitializedPropertyError("involvement")

        if self.is_uni:
            return self._involvement[self.side]

        return self._involvement

    @involvement.setter
    def involvement(self, value: dict[str, types.PatternType]) -> None:
        if not isinstance(value, property):
            self._involvement = value

    def get_pattern(
        self,
        get_from: Literal["involvement", "diagnosis"],
        modality: str,
    ) -> dict[str, Any]:
        """Get an involvement pattern for the given ``modality``."""
        if get_from == "involvement":
            pattern = self._involvement
        else:
            pattern = {
                side: self._diagnosis.get(side, {}).get(modality, {})
                for side in ["ipsi", "contra"]
            }

        if self.is_uni:
            return pattern.get(self.side, {})

        return pattern

    def md5_hash(
        self,
        for_comp: Literal["priors", "posteriors", "prevalences", "risks"],
        length: int = 6,
    ) -> str:
        """Return MD5 hash of the scenario ``for_comp``.

        >>> scenario = Scenario(t_stages=["early"], mode="BN")
        >>> scenario.as_dict("priors")
        {'mode': 'BN', 't_stages': ['early'], 't_stages_dist': array([1.])}
        >>> scenario.md5_hash("priors")
        '49f9cb'
        >>> scenario.md5_hash("posteriors", length=12)
        '1194fd880d47'
        """
        full_hash = hashlib.md5(str(self.as_dict(for_comp)).encode("utf-8")).hexdigest()  # noqa: S324
        return full_hash[:length]


def add_scenario_arguments(
    parser: argparse.ArgumentParser,
    for_comp: Literal["priors", "posteriors", "prevalences", "risks"],
) -> None:
    """Add scenario arguments to an argument parser.

    >>> parser = argparse.ArgumentParser()
    >>> add_scenario_arguments(parser, for_comp="priors")
    >>> args = parser.parse_args(["--t-stages", "early", "--mode", "BN"])
    >>> scenario = Scenario.from_namespace(args)
    >>> scenario.as_dict("priors")
    {'mode': 'BN', 't_stages': ['early'], 't_stages_dist': array([1.])}
    """
    parser.add_argument(
        "--t-stages",
        nargs="+",
        default=["early"],
        help="T-stages to consider.",
    )
    parser.add_argument(
        "--t-stages-dist",
        nargs="+",
        type=float,
        help=(
            "Distribution over T-stages. Prior distribution over hidden states will "
            "be marginalized over T-stages using this distribution."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["BN", "HMM"],
        default="HMM",
        help="Mode to use for computing the scenario.",
    )

    if for_comp == "priors":
        return

    parser.add_argument(
        "--midext",
        type=optional_bool,
        required=False,
        help=(
            "Use midline extension for computing the scenario. Only used with "
            "midline model."
        ),
    )

    if for_comp == "risks":
        parser.add_argument(
            "--ipsi-involvement",
            nargs="+",
            type=optional_bool,
            help="Involvement to compute quantitty for (ipsilateral side).",
        )
        parser.add_argument(
            "--contra-involvement",
            nargs="+",
            type=optional_bool,
            help="Involvement to compute quantitty for (contralateral side).",
        )

    if for_comp == "prevalences":
        parser.add_argument(
            "--modality",
            default="max_llh",
            help="Modality name to compute predicted and observed prevalence for.",
        )

    parser.add_argument(
        "--ipsi-diagnosis",
        nargs="+",
        type=optional_bool,
        help="Diagnosis of ipsilateral side.",
    )
    parser.add_argument(
        "--contra-diagnosis",
        nargs="+",
        type=optional_bool,
        help="Diagnosis of contralateral side.",
    )


if __name__ == "__main__":
    scenario = Scenario(t_stages=["a", "b"], t_stages_dist=[0.2, 0.8])
    import doctest

    doctest.testmod()
