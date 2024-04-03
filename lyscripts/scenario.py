"""
This module implements helpers and classes that help us deal with what we call a
*scenario*. A scenario is a set of parameters that determine how we compute priors,
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
from typing import Any, Literal, TypeVar

import numpy as np
from lymph import types

from lyscripts.utils import optional_bool

ScenarioT = TypeVar("ScenarioT", bound="Scenario")


class Scenario:
    """Class for storing configuration of a scenario.

    This may be used by the :py:mod:`.compute` and :py:mod:`.predict` modules to
    compute priors, posteriors, prevalences, and risks.
    """
    def __init__(
        self,
        t_stages: list[int | str] | None = None,
        t_stages_dist: Iterable[float] | None = None,
        mode: Literal["BN", "HMM"] = "HMM",
        midext: bool | None = None,
        diagnosis: dict[str, dict[str, types.PatternType]] | None = None,
        involvement: dict[str, types.PatternType] | None = None,
        is_uni: bool = False,
        side: str = "ipsi",
    ) -> None:
        """Initialize a scenario.

        If ``t_stages`` is set to ``None``, the scenario will be initialized with the
        default value ``["early"]``.
        """
        self.t_stages = t_stages or ["early"]
        self.t_stages_dist = t_stages_dist
        self.mode = mode
        self.midext = midext
        self._diagnosis = diagnosis or {}
        self._involvement = involvement or {}
        self.is_uni = is_uni
        self.side = side


    @classmethod
    def fields(cls) -> dict[str, Any]:
        """Return a list of fields that may make up a scenario."""
        params = inspect.signature(cls).parameters
        res = {}
        for field, param in params.items():
            if param.default == inspect.Parameter.empty:
                res[field] = None
            else:
                res[field] = param.default
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
        if self._t_stages_dist is None:
            return np.ones(len(self.t_stages)) / len(self.t_stages)

        if len(self._t_stages_dist) != len(self.t_stages):
            new_x = np.linspace(0., 1., len(self.t_stages))
            old_x = np.linspace(0., 1., len(self._t_stages_dist))
            self._t_stages_dist = np.interp(new_x, old_x, self._t_stages_dist)
            self._t_stages_dist /= self._t_stages_dist.sum()

        return np.array(self._t_stages_dist)

    @t_stages_dist.setter
    def t_stages_dist(self, value: Iterable[float]) -> None:
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
            pattern = getattr(namespace, f"{side}_involvement", None) or [None] * len(lnls)
            tmp = {lnl: val for lnl, val in zip(lnls, pattern)}
            getattr(scenario, "involvement")[side] = tmp

            pattern = getattr(namespace, f"{side}_diagnosis", None) or [None] * len(lnls)
            tmp = {lnl: val for lnl, val in zip(lnls, pattern)}
            mod_name = getattr(namespace, "modality", "max_llh")
            getattr(scenario, "diagnosis")[side] = {mod_name: tmp}

        return scenario

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
            kwargs = {
                field: scenario.get(field, value)
                for field, value in cls.fields().items()
            }
            kwargs.update(uni_side_kwargs)
            res.append(cls(**kwargs))

        return res


    def for_side(self, side: Literal["ipsi", "contra"]) -> ScenarioT:
        """Return the side-specific part of the scenario.

        >>> scenario = Scenario(involvement={"ipsi": {"II": True}})
        >>> scenario.involvement
        {'ipsi': {'II': True}}
        >>> scenario.for_side("ipsi").involvement
        {'II': True}
        """
        cls = type(self)
        kwargs = {field: getattr(self, field) for field in cls.fields()}
        kwargs["involvement"] = kwargs["involvement"].get(side, {})
        kwargs["diagnosis"] = kwargs["diagnosis"].get(side, {})
        return cls(**kwargs)


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

        res.update({
            "midext": self.midext,
            "diagnosis": self._diagnosis,
            "side": self.side,
            "is_uni": self.is_uni,
        })

        if for_comp == "risks":
            res["involvement"] = self._involvement

        return res


    @property
    def diagnosis(self) -> dict[str, dict[str, types.PatternType]] | dict[str, types.PatternType]:
        """Get bi- or unilateral diagosis, depending on attrs ``side`` and ``is_uni``."""
        if self.is_uni:
            return self._diagnosis[self.side]

        return self._diagnosis


    @property
    def involvement(self) -> dict[str, types.PatternType] | types.PatternType:
        """Get bi- or unilateral involvement, depending on attrs ``side`` and ``is_uni``."""
        if self.is_uni:
            return self._involvement[self.side]

        return self._involvement


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
                side: self._diagnosis[side][modality]
                for side in ["ipsi", "contra"]
            }

        if self.is_uni:
            return pattern[self.side]

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
        '2cd686a7fbad'
        """
        full_hash = hashlib.md5(str(self.as_dict(for_comp)).encode("utf-8")).hexdigest()
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
        "--t-stages", nargs="+", default=["early"],
        help="T-stages to consider.",
    )
    parser.add_argument(
        "--t-stages-dist", nargs="+", type=float,
        help=(
            "Distribution over T-stages. Prior distribution over hidden states will "
            "be marginalized over T-stages using this distribution."
        )
    )
    parser.add_argument(
        "--mode", choices=["BN", "HMM"], default="HMM",
        help="Mode to use for computing the scenario.",
    )

    if for_comp == "priors":
        return

    parser.add_argument(
        "--midext", type=optional_bool, required=False,
        help=(
            "Use midline extention for computing the scenario. Only used with "
            "midline model."
        ),
    )

    if for_comp in ["posteriors", "risks"]:
        modality_help = (
            "provided along with the diagnosis to compute the posterior distribution "
            "over hidden states."
        )
    else:
        modality_help = (
            "used to compute the prevalence of a diagnosis made with this modality."
        )

    if for_comp == "risks":
        parser.add_argument(
            "--ipsi-involvement", nargs="+", type=optional_bool,
            help="Involvement to compute quantitty for (ipsilateral side).",
        )
        parser.add_argument(
            "--contra-involvement", nargs="+", type=optional_bool,
            help="Involvement to compute quantitty for (contralateral side).",
        )

    if for_comp == "prevalences":
        parser.add_argument(
            "--modality", default="max_llh",
            help="Modality name to compute predicted and observed prevalence for.",
        )

    parser.add_argument(
        "--ipsi-diagnosis", nargs="+", type=optional_bool,
        help="Diagnosis of ipsilateral side.",
    )
    parser.add_argument(
        "--contra-diagnosis", nargs="+", type=optional_bool,
        help="Diagnosis of contralateral side.",
    )


if __name__ == "__main__":
    import doctest
    doctest.testmod()
