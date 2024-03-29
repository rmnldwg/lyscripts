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
from collections.abc import Generator, Iterable
from typing import Any, Literal, TypeVar

import numpy as np
from lymph import types

from lyscripts.utils import optional_bool

ScenarioT = TypeVar("ScenarioT", bound="Scenario")


class Scenario:
    """Class for storing configuration of a scenario.

    This may be used by the :py:mod:`.precompute` and :py:mod:`.predict` modules to
    compute priors, posteriors, prevalences, and risks.
    """
    def __init__(
        self,
        t_stages: list[int | str] | None = None,
        t_stages_dist: Iterable[float] | None = None,
        mode: Literal["BN", "HMM"] = "HMM",
        midext: bool | None = None,
        involvement: dict[str, types.PatternType] | None = None,
        modality: str | None = None,
        diagnosis: dict[str, dict[str, types.PatternType]] | None = None,
    ) -> None:
        """Initialize a scenario.

        If ``t_stages`` is set to ``None``, the scenario will be initialized with the
        default value ``["early"]``.
        """
        self.t_stages = t_stages or ["early"]
        self.t_stages_dist = t_stages_dist
        self.mode = mode
        self.midext = midext
        self.involvement = involvement
        self.modality = modality
        self.diagnosis = diagnosis


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
    def from_namespace(cls, namespace: argparse.Namespace) -> ScenarioT:
        """Create a scenario from an ``argparse`` namespace."""
        kwargs = {
            field: getattr(namespace, field, value)
            for field, value in cls.fields().items()
        }
        return cls(**kwargs)

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Generator[ScenarioT, None, None]:
        """Create scenarios from a dictionary of parameters.

        >>> params = {
        ...     "scenarios": [
        ...         {"t_stages": ["early"], "mode": "BN"},
        ...         {"t_stages": ["late"], "mode": "HMM"},
        ...     ]
        ... }
        >>> for scenario in Scenario.from_params(params):
        ...     print(scenario.t_stages, scenario.mode)
        ['early'] BN
        ['late'] HMM
        """
        scenarios = params.get("scenarios", [])
        for scenario in scenarios:
            kwargs = {field: scenario.get(field) for field in cls.fields()}
            yield cls(**kwargs)


    def for_priors(self) -> dict[str, Any]:
        """Return dict that may be used as keyword arguments for computing priors."""
        return {
            "mode": self.mode,
            "t_stages": self.t_stages,
            "t_stages_dist": self.t_stages_dist,
        }

    def for_posteriors(self, side: Literal["ipsi", "contra"] | None = None) -> dict[str, Any]:
        """Return dict that may be used as keyword arguments for computing posteriors."""
        return {
            "mode": self.mode,
            "t_stages": self.t_stages,
            "t_stages_dist": self.t_stages_dist,
            "midext": self.midext,
            "diagnosis": self.diagnosis.get(side) if side else self.diagnosis,
        }

    def for_prevalences(self, side: Literal["ipsi", "contra"] | None = None) -> dict[str, Any]:
        """Return dict that may be used as keyword arguments for computing prevalences."""
        return {
            "mode": self.mode,
            "t_stages": self.t_stages,
            "t_stages_dist": self.t_stages_dist,
            "midext": self.midext,
            "modality": self.modality,
            "involvement": self.involvement.get(side) if side else self.involvement,
        }

    def for_risks(self, side: Literal["ipsi", "contra"] | None = None) -> dict[str, Any]:
        """Return dict that may be used as keyword arguments for computing risks."""
        return {
            "mode": self.mode,
            "t_stages": self.t_stages,
            "t_stages_dist": self.t_stages_dist,
            "midext": self.midext,
            "involvement": self.involvement.get(side) if side else self.involvement,
            "diagnosis": self.diagnosis.get(side) if side else self.diagnosis,
        }

    def md5_hash(
        self,
        for_comp: Literal["priors", "posteriors", "prevalences", "risks"],
    ) -> str:
        """Return MD5 hash of the scenario ``for_comp``.

        >>> scenario = Scenario(t_stages=["early"], mode="BN")
        >>> scenario.for_priors()
        {'mode': 'BN', 't_stages': ['early'], 't_stages_dist': array([1.])}
        >>> scenario.md5_hash(for_comp="priors")
        '49f9cb2f5c33982395e723d7d9f71f41'
        """
        meth_name = f"for_{for_comp}"
        meth = getattr(self, meth_name)
        return hashlib.md5(str(meth()).encode("utf-8")).hexdigest()


def add_scenario_arguments(
    parser: argparse.ArgumentParser,
    for_comp: Literal["priors", "posteriors", "prevalences", "risks"],
) -> None:
    """Add scenario arguments to an argument parser.

    >>> parser = argparse.ArgumentParser()
    >>> add_scenario_arguments(parser, for_comp="priors")
    >>> args = parser.parse_args(["--t-stages", "early", "--mode", "BN"])
    >>> scenario = Scenario.from_namespace(args)
    >>> scenario.for_priors()
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
        "--midext", action=optional_bool, default=None,
        help=(
            "Use midline extention for computing the scenario. Only used with "
            "midline model."
        ),
    )

    if for_comp in ["posteriors", "risks"]:
        modality_help = (
            "The specificity and sensitivity of the defined modality is provided along "
            "with the diagnosis to compute the posterior distribution over hidden "
            "states."
        )
    else:
        modality_help = (
            "The specificity and sensitivity of the defined modality is provided to "
            "compute the prevalence of observing a specified involvement using this "
            "diagnostic modality."
        )

    parser.add_argument(
        "--modality", type=str, required=False,
        help=(
            "Name of diagnostic modality. Used to look for a defined modality in the "
            "params. " + modality_help
        ),
    )
    parser.add_argument(
        "--spec", type=float, required=False,
        help="Specificity of the diagnostic modality. Overrides value found in params."
    )
    parser.add_argument(
        "--sens", type=float, required=False,
        help="Sensitivity of the diagnostic modality. Overrides value found in params."
    )
    parser.add_argument(
        "--kind", choices=["clinical", "pathological"], required=False,
        help="Kind of diagnostic modality. Overrides value found in params."
    )

    if for_comp in ["posteriors", "risks"]:
        parser.add_argument(
            "--ipsi-diagnosis", nargs="+", type=optional_bool,
            help="Diagnosis of ipsilateral side.",
        )
        parser.add_argument(
            "--contra-diagnosis", nargs="+", type=optional_bool,
            help="Diagnosis of contralateral side.",
        )

    if for_comp in ["prevalences", "risks"]:
        parser.add_argument(
            "--ipsi-involvement", nargs="+", type=optional_bool,
            help="Involvement to compute quantitty for (ipsilateral side).",
        )
        parser.add_argument(
            "--contra-involvement", nargs="+", type=optional_bool,
            help="Involvement to compute quantitty for (contralateral side).",
        )


if __name__ == "__main__":
    import doctest
    doctest.testmod()
