"""
Create a custom pandas accessor to handle `LyProX`_ style data.

.. _LyProX: https://lyprox.org
"""
from collections.abc import Callable
from typing import Any

import pandas as pd
from lymph import types
from pandas.api.extensions import register_dataframe_accessor


def get_match_idx(
    match_idx,
    pattern: dict[str, bool | None],
    data: pd.DataFrame,
    invert: bool = False,
) -> pd.Series:
    """Get indices of rows in the ``data`` where the diagnosis matches the ``pattern``.

    This uses the ``match_idx`` as a starting point and updates it according to the
    ``pattern``. If ``invert`` is set to ``True``, the function returns the inverted
    indices.

    >>> pattern = {"II": True, "III": None}
    >>> data = pd.DataFrame.from_dict({
    ...     "II":  [True, False],
    ...     "III": [False, False],
    ... })
    >>> get_match_idx(True, pattern, data)
    0     True
    1    False
    Name: II, dtype: bool
    """
    for lnl, involvement in data.items():
        if lnl not in pattern or pattern[lnl] is None:
            continue
        if invert:
            match_idx |= involvement != pattern[lnl]
        else:
            match_idx &= involvement == pattern[lnl]

    return match_idx


@register_dataframe_accessor("ly")
class LyProXAccessor:
    """Custom pandas extension for handling `LyProX`_ data.

    .. _LyProX: https://lyprox.org
    """
    def __init__(self, obj: pd.DataFrame) -> None:
        self._validate(obj)
        self._obj = obj

    @staticmethod
    def _validate(obj: pd.DataFrame) -> None:
        """Check tht some important columns are present in the DataFrame."""
        # make sure columns has three levels
        if not obj.columns.nlevels == 3:
            raise AttributeError("DataFrame must have three levels of columns.")

        for col in ["patient", "tumor"]:
            if col not in obj:
                raise AttributeError(f"DataFrame must have a top-level '{col}' column.")

        LyProXAccessor._validate_tumor(obj["tumor", "1"])

    @staticmethod
    def _validate_tumor(tumor: pd.DataFrame) -> None:
        """Make sure important tumor information is present."""
        for col in ["t_stage"]:
            if col not in tumor:
                raise AttributeError(f"Tumor DataFrame must have a '{col}' column.")

    @property
    def t_stage(self) -> pd.Series:
        """Return the T-stage of the tumor.

        If the T-stage has been mapped, return the mapped value.
        """
        if "t_stage_mapped" in self._obj["tumor", "1"]:
            return self._obj["tumor", "1", "t_stage_mapped"]
        return self._obj["tumor", "1", "t_stage"]

    def map_t_stage(self, mapping: dict[int, Any] | Callable[[int], Any]) -> None:
        """Map the T-stage to a new value using the ``mapping``."""
        mapped = self._obj["tumor", "1", "t_stage"].map(mapping)
        self._obj["tumor", "1", "t_stage_mapped"] = mapped

    @property
    def midext(self) -> pd.Series:
        """Return the midline extension of the tumor."""
        return self._obj["tumor", "1", "extension"]

    def is_midext(self, midext: bool | None) -> pd.Series:
        """Return index where ``midext`` matches the midline extension."""
        if midext is None:
            return pd.Series([True] * len(self._obj))
        return self.midext == midext

    def match(
        self,
        pattern: dict[str, types.PatternType],
        modality: str,
    ) -> pd.Series:
        """Return index where ``pattern`` matches observation by the ``modality``."""
        observation = self._obj[modality]
        match_idx = pd.Series([True] * len(observation))

        for side in ["ipsi", "contra"]:
            if side in pattern:
                match_idx = get_match_idx(match_idx, pattern[side], observation[side])

        return match_idx
