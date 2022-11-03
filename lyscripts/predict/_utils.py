"""
Functions and methods that are shared by the prediction submodules.
"""
from typing import Dict, List, Optional

import numpy as np
from rich.progress import track

from lyscripts.utils import report


def clean_pattern(
    pattern: Optional[Dict[str, Dict[str, bool]]],
    lnls: List[str],
) -> Dict[str, Dict[str, bool]]:
    """
    Make sure the provided involvement `pattern` is correct. For each side of the neck,
    and for each of the `lnls` this should in the end contain `True`, `False` or `None`.
    For example,

    >>> pattern = {"ipsi": {"II": True}}
    >>> lnls = ["II", "III"]
    >>> clean_pattern(pattern, lnls)
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
            else:
                pattern[side][lnl] = bool(pattern[side][lnl])

    return pattern


def rich_enumerate(samples: np.ndarray, description: Optional[str] = None):
    """
    Create a progress bar while enumerating over the given samples.
    """
    enumerated_samples = enumerate(samples)
    if description is None:
        return enumerated_samples

    return track(
        enumerated_samples,
        total=len(samples),
        description=description,
        console=report,
        transient=True,
    )
