"""
Functions and methods that are shared by the prediction submodules.
"""
from typing import Dict, List, Optional

import numpy as np
from rich.progress import track

from ..helpers import report


def clean_pattern(
    pattern: Optional[Dict[str, Dict[str, bool]]],
    lnls: List[str],
) -> Dict[str, Dict[str, bool]]:
    """
    Make sure the provided involvement `pattern` is correct. For each side of the neck,
    and for each of the `lnls` this should in the end contain `True`, `False` or `None`.
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
    if description is None:
        return enumerate(samples)
    else:
        return track(
            samples,
            total=len(samples),
            description=description,
            console=report,
            transient=True,
        )
