"""
Functions and methods that are shared by the prediction submodules.
"""
from typing import Dict, List, Optional


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

            assert pattern[side][lnl] is None or isinstance(pattern[side][lnl], bool), (
                "Involvement pattern contained wrong type"
            )

    return pattern
