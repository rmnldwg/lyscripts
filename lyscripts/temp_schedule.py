"""
Generate inverse temperature schedules for thermodynamic integration using various
different methods.
"""
import argparse
from typing import Callable, List, Union

import numpy as np
import yaml
from rich.panel import Panel

from .helpers import report


def tolist(func: Callable) -> Callable:
    """Decorator to make sure the returned value is a list of floats."""
    def inner(*args) -> Union[np.ndarray, List[float]]:
        res = func(*args)
        if isinstance(res, np.ndarray):
            return res.tolist()
        return res
    return inner

@tolist
def geometric_schedule(n: int, *_a) -> np.ndarray:
    """Create a geometric sequence of `n` numbers from 0. to 1."""
    log_seq = np.logspace(0., 1., n)
    shifted_seq = log_seq - 1.
    geom_seq = shifted_seq / 9.
    return geom_seq

@tolist
def linear_schedule(n: int, *_a) -> np.ndarray:
    """Create a linear sequence of `n` numbers from 0. to 1."""
    return np.linspace(0., 1., n)

@tolist
def power_schedule(n: int, power: int, *_a) -> np.ndarray:
    """Create a power sequence of `n` numbers from 0. to 1."""
    lin_seq = np.linspace(0., 1., n)
    power_seq = lin_seq**power
    return power_seq

SCHEDULES = {
    "geometric": geometric_schedule,
    "linear": linear_schedule,
    "power": power_schedule,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=SCHEDULES.keys(), default=list(SCHEDULES.keys())[0],
        help="Choose the method to distribute the inverse temperature."
    )
    parser.add_argument(
        "--num", default=32, type=int,
        help="Number of inverse temperatures in the schedule"
    )
    parser.add_argument(
        "--pow", default=4, type=int,
        help="If a power schedule is chosen, use this as power"
    )

    args = parser.parse_args()

    with report.status(f"Create {args.method} sequence of length {args.num}..."):
        func = SCHEDULES[args.method]
        schedule = func(args.num, args.pow)
        yaml_output = yaml.dump({"temp_schedule": schedule})
        report.success(f"Created {args.method} sequence of length {args.num}")
        report.print(Panel(yaml_output, expand=False))
