r"""Generate inverse temperature schedules for thermodynamic integration.

Thermodynamic integration is quite sensitive to the specific schedule which is used.
I noticed in my models, that within the interval :math:`[0, 0.1]`, the increase in the
expected log-likelihood is very steep. Hence, the inverse temperature :math:`\beta`
must be more densely spaced in the beginning.

This can be achieved by using a power sequence: Generate :math:`n` linearly spaced
points in the interval :math:`[0, 1]` and then transform each point by computing
:math:`\beta_i^k` where :math:`k` could e.g. be 5.
"""

from typing import Literal

import numpy as np
from loguru import logger
from pydantic import Field

from lyscripts.cli import assemble_main
from lyscripts.configs import BaseCLI


def geometric_schedule(num: int, *_a) -> np.ndarray:
    """Create a geometric sequence of ``num`` numbers from 0 to 1."""
    log_seq = np.logspace(0.0, 1.0, num)
    shifted_seq = log_seq - 1.0
    return shifted_seq / 9.0


def linear_schedule(num: int, *_a) -> np.ndarray:
    """Create a linear sequence of ``num`` numbers from 0 to 1.

    Equivalent to the :py:func:`power_schedule` with ``power=1``.
    """
    return np.linspace(0.0, 1.0, num)


def power_schedule(num: int, power: float, *_a) -> np.ndarray:
    """Create a power sequence of ``num`` numbers from 0 to 1.

    This is essentially a :py:func:`linear_schedule` of ``num`` numbers from 0 to 1,
    but each number is raised to the power of ``power``.
    """
    lin_seq = np.linspace(0.0, 1.0, num)
    return lin_seq**power


SCHEDULES = {
    "geometric": geometric_schedule,
    "linear": linear_schedule,
    "power": power_schedule,
}


class ScheduleCLI(BaseCLI):
    """Generate an inverse temperature schedule for thermodynamic integration."""

    method: Literal["geometric", "linear", "power"] = Field(
        default="geometric",
        description="Choose the method to distribute the inverse temperatures.",
    )
    num: int = Field(
        default=32,
        description="Number of inverse temperatures in the schedule.",
    )
    power: float = Field(
        default=4,
        description="If a power schedule is chosen, use this as power.",
    )

    def cli_cmd(self) -> None:
        """Start the ``schedule`` command."""
        logger.debug(self.model_dump_json(indent=2))

        func = SCHEDULES[self.method]
        schedule = func(self.num, self.power)

        for inv_temp in schedule:
            # print is necessary to allow piping the output
            print(inv_temp)  # noqa: T201


if __name__ == "__main__":
    main = assemble_main(settings_cls=ScheduleCLI, prog_name="schedule")
    main()
