r"""Generate inverse temperature schedules for thermodynamic integration.

Thermodynamic integration is quite sensitive to the specific schedule which is used.
I noticed in my models, that within the interval :math:`[0, 0.1]`, the increase in the
expected log-likelihood is very steep. Hence, the inverse temperature :math:`\beta`
must be more densely spaced in the beginning.

This can be achieved by using a power sequence: Generate :math:`n` linearly spaced
points in the interval :math:`[0, 1]` and then transform each point by computing
:math:`\beta_i^k` where :math:`k` could e.g. be 5.
"""

from loguru import logger

from lyscripts.cli import assemble_main
from lyscripts.configs import BaseCLI, ScheduleConfig


class ScheduleCLI(ScheduleConfig, BaseCLI):
    """Generate an inverse temperature schedule for thermodynamic integration."""

    def cli_cmd(self) -> None:
        """Start the ``schedule`` command."""
        logger.debug(self.model_dump_json(indent=2))

        for inv_temp in self.get_schedule():
            # print is necessary to allow piping the output
            print(inv_temp)  # noqa: T201


if __name__ == "__main__":
    main = assemble_main(settings_cls=ScheduleCLI, prog_name="schedule")
    main()
