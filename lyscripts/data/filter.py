"""Filter a dataset according to some common criteria.

This is essentially a command line interface to building a
:py:class:`query object <lydata.accessor.Q>` and applying it to the dataset.
"""

from pathlib import Path
from typing import Literal

from loguru import logger
from lydata import Q
from pydantic import Field
from pydantic_settings import CliImplicitFlag

from lyscripts.cli import assemble_main
from lyscripts.configs import BaseCLI, DataConfig
from lyscripts.data.utils import save_table_to_csv


class FilterCLI(BaseCLI):
    """In- or exclude patients where a certain column fulfills a certain condition."""

    input: DataConfig
    include: CliImplicitFlag[bool] = Field(
        False,
        description="Include patients where the condition is met (default: exclude).",
    )
    column: list[str] | str = Field(
        description=(
            "The column to filter by. May be a tuple of three strings, since data "
            "has a three-level header. If it is only one string, the lydata package "
            "tries to map that to a three-level header."
        )
    )
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "in", "contains"] = Field(
        description="The operator to use for comparison."
    )
    value: float | int | str = Field(description="The value to compare against.")
    output_file: Path = Field(description="The path to save the filtered dataset to.")

    def model_post_init(self, __context):
        """Cast to ``float``, if not possible ``int``, if not possible ``str``."""
        if isinstance(self.column, list):
            if len(self.column) == 1:
                self.column = self.column[0]
            elif len(self.column) == 3:
                self.column = tuple(self.column)
            else:
                raise ValueError(
                    "The column attribute must be an iterable of three strings or a "
                    f"single string, but it is {self.column}."
                )

        try:
            self.value = float(self.value)
            return super().model_post_init(__context)
        except ValueError:
            pass

        try:
            self.value = int(self.value)
            return super().model_post_init(__context)
        except ValueError:
            pass

        return super().model_post_init(__context)

    def cli_cmd(self):
        """Execute the ``filter`` command.

        This command uses the :py:class:`~lydata.accessor.Q` objects of the `lydata`_
        library to filter the dataset according to the given criteria.

        .. _lydata: https://lydata.readthedocs.io
        """
        logger.debug(self.model_dump_json(indent=2))

        data = self.input.load()
        query = Q(
            column=self.column,
            operator=self.operator,
            value=self.value,
        )
        logger.debug(f"Created query object: {query}")
        mask = query.execute(data)

        if self.include:
            filtered = data[mask]
            logger.info(f"Keeping {sum(mask)} of {len(data)} patients.")
        else:
            filtered = data[~mask]
            logger.info(f"Excluding {sum(mask)} of {len(data)} patients.")

        save_table_to_csv(file_path=self.output_file, table=filtered)


if __name__ == "__main__":
    main = assemble_main(settings_cls=FilterCLI, prog_name="filter")
    main()
