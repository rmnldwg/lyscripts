"""Enhance the dataset by inferring additional columns from the data.

This is a command-line interface to the methods
:py:meth:`~lydata.accessor.LyDataAccessor.combine` and
:py:meth:`~lydata.accessor.LyDataAccessor.augment` of the
:py:class:`~lydata.accessor.LyDataAccessor` class.
"""

from typing import Literal

from loguru import logger
from lydata.accessor import LyDataFrame
from lydata.utils import ModalityConfig

from lyscripts.cli import assemble_main
from lyscripts.configs import BaseCLI, DataConfig
from lyscripts.data.utils import save_table_to_csv


class EnhanceCLI(BaseCLI):
    """Enhance the dataset by inferring additional columns from the data."""

    input: DataConfig
    modalities: dict[str, ModalityConfig] | None = None
    method: Literal["max_llh", "rank"] = "max_llh"
    lnl_subdivisions: dict[str, list[str]] = {
        "I": ["a", "b"],
        "II": ["a", "b"],
        "V": ["a", "b"],
    }
    output_file: str

    def cli_cmd(self) -> None:
        """Infer additional columns from the data and save the enhanced dataset.

        This basically provides a CLI to the
        :py:func:`~lydata.accessor.LyDataAccessor.augment` function. See its docs for
        more details on what exactly is happening here.
        """
        logger.debug(self.model_dump_json(indent=2))

        data: LyDataFrame = self.input.load()
        data = data.ly.enhance(
            modalities=self.modalities,
            method=self.method,
            subdivisions=self.lnl_subdivisions,
        )
        save_table_to_csv(file_path=self.output_file, table=data)


if __name__ == "__main__":
    main = assemble_main(settings_cls=EnhanceCLI, prog_name="enhance")
    main()
