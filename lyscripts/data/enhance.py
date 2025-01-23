"""Enhance the dataset by inferring additional columns from the data.

This is a command-line interface to the
:py:func:`~lydata.utils.infer_and_combine_levels` function.
"""

from typing import Literal

from loguru import logger
from lydata import infer_and_combine_levels
from lydata.utils import ModalityConfig

from lyscripts.cli import assemble_main
from lyscripts.configs import BaseCLI, DataConfig
from lyscripts.data.utils import save_table_to_csv


class EnhanceCLI(BaseCLI):
    """Enhance the dataset by inferring additional columns from the data."""

    input: DataConfig
    modalities: dict[str, ModalityConfig] | None = None
    method: Literal["max_llh", "rank"] = "max_llh"
    sides: list[Literal["ipsi", "contra"]] = ["ipsi", "contra"]
    lnl_subdivisions: dict[str, list[str]] = {
        "I": ["a", "b"],
        "II": ["a", "b"],
        "V": ["a", "b"],
    }
    output_file: str

    def cli_cmd(self) -> None:
        """Infer additional columns from the data and save the enhanced dataset.

        This basically provides a CLI to the
        :py:func:`~lydata.utils.infer_and_combine_levels` function. See its docs for
        more details on what exactly is happening here.
        """
        logger.debug(self.model_dump_json(indent=2))

        data = self.input.load()
        modality_names = list(self.modalities.keys()) if self.modalities else None

        infer_lvls_kwargs = {
            "modalities": modality_names,
            "sides": self.sides,
            "subdivisions": self.lnl_subdivisions,
        }
        enhanced = infer_and_combine_levels(
            dataset=data,
            infer_superlevels_kwargs=infer_lvls_kwargs,
            infer_sublevels_kwargs=infer_lvls_kwargs,
            combine_kwargs={
                "modalities": self.modalities,
                "method": self.method,
            },
        )
        save_table_to_csv(file_path=self.output_file, table=enhanced)


if __name__ == "__main__":
    main = assemble_main(settings_cls=EnhanceCLI, prog_name="enhance")
    main()
