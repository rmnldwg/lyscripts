"""Join multiple lymphatic progression datasets into a single dataset."""

from pathlib import Path

import pandas as pd
from pydantic import Field

from lyscripts.cli import _assemble_main
from lyscripts.configs import BaseCLI, DataConfig
from lyscripts.data.utils import save_table_to_csv


class JoinCLI(BaseCLI):
    """Join multiple lymphatic progression datasets into a single dataset."""

    inputs: list[DataConfig] = Field(description="The datasets to join.")
    output: Path = Field(description="The path to the output dataset.")

    def cli_cmd(self) -> None:
        """Start the ``join`` subcommand."""
        joined = None

        for data_config in self.inputs:
            data = data_config.load()
            if joined is None:
                joined = data
            else:
                joined = pd.concat(
                    [joined, data],
                    axis="index",
                    ignore_index=True,
                )

        save_table_to_csv(file_path=self.output, table=joined)


if __name__ == "__main__":
    main = _assemble_main(settings_cls=JoinCLI, prog_name="join")
    main()
