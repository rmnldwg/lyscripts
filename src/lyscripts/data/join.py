"""Join multiple lymphatic progression datasets into a single dataset."""

from pathlib import Path

import pandas as pd
from pydantic import Field

from lyscripts.cli import assemble_main
from lyscripts.configs import BaseCLI, DataConfig
from lyscripts.data.utils import save_table_to_csv


class JoinCLI(BaseCLI):
    """Join multiple lymphatic progression datasets into a single dataset."""

    inputs: list[DataConfig] = Field(description="The datasets to join.")
    output_file: Path = Field(description="The path to the output dataset.")

    def cli_cmd(self) -> None:
        r"""Start the ``join`` subcommand.

        This will load all datasets specified in the ``inputs`` attribute and
        concatenate them into a single dataset.

        Unfortunately, the use of `pydantic`_ does make this particular command a
        little bit more complicated (but also more powerful): If one simply wants to
        concatenate multiple datasets on disk, the ``inputs`` should be provided like
        this:

        .. code-block:: bash

            lyscripts data join \
            --inputs '{"source": "file1.csv"}' \
            --inputs '{"source": "file2.csv"}' \
            --output-file "joined.csv"

        But it also allows for concatenating datasets fetched directly from the
        `lydata Github repo`_. Due to the rather complex command signature, we
        recommend defining what to concatenate using a YAML file:

        .. code-block:: yaml

            inputs:
              - data.year: 2021
                data.institution: "usz"
                data.subsite: "oropharynx"
              - data.year: 2021
                data.institution: "clb"
                data.subsite: "oropharynx"

        Then, the command will look like this:

        .. code-block:: bash

            lyscripts data join --configs datasets.ly.yaml --output-file joined.csv

        .. _pydantic: https://docs.pydantic.dev/latest/
        .. _lydata Github repo: https://github.com/rmnldwg/lydata
        """
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

        save_table_to_csv(file_path=self.output_file, table=joined)


if __name__ == "__main__":
    main = assemble_main(settings_cls=JoinCLI, prog_name="join")
    main()
