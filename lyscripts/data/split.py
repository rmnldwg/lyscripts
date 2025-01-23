"""Split a dataset into cross-validation folds based on params.yaml file."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import Field

from lyscripts.cli import assemble_main
from lyscripts.configs import BaseCLI, CrossValidationConfig, DataConfig
from lyscripts.data.utils import save_table_to_csv

warnings.simplefilter(action="ignore", category=FutureWarning)


class SplitCLI(BaseCLI):
    """Split a dataset into cross-validation folds."""

    input: DataConfig
    cross_validation: CrossValidationConfig = CrossValidationConfig()
    output_dir: Path = Field(description="The folder to store the split CSV files in.")

    def cli_cmd(self) -> None:
        """Run the ``split`` subcommand.

        This will load the dataset specified in the ``input`` argument and split it
        into the number of folds specified in the ``cross_validation`` argument. The
        resulting splits will be stored in the folder specified in the ``output_dir``
        argument.
        """
        logger.debug(self.model_dump_json(indent=2))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensure output directory {self.output_dir} exists")

        data = self.input.load()

        shuffled_data = data.sample(
            frac=1.0,
            replace=False,
            random_state=self.cross_validation.seed,
        ).reset_index(drop=True)

        split_datas = np.array_split(
            ary=shuffled_data,
            indices_or_sections=self.cross_validation.folds,
        )
        for fold in range(self.cross_validation.folds):
            _train_datas = [
                split_datas[i] for i in range(self.cross_validation.folds) if i != fold
            ]
            train_data = pd.concat(
                objs=_train_datas,
                axis="index",
                ignore_index=True,
            )
            eval_data = split_datas[fold]

            save_table_to_csv(
                file_path=self.output_dir / f"{fold}_train.csv",
                table=train_data,
            )
            save_table_to_csv(
                file_path=self.output_dir / f"{fold}_eval.csv",
                table=eval_data,
            )


if __name__ == "__main__":
    main = assemble_main(settings_cls=SplitCLI, prog_name="split")
    main()
