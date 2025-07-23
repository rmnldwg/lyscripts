"""Small command to fetch the data from a remote using the lydata package."""

from pathlib import Path

import lydata  # noqa: F401
from loguru import logger
from lydata.loader import LyDataset
from pydantic import Field

from lyscripts.cli import assemble_main
from lyscripts.configs import BaseCLI


class FetchCLI(LyDataset, BaseCLI):
    """Fetch a specific dataset from the lyDATA repository."""

    github_token: str | None = Field(
        default=None,
        description=(
            "GitHub token to access private datasets. Can also be provided as "
            "`GITHUB_TOKEN` environment variable."
        ),
    )
    github_user: str | None = Field(
        default=None,
        description=(
            "GitHub user for non-token login. Can also be provided as "
            "`GITHUB_USER` environment variable."
        ),
    )
    github_password: str | None = Field(
        default=None,
        description=(
            "GitHub password for non-token login. Can also be provided as "
            "`GITHUB_PASSWORD` environment variable."
        ),
    )
    output_file: Path = Field(description="The path to save the dataset to.")

    def cli_cmd(self):
        """Execute the ``fetch`` command."""
        logger.enable("lydata")
        logger.debug(self.model_dump_json(indent=2))

        dataset = self.get_dataframe(
            use_github=True,
            token=self.github_token,
            user=self.github_user,
            password=self.github_password,
        )
        dataset.to_csv(self.output_file, index=False)
        logger.success(f"Fetched dataset and saved to {self.output_file}")


if __name__ == "__main__":
    main = assemble_main(settings_cls=FetchCLI, prog_name="fetch")
    main()
