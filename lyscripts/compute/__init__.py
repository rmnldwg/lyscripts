"""Commands to compute prior and posterior state distributions from model samples.

This can in turn speed up the computation of risks and prevalences.
"""

from pydantic_settings import BaseSettings, CliApp, CliSubCommand

from lyscripts.compute import posteriors, prevalences, priors, risks


class ComputeCLI(BaseSettings):
    """Compute priors, posteriors, risks, and prevalences from model samples."""

    priors: CliSubCommand[priors.PriorsCLI]
    posteriors: CliSubCommand[posteriors.PosteriorsCLI]
    risks: CliSubCommand[risks.RisksCLI]
    prevalences: CliSubCommand[prevalences.PrevalencesCLI]

    def cli_cmd(self) -> None:
        """Start the ``compute`` subcommand."""
        CliApp.run_subcommand(self)
