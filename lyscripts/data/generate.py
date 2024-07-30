"""Script to create some test data for the integration tests."""

import argparse
import logging
from pathlib import Path

import numpy as np
from pydantic_settings import BaseSettings, CliSettingsSource

from lyscripts.configs import (
    DistributionConfig,
    GraphConfig,
    ModalityConfig,
    ModelConfig,
    add_dists,
    add_modalities,
    construct_model,
)
from lyscripts.data.utils import save_table_to_csv
from lyscripts.utils import merge_yaml_configs

logger = logging.getLogger(__name__)


class CmdSettings(BaseSettings):
    """Settings for the command-line interface."""

    model: ModelConfig
    graph: GraphConfig
    distributions: dict[str, DistributionConfig]
    t_stages_dist: dict[str, float]
    modalities: dict[str, ModalityConfig]
    params: dict[str, float]
    num_patients: int = 200
    output_file: str
    seed: int = 42

    def model_post_init(self, __context) -> None:
        """Make sure distribution over T-stages is normalized."""
        total = 0.0
        for t_stage in self.distributions:
            if t_stage not in self.t_stages_dist:
                raise ValueError(f"Missing distribution for T-stage {t_stage}.")

            total += self.t_stages_dist[t_stage]

        if not np.isclose(total, 1.0):
            raise ValueError("Sum of T-stage distributions must be 1.")

        return super().model_post_init(__context)


def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """Add an ``ArgumentParser`` to the subparsers action."""
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """Add arguments to a ``subparsers`` instance and run its main function when chosen.

    This is called by the parent module that is called via the command line.
    """
    parser.add_argument(
        "--configs",
        default=[],
        nargs="*",
        help="Path(s) to YAML configuration file(s).",
    )
    parser.set_defaults(
        run_main=main,
        cli_settings_source=CliSettingsSource(
            settings_cls=CmdSettings,
            cli_use_class_docs_for_groups=True,
            root_parser=parser,
        ),
    )


def main(args: argparse.Namespace) -> None:
    """Run main script."""
    yaml_config = merge_yaml_configs(args.configs)
    settings = CmdSettings(
        _cli_settings_source=args.cli_settings_source(parsed_args=args), **yaml_config
    )
    logger.debug(settings.model_dump_json(indent=2))

    model = construct_model(settings.model, settings.graph)
    model = add_dists(model, settings.distributions)
    model = add_modalities(model, settings.modalities)
    model.set_params(**settings.params)
    logger.info(f"Set parameters: {model.get_params(as_dict=True)}")

    synth_data = model.draw_patients(
        num=settings.num_patients,
        stage_dist=list(settings.t_stages_dist.values()),
        seed=settings.seed,
    )
    logger.info(f"Generated synthetic data with shape {synth_data.shape}")

    save_table_to_csv(file_path=settings.output_file, table=synth_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)
