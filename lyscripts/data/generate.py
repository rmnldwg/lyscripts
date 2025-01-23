"""Script to generate a synthetic dataset.

The generation is done by the :py:meth:`~lymph.models.Unilateral.draw_patients` method
of
the `lymph`_ package, which is why this requires the specification of a model
via the :py:class:`~lyscripts.configs.ModelConfig` class.

.. _lymph: https://lymph-model.readthedocs.io/
"""

import numpy as np
from loguru import logger
from lydata.utils import ModalityConfig
from pydantic import Field

from lyscripts.cli import assemble_main
from lyscripts.configs import (
    BaseCLI,
    DistributionConfig,
    GraphConfig,
    ModelConfig,
    add_dists,
    add_modalities,
    construct_model,
)
from lyscripts.data.utils import save_table_to_csv


class GenerateCLI(BaseCLI):
    """Settings for the command-line interface."""

    graph: GraphConfig
    model: ModelConfig = ModelConfig()
    distributions: dict[str, DistributionConfig] = Field(
        default={},
        description=(
            "Mapping of model T-categories to predefined distributions over "
            "diagnose times."
        ),
    )
    t_stages_dist: dict[str, float] = Field(
        description=(
            "Specify what fraction of generated patients should come from the "
            "respective T-Stage."
        )
    )
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

    def cli_cmd(self) -> None:
        """Run the ``generate`` command.

        Here, the command constructs a model from the settings provided via the
        arguments. It then generates a synthetic dataset using the
        :py:meth:`~lymph.models.Unilateral.draw_patients` from the `lymph`_ package.

        .. _lymph: https://lymph-model.readthedocs.io/
        """
        logger.debug(self.model_dump_json(indent=2))

        model = construct_model(self.model, self.graph)
        model = add_dists(model, self.distributions)
        model = add_modalities(model, self.modalities)
        model.set_params(**self.params)
        logger.info(f"Set parameters: {model.get_params(as_dict=True)}")

        synth_data = model.draw_patients(
            num=self.num_patients,
            stage_dist=list(self.t_stages_dist.values()),
            seed=self.seed,
        )
        logger.info(f"Generated synthetic data with shape {synth_data.shape}")

        save_table_to_csv(file_path=self.output_file, table=synth_data)


if __name__ == "__main__":
    main = assemble_main(settings_cls=GenerateCLI, prog_name="data generate")
    main()
