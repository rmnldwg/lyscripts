"""Export a JSON schema for lyscripts configuration files."""

import json

from lydata.utils import ModalityConfig
from pydantic import BaseModel, Field

from lyscripts import configs


class SchemaSettings(BaseModel):
    """Settings for generating a JSON schema for lyscripts configuration files."""

    version: int = Field(
        description=(
            "Version of the configuration. Must conform to the major version of the "
            "lyscripts package (can only be 1 at the moment). This is used to avoid "
            "compatibility issues when the configuration format changes."
        ),
        ge=1,
        le=1,
    )
    cross_validation: configs.CrossValidationConfig = None
    data: configs.DataConfig = None
    diagnosis: configs.DiagnosisConfig = None
    distributions: dict[str, configs.DistributionConfig] = {}
    graph: configs.GraphConfig = None
    involvement: configs.InvolvementConfig = None
    modalities: dict[str, ModalityConfig] = {}
    model: configs.ModelConfig = None
    sampling: configs.SamplingConfig = None
    scenarios: list[configs.ScenarioConfig] = []


def main() -> None:
    """Generate a JSON schema for lyscripts configuration files."""
    schema = SchemaSettings.model_json_schema()
    print(json.dumps(schema, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
