"""A fusion of all :py:mod:`configs`, allowing the creation of a JSON schema.

This command is not intended to be used by the end user. Rather, it exists such that
the developers and maintainers can create a JSON schema from all the defined
:py:mod:`configs` an store that in the `source code repository`_. Subsequently, the
end user can point their IDE to this schema, hosted on GitHub to provide them with
auto-completion and validation of their YAML configuration files that they feed into
the lyscripts CLIs when they build pipelines or scripts with it.

The `URL for the schema`_ can for example be used in the settings of VS Code like this:

.. code:: json

    {
        "yaml.schemas": {
            "https://raw.githubusercontent.com/rmnldwg/lyscripts/main/schemas/ly.json": "*.ly.yaml"
        },
    }

Which would enable auto-completion and validation for all files with the extension
``.ly.yaml`` in the workspace.

.. _source code repository: https://github.com/rmnldwg/lyscripts
.. _URL for the schema: https://raw.githubusercontent.com/rmnldwg/lyscripts/main/schemas/ly.json
"""  # noqa: E501

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
