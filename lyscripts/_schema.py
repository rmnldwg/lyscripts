"""Export a JSON schema for lyscripts configuration files."""

import json

from lyscripts import sample
from lyscripts.compute import priors


class SchemaSettings(sample.CmdSettings, priors.CmdSettings):
    """Settings for generating a JSON schema for lyscripts configuration files."""

    ...


def main() -> None:
    """Generate a JSON schema for lyscripts configuration files."""
    schema = SchemaSettings.model_json_schema()
    # Different cmds may require different settings, so we remove the required field.
    schema.pop("required")
    print(json.dumps(schema, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
