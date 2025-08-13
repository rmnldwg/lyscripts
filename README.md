<img src="https://raw.githubusercontent.com/lycosystem/lyscripts/main/github-social-card.png" alt="social card" style="width:830px;"/>

[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/lycosystem/lyscripts/blob/main/LICENSE)
[![GitHub repo](https://img.shields.io/badge/lycosystem%2Flymph-grey.svg?style=flat&logo=github)](https://github.com/lycosystem/lyscripts)
[![build badge](https://github.com/lycosystem/lyscripts/actions/workflows/release.yml/badge.svg?style=flat)](https://pypi.org/project/lyscripts/)
[![docs badge](https://readthedocs.org/projects/lyscripts/badge/?version=latest)](https://lyscripts.readthedocs.io/en/latest/?badge=latest)
[![tests badge](https://github.com/lycosystem/lyscripts/actions/workflows/tests.yml/badge.svg?style=flat)](https://lyscripts.readthedocs.io/en/latest/?badge=latest)
[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/lycosystem/lyscripts/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/lycosystem/lyscripts/blob/python-coverage-comment-action-data/htmlcov/index.html)

## What are these `lyscripts`?

This package provides convenient scripts for performing inference and learning regarding the lymphatic spread of head & neck cancer. Essentially, it provides a *command line interface* (CLI) to the [lymph](https://github.com/lycosystem/lymph) library and the [lydata](https://github.com/lycosystem/lydata) repository that stores lymphatic progression data.

We are making these "convenience" scripts public, because doing so is one necessary requirement to making our research easily and fully reproducible. There exists another repository, [lynference](https://github.com/lycosystem/lynference), where we stored the pipelines that produced our published results in a persistent way.

## Installation

These scripts can be installed via `pip`:

```bash
pip install lyscripts
```

or installed from source by cloning this repo

```bash
git clone https://github.com/lycosystem/lyscripts.git
cd lyscripts
pip install .
```

## Usage

This package is intended to be mainly used as a collection of Python scripts that would be scattered throughout my projects, if I didn't bundle them here. Hence, they're mostly command line tools that do basic and repetitive stuff.

### As a Command Line Tool

Simply run

```
lyscripts --help
```

in your terminal to display the help text for the main command. It will list all subcommands that are avialable, which you can then also call with `lyscripts <subcommand> --help` to get more information on its use and the available arguments.

For example, one subcommand is `lyscripts data collect`, which will launch a small web server that allows a user to enter patient records on lymphatic involvement in head and neck cancer one row at a time and construct a standardized CSV file from it.

<!--- ### --->

You can also refer to the [documentation] for a written-down version of all these help texts and even more context on how and why to use the provided commands.

### As a Library

Head over to the [documentation] for more information on the individual modules, classes, and functions that are implemented in this package.

[documentation]: https://lyscripts.readthedocs.io

### Configuration YAML Schema

Most of the CLI commands allow passing a list of `--configs` in the form of YAML files. If for a particular CLI argument no value is passed directly, the program looks for the corresponding value in the merged YAML files (if multiple files are provided, later ones may overwrite earlier ones).

For these YAML files we provide a unified schema containing all possible fields that any of the CLIs may accept. It is located at `schemas/ly.json` in this repository. So, one could configure e.g. VS Code to consider this schema for all `*.ly.yaml` files. Here is how that could look like in the JSON settings of VS Code:

```json
{
    "yaml.schemas": {
        "https://raw.githubusercontent.com/lycosystem/lyscripts/main/schemas/ly.json": "*.ly.yaml"
    }
}
```

Subsequently, all files ending in `.ly.yaml` will have helpful autocompletion on the allowed/expected types available.
