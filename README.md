<img src="https://raw.githubusercontent.com/rmnldwg/lyscripts/main/github-social-card.png" alt="social card" style="width:830px;"/>

[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/rmnldwg/lyscripts/blob/main/LICENSE)
[![GitHub repo](https://img.shields.io/badge/rmnldwg%2Flymph-grey.svg?style=flat&logo=github)](https://github.com/rmnldwg/lyscripts)
[![build badge](https://github.com/rmnldwg/lyscripts/actions/workflows/build.yml/badge.svg?style=flat)](https://pypi.org/project/lyscripts/)
[![docs badge](https://github.com/rmnldwg/lyscripts/actions/workflows/docs.yml/badge.svg?style=flat)](https://rmnldwg.github.io/lyscripts/)
[![tests badge](https://github.com/rmnldwg/lyscripts/actions/workflows/tests.yml/badge.svg?style=flat)](https://rmnldwg.github.io/lyscripts/)


## What are these `lyscripts`?

This package provides convenient scripts for performing inference and learning regarding the lymphatic spread of head & neck cancer. Essentially, it provides a *command line interface* (CLI) to the [lymph](https://github.com/rmnldwg/lymph) library.

We are making these "convenience" scripts public, because doing so is one necessary requirement to making our research easily and fully reproducible. There exists another repository, [lynference](https://github.com/rmnldwg/lynference), where we store the pipelines that produce(d) our published results in a persistent way. Head over there to learn more about how to reproduce our work.


## Installation

These scripts can be installed via `pip`:

```bash
pip install lyscripts
```

or installed from source by cloning this repo

```bash
git clone https://github.com/rmnldwg/lyscripts.git
cd lyscripts
pip install .
```


## Usage

This package is intended to be mainly used as a collection of Python scripts that would be scattered throughout my projects, if I didn't bundle them here. Hence, they're mostly command line tools that do basic and repetitive stuff.


### As a command line tool

Simply run

```
lyscripts --help
```

in your terminal and let the output guide you through the functions of the program.

You can also refer to the [documentation] for a written-down version of all these help texts and even more context on how and why to use the provided commands.


### As a library

Head over to the [documentation] for more information on the individual modules, classes, and functions that are implemented in this package.

[documentation]: https://lyscripts.readthedocs.io
