<a name="0.5.11"></a>
## [0.5.11] - 2022-11-06

### Bug Fixes
- remove useless import & rename prediction `utils`
- histogram & respective posterior have same color
- fix two bugs detected during integration test:
  1. The custom enumerate with optional progress bar did not enumerate
  2. Function checking if midline extension matches did not work for some lymph classes
- fix wrong import in submodules `plot` & `predict`
- correct relative imports & remove unused functions
- fix small inconsistency in script call

### Code Refactoring
- use function for loading YAML in all scripts
- further modularize tasks, e.g. params loading
- rename test modules
- put function saving figures in separate utility
- replace common main tasks with `util` functions
- pull out function to extract model param labels
- make `utils` public and hence documented
- greatly simplify histogram plotting script
- pull shared functions into `_utils`
- update [`rich_argparse`] & add highlighting (fixes [#20])
- outsource rich enumeration of predictions
- make prevalence prediction much cleaner
- risk & prevalence share method to clean pattern

### Documentation
- update help in predict's docstrings

### Features
- write decorators for general tasks
- add nice helper functions to plot (fixes [#21])
- risk & prevalence can use thinned samples
- write neat recursive functions to flatten dictionaries

### Testing
- implement more `utils` tests
- add test to the `save_figure` utility
- add test to new params loader
- add small doctest to `get_size` plot utility
- add checks for plotting utils
- write simple tests for prevalence prediction
- add doctest & pytest for predict `utils`


<a name="0.5.10"></a>
## [0.5.10] - 2022-10-13

### Bug Fixes
- pick correct consensus method for enhancement ([#17])
- sample does not crash when `pools` not given ([#16])
- add thinning to convergence sampling, too ([#15])

### Documentation
- fix typos & add favicon to docs

<a name="0.5.9"></a>
## [0.5.9] - 2022-09-16

### Documentation
- don't use relative path for social card

### Features
- `sample` command has a new optional argument `--pools` with which one can adjust the number of multiprocessing pools used during the sampling procedure. Fixes [#13]

<a name="0.5.8"></a>
## [0.5.8] - 2022-09-12

### Bug Fixes
- The function `get_midline_ext_prob` in the prevalence prediction now
does not throw an error anymore when unilateral data is provided, but
returns `None` instead. Fixes [#11]

### Features
- add entry points to CLI. This enables one to call `lyscripts ...` directly, instead of having to use `python -m lyscripts ...` all the time.

### Documentation
- add social card to README
- remove `python -m` prefix from command usage in docstrings

<a name="0.5.7"></a>
## [0.5.7] - 2022-08-29

### Bug Fixes
- fix `enhance`'s issue with varying LNLs across modalities ([#8])

### Features
- add progress bar to `enhance` script

<a name="0.5.6"></a>
## [0.5.6] - 2022-08-29

### Bug Fixes
- can choose list of defined mods in params. This allows one to choose different lists of modalities for e.g. the `enhance` script and the `sampling` one.

### Documentation
- correct typos in the changed docstrings
- update docstring of changed scripts

<a name="0.5.5"></a>
## [0.5.5] - 2022-08-25

### Bug Fixes
- clean script was using deprecated lymph.utils. This script has now been incorporated into these scripts.

### Documentation
- update README and add docstrings about `enhance`

### Features
- add enhancement scipt that computes additional diagnostic modalities, combining existing ones.

<a name="0.5.4"></a>
## [0.5.4] - 2022-08-24

### Documentation
- add call signature to docs every subcommand's `main()`
- add badges, installation & usage to README
- fix pdoc issue with importing `__main__` files

### Maintenance
- make pyproject.toml look nice on PyPI
- tell git to ignore docs dir
- set up git-chglog for creating changelogs
- add pre-commit hook to check commit msg


<a name="0.5.3"></a>
## [0.5.3] - 2022-08-22

[Unreleased]: https://github.com/rmnldwg/lyscripts/compare/0.5.11...HEAD
[0.5.11]: https://github.com/rmnldwg/lyscripts/compare/0.5.10...0.5.11
[0.5.10]: https://github.com/rmnldwg/lyscripts/compare/0.5.9...0.5.10
[0.5.9]: https://github.com/rmnldwg/lyscripts/compare/0.5.8...0.5.9
[0.5.8]: https://github.com/rmnldwg/lyscripts/compare/0.5.7...0.5.8
[0.5.7]: https://github.com/rmnldwg/lyscripts/compare/0.5.6...0.5.7
[0.5.6]: https://github.com/rmnldwg/lyscripts/compare/0.5.5...0.5.6
[0.5.5]: https://github.com/rmnldwg/lyscripts/compare/0.5.4...0.5.5
[0.5.4]: https://github.com/rmnldwg/lyscripts/compare/0.5.3...0.5.4
[0.5.3]: https://github.com/rmnldwg/lyscripts/compare/0.5.2...0.5.3

[#8]: https://github.com/rmnldwg/lyscripts/issues/8
[#11]: https://github.com/rmnldwg/lyscripts/issues/11
[#13]: https://github.com/rmnldwg/lyscripts/issues/13
[#15]: https://github.com/rmnldwg/lyscripts/issues/15
[#16]: https://github.com/rmnldwg/lyscripts/issues/16
[#17]: https://github.com/rmnldwg/lyscripts/issues/17
[#20]: https://github.com/rmnldwg/lyscripts/issues/20
[#21]: https://github.com/rmnldwg/lyscripts/issues/21

[`rich_argparse`]: https://github.com/hamdanal/rich_argparse
