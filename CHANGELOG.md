<a name="unreleased"></a>
## [Unreleased]

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

[Unreleased]: https://github.com/rmnldwg/lyscripts/compare/0.5.8...HEAD
[0.5.8]: https://github.com/rmnldwg/lyscripts/compare/0.5.7...0.5.8
[0.5.7]: https://github.com/rmnldwg/lyscripts/compare/0.5.6...0.5.7
[0.5.6]: https://github.com/rmnldwg/lyscripts/compare/0.5.5...0.5.6
[0.5.5]: https://github.com/rmnldwg/lyscripts/compare/0.5.4...0.5.5
[0.5.4]: https://github.com/rmnldwg/lyscripts/compare/0.5.3...0.5.4
[0.5.3]: https://github.com/rmnldwg/lyscripts/compare/0.5.2...0.5.3

[#8]: https://github.com/rmnldwg/lyscripts/issues/8
[#11]: https://github.com/rmnldwg/lyscripts/issues/11
