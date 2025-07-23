# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0rc3] - 2025-07-22

### Documentation

- Fix `join` command's example call.

### Features

- Add `data fetch` command. Fixes [#75].

### Change

- Access data only via lydata for compatibility.\
  We have changed the lydata 2nd level headers slightly for the
  patient and tumor info (see
  https://github.com/lycosystem/lydata/issues/21 for more info).\
  Since the lydata package was already updated to be compatible with
  that change, we simply need to route every access the lyscripts
  make to the data through lydata package and hence be compatible
  too.
- Make CLI work with new lydata format.\
  This is also related to https://github.com/lycosystem/lydata/issues/21

## [1.0.0rc2] - 2025-06-26

### Bug Fixes

- Divide by midext prevalence. Fixes [#72].\
  This fixes a bug we reintroduced where we didn't compute observed and
  model prevalence in an analogous and comparable way.

### Documentation

- Fix build badge in README.
- Fix outdated links to rmnldwg.

### Miscellaneous Tasks

- Update pre-commit & ruff rules.

### Testing

- Fix the dataset used for testing prevalences.

### Build

- Switch to `src` layout. Fixes [#74].

### Ci

- Use tests action with coverage.

## [1.0.0rc1] - 2025-05-27

### Bug Fixes

- Use 'fork' start method under MacOS.
- Specify config file encoding for other OSes.
- Cast scenario pattern to bool if possible. Fixes [#70].\
  Since we allow defining a pattern with keywords like `"involved"` or `1`
  instead of only `True`, we also need to make sure that is correctly cast
  to its boolean value for lydata's `C` objects.

### Documentation

- Add warning for Windows & MacOS regarding multiprocess(ing).

### Miscellaneous Tasks

- Add link to changelog.

### Testing

- Ensure observed prevalence is correct. Related [#70].

### Build

- Exclude buggy pydantic-settings. <https://github.com/pydantic/pydantic-settings/issues/605> is present in
version 2.9.0 and 2.9.1 of pydantic-settings. So, these versions must be
excluded.

### Ci

- Update release scripts to use [OIDC](https://docs.pypi.org/trusted-publishers/). This is more secure.

### Remove

- Unnecessary custom help formatter.

## [1.0.0.a7] - 2025-04-15

### Bug Fixes

- Make config -> model -> config round trip test pass.
- Remove `thin_by` factor in wrong place.
- Pass involvement & diagnosis correctly to risks and prevalences.

### Features

- Create Modality config from model.
- Create Graph config from model.
- Create Model config from model.

Note that the three features above come with certain limitations. It is not possible
to export all aspects of a model to a configuration. Especially the distributions
over diagnosis times cannot be converted to a `DistributionConfig`.

### Miscellaneous Tasks

- Fix changelog version link.
- Update ly schema.

### Testing

- Round trip config -> model -> config.

### Merge

- Branch 'main' into 'dev'.
- Branch 'configs-from-model' into 'dev'.\
  Constructing configs from models is only partially possible with the
  current implementation of the `lymph` model.

## [1.0.0.a6] - 2025-03-12

### Bug Fixes

- Better handle midline model. \
  This means disabling the evolution over midline extension. Also, since the new
  version of `lymph-model`, the `midext_prob` parameter is not epected to be the
  first one anymore when passed to `set_params()`.
- Pass only ipsilateral diagnosis to unilateral model.
- Pass diagnose & involvement correctly to models as dict.

### Testing

- Ensure unilateral model receives correct diagnosis.
- Test that diagnosis is used correctly in posteriors.

### Build

- Bump lydata & lymph-model dependency.

## [1.0.0.a5] - 2025-02-05

### Bug Fixes

- Provide rich `console` to compute progress bars.
- Correctly build deprecated models.

### Features

- Enable sampling only named parameter subset.

### Testing

- Check model construction & named params.
- Update integration test config YAML files.
- Add external model symbol check.
- Adding dists works correctly.\
  Previously, it could happen that in a `Bilateral` or `Midline` model the
  individual submodel's distributions where not synced.

### Build

- Bump lymph-model.

### Change

- Require every YAML file to have `version`.
- Better version-related error and docs.
- Make in-/output names more consistent.

### Remove

- Drop CLI required argument `version`.

## [1.0.0.a4] - 2025-01-23

### Bug Fixes

- Update data loading to new lydata API.
- Add sampling config back to sample CLI.
- Finish `data filter` command.
- Correctly log number of excluded pateints in `lyproxify`.
- Allow extra args in CLI cmds.
- Logging during progress bar.

### Documentation

- Deactivate help of removed commands.
- Link only to stable versions.
- Fix intersphinx links.
- Update link to schedule module.
- Configure how pydantic models are displayed.
- Add more info about schema.
- Better explain sampling.
- Add proper info to `cli_cmd()` methods.

### Features

- Add mandatory `version` field to command settings.\
  This will allow to differentiate between old and new configs and create
  the models accordingly.
- Add translation of old to new model configs.
- Add dynamic YAML config source.
- Configure logging nicely.
- Update `data enhance` command.
- Update `data join` command.
- Update `data filter` command.
- Update `data split` command.
- Capture lydata logging output.
- Update YAML schema for CLIs.
- Update `data lyproxify` cmd.
- Allow sampling only specified params.\
  Via a new CLI arg named `param_names` one may restrict the parameters
  sampled to a named subset. In combination with the fact that any Python
  model may be loaded, this results in an enormously flexible sampler.
- Update inv temp `schedule` cmd.
- Allow providing start state to sampling func.

### Testing

- Replace subprocess calls with monkeypatch.\
  This allows for better debugging during test calls.
- Load generated data correctly.\
  The synthetic data for testing already has "early" and "late" as
  T-stages. Thus, the mapping needed to be adapted.

### Build

- pydantic-settings >= 2.7 needed.

### Change

- Make `version` field in command settings required.
- Use pydantic for subcommands.
- Use loguru over default logging.
- Rename `data` field to `input`.
- Use pydantic for plot utils, too.
- Use rich logging handler.

### Refac

- Slightly change CLI inheritances.
- Sort configs alphabetically.
- Make sampling more reusable.

### Remove

- Unused utility functions.
- Plotting scripts except histogram/betaposterp helpers.

## [1.0.0.a3] - 2024-11-15

### Bug Fixes

- (**plot**) Turn off label by passing `None`.
- (**plot**) Don't fail on wrong wrong in `.draw()`.
- Polish dataclass configuration.
- Make pydantic work with subparsers.
- (**data**) Correct argument name of save function.
- (**comp**) Add distributions & fix dir type for priors.
- (**config**) Don't thin samples twice.

### Documentation

- (**configs**) Add to sphinx docs.
- Add intersphinx link to lydata.
- (**config**) Improve `construct_model` docstring.
- Correct copyright year.
- Clean up refs to deleted modules.
- Fix links to documentation in readme.

### Features

- (**config**) Write new methods to assemble model.
- (**plot**) Add func `split_legends()`. Related [#60].\
  This allows the user to separate many plot's labels into a number of
  different legends.
- (**configs**) Add graph config.
- (**sample**) Better config management.\
  The new sample command can merge configurations from multiple YAML files
  as well as the command line. It does a better job at reporting progress
  when the log-level is set to debug and it more consistently stores
  samples and metric logs during burn-in.
- (**sample**) Add JSON schema for params.\
  With this JSON schema one can get auto-completion in most popular
  code editors for the configs of most commands.
- (**config**) Allow loading external model.\
  It is now possible to not only specify a model using a fixed YAML
  schema, but also via a plain Python file that defines a global `model`
  variable.
- (**sample**) Add inverse temperature.\
  With this addition, it is now in principle possible to do thermodynamic
  integration. But it is not yet fully fleshed out for a nice user
  experience.
- Add tiny script to generate JSON schema.\
  The script creates a JSON schema based on all CLI commands' settings.
- (**comp**) Rewrite posteriors command that uses pydantic and joblib
- (**comp**) Rewrite risks command that uses pydantic and joblib
- (**sample**) Add iterations/second column to burnin progress display.
- (**sample**) Show total iterations to sampling progress display.
- (**config**) Allow converting diagnosis to involvement.
- (**compute**) Rewrite prevalence command that uses pydantic and joblib

### Miscellaneous Tasks

- Bump pre-commit hooks.
- Add ruff linting rules.
- Switch to ruff, drop pycln & isort.
- Run ruff & clean up codebase.
- Ignore some ruff rules in tests dir.

### Styling

- More cleanup to satisfy ruff.
- Improve docstrings and code style a bit.

### Testing

- Add some basic testing for config.
- (**data**) Basic integration test for `generate`.
- Add sampling step to integration tests.
- (**config**) Check the external loading feature.
- Extend & unify integration test for priors.

### Build

- Remove upper cap in dependencies.
- Bump lymph-model to 1.2.3.
- Remove `dev` from optional dependencies.\
  This is because a lot of dev tools like ruff, pre-commit, ... are
  installed globally (e.g. with pipx) instead of per venv.
- Bump `lydata` dependency to 0.2.0.

### Change

- (**plot**) Improve beta post & hist. Fixes [60].\
  The histograms and beta posteriors are now better implemented, allowing
  a user to extend the `draw()` function's abilities by adding classes
  similar to `Histogram` and `BetaPosterior`.
- (**plot**) More flexible labels.
- Use pydantic over dacite.
- Switch to pydantic config for sampling (WIP).
- (**compute**) Use pydantic & joblib over dataclasses and custom caching.
- ⚠ **BREAKING** (**compute**) Add priors joblib cache
- ⚠ **BREAKING** (**data**) Replace the `generate` cmd.\
  This was just supposed to be a little script to generate data for an
  integration test, but it turns out that it could just be used to update
  the old `generate` command.
  BREAKING CHANGES: `generate` command is better configurable
- (**config**) Merge sample/sampling configs.
- Use lydata's `ModalityConfig`.\
  Since the [lydata](https://github.com/rmnldwg/lydata) package is
  evolving quickly, I added it as a dependency and moved the first bit of
  code over there.
- Enable use of lydata to load patient data.
- (**comp**) HDF5 file storage more versatile.
- (**sample**) Store history in .tmp file.\
  This serves an indication about whether or not a burn-in sampling round
  has been interrupted. The sampler may then continue from where it left off.
- ⚠ **BREAKING** (**compute**) Update prevalence computation.
- More useful YAML load/merge logging.
- Improve logging of some utilities.

### Remove

- Outdated streamlit app.
- Temporary test file.
- Delete remaining streamlit code.
- ⚠ **BREAKING** Unused HDF5 cache and scenarios.\
  Both these things are superseded by better stuff based on pydantic.

## [1.0.0.a2] - 2024-04-28

### 🚀 Features

- *(sample)* Allow no multiprocessing (0 cores)
- *(scenario)* Add `from_dict` classmethod
- *(plot)* Add `offset` to hist and beta dist

### 🐛 Bug Fixes

- Use correct heaer rows, fixes [#57]
- *(compute)* Observe bilateral prevalence
- *(scenario)* Dataclass with prpoerty issue
- Ensure sides in scenario diagnose/involvement
- *(compute)* Correct obs and pred prevalences

### 📚 Documentation

- *(data)* Refactor lyproxify docstrings
- Update badge to link to RTD, fixes [#53]

### 🧪 Testing

- Fix testing setup & add new histogram test

### Change

- *(data)* Make copy before edit in-place
- *(scenario)* Make scenario dataclass

### Merge

- Branch '57-lyproxify-loads-wrong-number-of-header-rows' into 'dev'

## [1.0.0.a1] - 2024-04-03

### 🚀 Features

- *(data)* Add simple data filter command, fixes [#51]
- Customize log handler for better filename
- *(sample)* Allow custom T-stage mapping
- *(sample)* Allow to load `side` data
- *(utils)* Allow `Unilateral.binary` in params
- *(sample)* Display time elapsed during burnin
- *(predict)* Add cmd to precompute state dists
- *(post)* Start with posterior cmd (WIP)
- *(precompute)* Add `priors` cmd, related to [#54]
- *(precompute)* Work on posterior (WIP)
- *(utils)* Allow keywords in modalities def
- *(post)* Compute for multiple scenarios
- *(predict)* Update prevalences cmd (WIP)
- Add class for storing scenarios
- *(precompute)* Priors from list of scenarios
- *(predict)* Finish prevalences cmd
- *(data)* Implement custom pandas accessor
- *(scenario)* Track laterality as well
- *(compute)* Risk works with lymph v1, too, now

### 🐛 Bug Fixes

- [**breaking**] Use modern lydata cols for t_stage matching
- Wrong lnls in predict prevalences
- [**breaking**] Update prev prediction to new lymph API
- *(sample)* Match T-stage mapping with lymph API
- *(data)* Stop dtype change during `concat`
- *(sample)* Skip 0 iter convergence check
- *(sample)* Missing import
- *(sample)* Only pass `side` to unilateral model
- *(sample)* Display converged message nicely
- *(utils)* Correct default args for `get_chain()`
- Wrong posterior shape
- *(predict)* Even out some bugs
- Don't pycln accessor import
- *(data)* Enhance failed due to copy on write

### 📚 Documentation

- Start with basic sphinx setup
- Start organizing top-level cmds with sphinx
- Include all modules in docs
- Update docstrings to reST format
- Add document files for precompute subcmd
- Allow links to lymph docs
- Fix `temp_schedule` docstring
- Shorten titles
- *(predict)* Update prevalence module docstring
- Refactor docs for new `compute` subcommand
- Fix typos and missing modules

### 🧪 Testing

- *(data)* Ensure new joining works correctly
- *(sample)* Check some sampling methods
- Fix typos in tests
- Update failing tests

### ⚙️ Miscellaneous Tasks

- Add readthedocs config file
- Remove pdoc action
- Update changelog

### Build

- Bump lymph-model to v1.0
- Bump lymph version & add sphinx deps

### Change

- *(sample)* [**breaking**] Start on new sample command (WIP)
- *(sample)* [**breaking**] Reimplement sampling command
- *(precompute)* Use HDF5 cache
- *(precompute)* Make recursive. Related: [#54]
- *(prevs)* Start on updated `prevalences` cmd
- Replace 'diagnose' & bump lymph to 1.2.0
- Simplify scenario handling (WIP)
- *(precompute)* Posteriors only from priors
- *(scenario)* Shorten hash to 6 digits

### Merge

- Branch 'main' into 'dev'
- Branch '51-filter-command' into 'dev'
- Branch '53-use-sphinx-for-documentation' into 'dev'
- Branch '54-add-precompute-commands' into 'dev'

### Refac

- *(sample)* Better progress tracking
- *(precompute)* Comp state dist in own submod
- *(utils)* Move funcs out of `precompute`
- Bundle adding scneario args to parser
- *(compute)* Predict & precompute -> compute

### Remove

- [**breaking**] Midline_ext in create_patient_row for now

## [1.0.0.a0] - 2023-12-20

### Bug Fixes

- Update imports to new lymph version
- Remove references to `clean` command

### Miscellaneous Tasks

- [**breaking**] Lyprox to lymph convert not necessary anymore

### Build

- Bump lymph-model version to `>=1.0.0.a4`
- Bump type hints to Python 3.10

### Change

- [**breaking**] `evaluate`: command does not depend on lymph model anymore
- Simplify `log_state()` decorator
- [**breaking**] Change model initialization in some places
- [**breaking**] Use deorated function name for `log_state()` message

### Refac

- [**breaking**] Deduplicate data loading functions: `load_csv_table()` was removed and `load_data_for_model()` renamed to `load_patient_data()`
- Change function names & remove logger from decorated function calls

### Remove

- [**breaking**] Delete unnecessary functions

## [0.7.3] - 2023-08-29

### Bug Fixes

- **data:** `enhance` command is now deterministic, fixes [#40]
- **plot:** correct color keyword arguments & swap arguments in `save_figure` function, fixes [#45]
- **sample:** use global numpy random state, fixes [#31]

### Maintenance

- fix upper version bound of lymph-model package

### Testing

- **sample:** add test for determinism of sampling, related to [#31]

## [0.7.2] - 2023-07-31

### Bug Fixes

- `enhance`: fix bug introduced in [0.7.1]

## [0.7.1] - 2023-07-31

### Bug Fixes

- `enhance`: negative sublevels don't overwrite superlevels anymore. Fixes [#44].

### Maintenance

- bump pre-commit hooks

## [0.7.0] - 2023-06-26

### Bug Fixes

- add modalities from params in synthetic data generation

### Features

- add extensible & versatile logging decorator
- add `--log-level` option to top-level lyscripts command
- add log-level to `log_state` decorator

### Other

- all commands now use the logging library for status updates/ouputs. This fixes [#2].

## [0.6.9] - 2023-06-21

### Bug Fixes

- change the indentation length in the generated markdown data documentation to 4 spaces. Fixes [#41].

## [0.6.8] - 2023-05-30

### Bug Fixes

- flattening error in `lyproxify`
- more robust lyproxify working again

### Documentation

- add detail to docstring of `lyproxify` func

### Features

- add func to generate md docs from column map
- add two new dict modifying functions

## [0.6.7] - 2023-05-23

### Bug Fixes

- make flatten/unflatten funcs more consistent
- add `max_depth` option for `flatten` function
- bump isort version to avoid error

### Features

- add `unflatten` function

## [0.6.6] - 2022-12-01

### Bug Fixes

- pull another function out of a `rich` context, this time in the `join` command. Related to [#33].

## [0.6.5] - 2022-12-01

### Bug Fixes

- swap arguments in the `save_figure` call of the `corner` command
- pull a function using [`rich`] to report its status out of an enclosing [`rich`] context. This fixes [#33].

## [0.6.4] - 2022-12-01

### Bug Fixes

- `hist_kwargs` now overrides the default plot settings for `Histogram`. This fixes [#30]

### Features

- the `lyscripts sample` command now has an argument `--seed` with the aim of making sampling runs reproducible via a random number generator seed. However, it seems as if the [`emcee`] package does not properly support this as runs using the same seed still produce different results. Related to, but not resolving [#31].

## [0.6.3] - 2022-11-25

### Bug Fixes

- `lyproxify`: apply re-indexing only *after* excluding patients
- fix `SettingWithCopyWarning` during re-indexing in `lyproxify`

## [0.6.2] - 2022-11-25

### Bug Fixes

- `lyproxify` cleans empty header cell names

### Documentation

- update lyproxify's `main` docstring
- improve `report_state` & `exclude_patients` documentation
- update top-level `lyproxify` help in README.md

### Features

- allow muting `report_state` decorator globally for a decorated function, while also allowing to override the verbosity per function call
- allow adding an index column during `lyproxify`
- add options to `lyproxify` for dropping rows and columns before starting transformation of raw data
- the `report_state` decorator can now be configured to exit the program when encountering an unexpected exception

## [0.6.1] - 2022-11-24

### Features

- add new command under `lyscripts data` to preprocess any raw data into a format that can be parsed by [LyProX]. Fixes [#25]

## [0.6.0] - 2022-11-23

### Bug Fixes

- display errors and stop, but don't reraise
- add & update main entry point for script use

### Code Refactoring

- use `lyscripts.utils` consistently across data commands
- use `lyscripts.utils` for `evaluate` script
- pull out method to compare prevalence for one sample
- write modular functions for loading YAML, CSV and HDF5 data
- make `lyscripts data join` command a bit more readable
- further modularize `lyscripts data ...` scripts
- standardize CSV saving process
- start to add `utils` for data commands
- put data commands in separate submodule, fixes [#5] (**BREAKING CHANGE!**)

### Documentation

- expand documentation on data, plot & predict subcommands
- enrich the module documentation of predict scripts
- update docstrings of data commands

### Features

- add YAML scenario output to prevalence app
- working version of prevalence app
- add prevalence plot to app
- allow constructing the `lyscripts.plot.utils.Histogram` and `lyscripts.plot.utils.Posterior` from plain data without HDF5 file (**BREAKING CHANGE!**)
- `lyscripts.temp_schedule` output does not have pretty border anymore, making copy & paste easier
- use generators for risk & prevalence prediction, fixes [#23]
- add more params widgets for prevalence app
- add t_stage, midline_ext, ... to prevalence app
- add `LyScriptsError` for passing up messages
- make smart decorators for status reporting
- implement setup of prevalence app
- start implementing streamlit apps

### Testing

- add GitHub action for tests
- fix missing import for corner doctests
- generally, the module is now partially covered by unit tests

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

## [0.5.10] - 2022-10-13

### Bug Fixes

- pick correct consensus method for enhancement ([#17])
- sample does not crash when `pools` not given ([#16])
- add thinning to convergence sampling, too ([#15])

### Documentation

- fix typos & add favicon to docs

## [0.5.9] - 2022-09-16

### Documentation

- don't use relative path for social card

### Features

- `sample` command has a new optional argument `--pools` with which one can adjust the number of multiprocessing pools used during the sampling procedure. Fixes [#13]

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

## [0.5.7] - 2022-08-29

### Bug Fixes

- fix `enhance`'s issue with varying LNLs across modalities ([#8])

### Features

- add progress bar to `enhance` script

## [0.5.6] - 2022-08-29

### Bug Fixes

- can choose list of defined mods in params. This allows one to choose different lists of modalities for e.g. the `enhance` script and the `sampling` one.

### Documentation

- correct typos in the changed docstrings
- update docstring of changed scripts

## [0.5.5] - 2022-08-25

### Bug Fixes

- clean script was using deprecated lymph.utils. This script has now been incorporated into these scripts.

### Documentation

- update README and add docstrings about `enhance`

### Features

- add enhancement scipt that computes additional diagnostic modalities, combining existing ones.

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

## [0.5.3] - 2022-08-22

<!-- markdownlint-disable-file MD024 -->
[1.0.0rc2]: https://github.com/lycosystem/lyscripts/compare/1.0.0rc1...1.0.0rc2
[1.0.0rc1]: https://github.com/lycosystem/lyscripts/compare/1.0.0.a7...1.0.0rc1
[1.0.0.a7]: https://github.com/lycosystem/lyscripts/compare/1.0.0.a6...1.0.0.a7
[1.0.0.a6]: https://github.com/lycosystem/lyscripts/compare/1.0.0.a5...1.0.0.a6
[1.0.0.a5]: https://github.com/lycosystem/lyscripts/compare/1.0.0.a4...1.0.0.a5
[1.0.0.a4]: https://github.com/lycosystem/lyscripts/compare/1.0.0.a3...1.0.0.a4
[1.0.0.a3]: https://github.com/lycosystem/lyscripts/compare/1.0.0.a2...1.0.0.a3
[1.0.0.a2]: https://github.com/lycosystem/lyscripts/compare/1.0.0.a1...1.0.0.a2
[1.0.0.a1]: https://github.com/lycosystem/lyscripts/compare/1.0.0.a0...1.0.0.a1
[1.0.0.a0]: https://github.com/lycosystem/lyscripts/compare/0.7.3...1.0.0.a0
[0.7.3]: https://github.com/lycosystem/lyscripts/compare/0.7.2...0.7.3
[0.7.2]: https://github.com/lycosystem/lyscripts/compare/0.7.1...0.7.2
[0.7.1]: https://github.com/lycosystem/lyscripts/compare/0.7.0...0.7.1
[0.7.0]: https://github.com/lycosystem/lyscripts/compare/0.6.9...0.7.0
[0.6.9]: https://github.com/lycosystem/lyscripts/compare/0.6.8...0.6.9
[0.6.8]: https://github.com/lycosystem/lyscripts/compare/0.6.7...0.6.8
[0.6.7]: https://github.com/lycosystem/lyscripts/compare/0.6.6...0.6.7
[0.6.6]: https://github.com/lycosystem/lyscripts/compare/0.6.5...0.6.6
[0.6.5]: https://github.com/lycosystem/lyscripts/compare/0.6.4...0.6.5
[0.6.4]: https://github.com/lycosystem/lyscripts/compare/0.6.3...0.6.4
[0.6.3]: https://github.com/lycosystem/lyscripts/compare/0.6.2...0.6.3
[0.6.2]: https://github.com/lycosystem/lyscripts/compare/0.6.1...0.6.2
[0.6.1]: https://github.com/lycosystem/lyscripts/compare/0.6.0...0.6.1
[0.6.0]: https://github.com/lycosystem/lyscripts/compare/0.5.11...0.6.0
[0.5.11]: https://github.com/lycosystem/lyscripts/compare/0.5.10...0.5.11
[0.5.10]: https://github.com/lycosystem/lyscripts/compare/0.5.9...0.5.10
[0.5.9]: https://github.com/lycosystem/lyscripts/compare/0.5.8...0.5.9
[0.5.8]: https://github.com/lycosystem/lyscripts/compare/0.5.7...0.5.8
[0.5.7]: https://github.com/lycosystem/lyscripts/compare/0.5.6...0.5.7
[0.5.6]: https://github.com/lycosystem/lyscripts/compare/0.5.5...0.5.6
[0.5.5]: https://github.com/lycosystem/lyscripts/compare/0.5.4...0.5.5
[0.5.4]: https://github.com/lycosystem/lyscripts/compare/0.5.3...0.5.4
[0.5.3]: https://github.com/lycosystem/lyscripts/compare/0.5.2...0.5.3

[#2]: https://github.com/lycosystem/lyscripts/issues/2
[#5]: https://github.com/lycosystem/lyscripts/issues/5
[#8]: https://github.com/lycosystem/lyscripts/issues/8
[#11]: https://github.com/lycosystem/lyscripts/issues/11
[#13]: https://github.com/lycosystem/lyscripts/issues/13
[#15]: https://github.com/lycosystem/lyscripts/issues/15
[#16]: https://github.com/lycosystem/lyscripts/issues/16
[#17]: https://github.com/lycosystem/lyscripts/issues/17
[#20]: https://github.com/lycosystem/lyscripts/issues/20
[#21]: https://github.com/lycosystem/lyscripts/issues/21
[#23]: https://github.com/lycosystem/lyscripts/issues/23
[#25]: https://github.com/lycosystem/lyscripts/issues/25
[#30]: https://github.com/lycosystem/lyscripts/issues/30
[#31]: https://github.com/lycosystem/lyscripts/issues/31
[#33]: https://github.com/lycosystem/lyscripts/issues/33
[#40]: https://github.com/lycosystem/lyscripts/issues/40
[#41]: https://github.com/lycosystem/lyscripts/issues/41
[#44]: https://github.com/lycosystem/lyscripts/issues/44
[#45]: https://github.com/lycosystem/lyscripts/issues/45
[#51]: https://github.com/lycosystem/lyscripts/issues/51
[#53]: https://github.com/lycosystem/lyscripts/issues/53
[#54]: https://github.com/lycosystem/lyscripts/issues/54
[#57]: https://github.com/lycosystem/lyscripts/issues/57
[#70]: https://github.com/lycosystem/lyscripts/issues/70
[#72]: https://github.com/lycosystem/lyscripts/issues/72
[#74]: https://github.com/lycosystem/lyscripts/issues/74

[`emcee`]: https://emcee.readthedocs.io/en/stable/
[`rich`]: https://rich.readthedocs.io/en/latest/
[`rich_argparse`]: https://github.com/hamdanal/rich_argparse
[LyProX]: https://lyprox.org
