![social card](https://raw.githubusercontent.com/rmnldwg/lyscripts/main/github-social-card.png)

[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/rmnldwg/lyscripts/blob/main/LICENSE)
[![GitHub repo](https://img.shields.io/badge/rmnldwg%2Flymph-grey.svg?style=flat&logo=github)](https://github.com/rmnldwg/lyscripts)
[![build badge](https://github.com/rmnldwg/lyscripts/actions/workflows/build.yml/badge.svg?style=flat)](https://pypi.org/project/lyscripts/)
[![docs badge](https://github.com/rmnldwg/lyscripts/actions/workflows/docs.yml/badge.svg?style=flat)](https://rmnldwg.github.io/lyscripts/)
[![tests badge](https://github.com/rmnldwg/lyscripts/actions/workflows/tests.yml/badge.svg?style=flat)](https://rmnldwg.github.io/lyscripts/)

## What are these `lyscripts`?

This package provides convenient scripts for performing inference and learning regarding the lymphatic spread of head & neck cancer. Essentially, it provides a *command line interface* (CLI) to the [`lymph`](https://github.com/rmnldwg/lymph) library.

We are making these "convenience" scripts public, because doing so is one necessary requirement to making our research easily and fully reproducible. There exists another repository, [`lynference`](https://github.com/rmnldwg/lynference), where we store the pipelines that produce(d) our published results in a persistent way. Head over there to learn more about how to reproduce our work.

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

After installing the package, run `lyscripts --help` to see the following output:

```
usage: lyscripts [-h] [-v]
                 {generate,join,enhance,clean,split,sample,evaluate,predict,plot,
                  temp_schedule}
                 ...

Utility for performing common tasks w.r.t. the inference and prediction tasks one
can use the `lymph` package for.


POSITIONAL ARGUMENTS
  {generate,join,enhance,clean,split,
   sample,evaluate,predict,plot,
   temp_schedule}

    generate                            Generate synthetic patient data for testing
                                        and validation purposes.

    join                                Join datasets from different sources (but of
                                        the same format) into one.

    enhance                             Enhance a LyProX-style CSV dataset in two ways:

                                        1. Add consensus diagnoses based on all
                                        available modalities using on of two
                                        methods: `max_llh` infers the most likely
                                        true state of involvement given only the
                                        available diagnoses. `rank` uses the
                                        available diagnositc modalities and ranks
                                        them based on their respective sensitivity
                                        and specificity.

                                        2. Complete sub- & super-level fields. This
                                        means that if a dataset reports LNLs IIa and
                                        IIb separately, this script will add the
                                        column for LNL II and fill it with the
                                        correct values. Conversely, if e.g. LNL II
                                        is reported to be healthy, we can assume the
                                        sublevels IIa and IIb would have been
                                        reported as healthy, too.

    clean                               Transform the enhanced lyDATA CSV files into
                                        a format that can be used by the lymph model
                                        using this package's utilities.

    split                               Split the full dataset into cross-validation
                                        folds according to the content of the
                                        params.yaml file.

    sample                              Learn the spread probabilities of the HMM
                                        for lymphatic tumor progression using the
                                        preprocessed data as input and MCMC as
                                        sampling method.

                                        This is the central script performing for
                                        our project on modelling lymphatic spread in
                                        head & neck cancer. We use it for model
                                        comparison via the thermodynamic integration
                                        functionality and use the sampled parameter
                                        estimates for risk predictions. This risk
                                        estimate may in turn some day guide
                                        clinicians to make more objective decisions
                                        with respect to defining the *elective
                                        clinical target volume* (CTV-N) in
                                        radiotherapy.

    evaluate                            Evaluate the performance of the trained
                                        model by computing quantities like the
                                        Bayesian information criterion (BIC) or (if
                                        thermodynamic integration was performed) the
                                        actual evidence (with error) of the model.

    predict                             This module provides functions and scripts
                                        to predict the risk of hidden involvement,
                                        given observed diagnoses, and prevalences of
                                        patterns for diagnostic modalities.

    plot                                Provide various plotting utilities for
                                        displaying results of e.g. the inference or
                                        prediction process.

    temp_schedule                       Generate inverse temperature schedules for
                                        thermodynamic integration using various
                                        different methods.

                                        Thermodynamic integration is quite sensitive
                                        to the specific schedule which is used. I
                                        noticed in my models, that within the
                                        interval $[0, 0.1]$, the increase in the
                                        expected log-likelihood is very steep.
                                        Hence, the inverse temparature $\beta$ must
                                        be more densely spaced in the beginning.

                                        This can be achieved by using a power
                                        sequence: Generate $n$ linearly spaced
                                        points in the interval $[0, 1]$ and then
                                        transform each point by computing
                                        $\beta_i^k$ where $k$ could e.g. be 5.


OPTIONAL ARGUMENTS
  -h, --help                            show this help message and exit
  -v, --version                         Display the version of lyscripts (default: False)
```

Each of the individual subcommands provides a help page like this respectively that detail the positional and optional arguments along with their function.
