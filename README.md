# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/lycosystem/lyscripts/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                  |    Stmts |     Miss |   Cover |   Missing |
|-------------------------------------- | -------: | -------: | ------: | --------: |
| src/lyscripts/\_\_init\_\_.py         |       34 |        7 |     79% |57-58, 66-72 |
| src/lyscripts/\_\_main\_\_.py         |        3 |        3 |      0% |       3-6 |
| src/lyscripts/\_version.py            |       13 |        3 |     77% |      8-11 |
| src/lyscripts/cli.py                  |       26 |       12 |     54% |61-67, 81-86 |
| src/lyscripts/compute/\_\_init\_\_.py |        9 |        1 |     89% |        21 |
| src/lyscripts/compute/\_\_main\_\_.py |        5 |        5 |      0% |       3-8 |
| src/lyscripts/compute/posteriors.py   |       46 |       19 |     59% |97-137, 141-142 |
| src/lyscripts/compute/prevalences.py  |       82 |        7 |     91% |59-61, 95-100, 234-235 |
| src/lyscripts/compute/priors.py       |       35 |        2 |     94% |   110-111 |
| src/lyscripts/compute/risks.py        |       51 |       33 |     35% |47-65, 81-135, 139-140 |
| src/lyscripts/compute/utils.py        |      120 |        6 |     95% |95, 146, 177, 188, 240, 250 |
| src/lyscripts/configs.py              |      255 |       27 |     89% |90, 122, 165, 173, 217, 271, 277-278, 286, 472, 506-509, 514, 585, 624-637, 680 |
| src/lyscripts/data/\_\_init\_\_.py    |       13 |        1 |     92% |        45 |
| src/lyscripts/data/\_\_main\_\_.py    |       18 |       18 |      0% |      3-36 |
| src/lyscripts/data/enhance.py         |       24 |        8 |     67% |39-58, 62-63 |
| src/lyscripts/data/fetch.py           |       21 |        7 |     67% |42-52, 56-57 |
| src/lyscripts/data/filter.py          |       49 |       30 |     39% |43-66, 76-94, 98-99 |
| src/lyscripts/data/generate.py        |       39 |        4 |     90% |58, 63, 95-96 |
| src/lyscripts/data/join.py            |       20 |        9 |     55% |60-73, 77-78 |
| src/lyscripts/data/lyproxify.py       |      123 |       67 |     46% |31-34, 39-46, 90-119, 132-142, 173, 250-282, 293-307, 340-341 |
| src/lyscripts/data/split.py           |       30 |       14 |     53% |33-65, 72-73 |
| src/lyscripts/data/utils.py           |        9 |        0 |    100% |           |
| src/lyscripts/decorators.py           |       41 |        4 |     90% | 53-55, 70 |
| src/lyscripts/evaluate.py             |       75 |       57 |     24% |29-35, 43-70, 87, 104-109, 120-141, 146-204, 208-212 |
| src/lyscripts/plots.py                |      166 |       23 |     86% |26-27, 46-47, 56, 94, 99, 104, 185-186, 336, 341, 370-392, 399 |
| src/lyscripts/sample.py               |      136 |       11 |     92% |35-36, 74, 132, 172, 188, 301, 309, 313, 420-421 |
| src/lyscripts/schedule.py             |       29 |       13 |     55% |25-27, 35, 44-45, 73-80, 84-85 |
| src/lyscripts/schema.py               |       21 |        3 |     86% | 60-61, 65 |
| src/lyscripts/utils.py                |       84 |        5 |     94% |25, 141-142, 196-197 |
|                             **TOTAL** | **1577** |  **399** | **75%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/lycosystem/lyscripts/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/lycosystem/lyscripts/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/lycosystem/lyscripts/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/lycosystem/lyscripts/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Flycosystem%2Flyscripts%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/lycosystem/lyscripts/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.