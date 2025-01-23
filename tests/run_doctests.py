"""Script to run doctests in the modules of `lyscripts`."""

import doctest

from lyscripts import plots, utils
from lyscripts.compute import prevalences, risks

if __name__ == "__main__":
    doctest.testmod(utils, verbose=True)
    doctest.testmod(plots, verbose=True)
    doctest.testmod(prevalences, verbose=True)
    doctest.testmod(risks, verbose=True)
