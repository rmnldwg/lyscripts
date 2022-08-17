"""
.. include:: ../README.md

This package provides scripts to the `lynference` repository, where the inference
pipelines live and run regarding the lympahtic progression project.

Since I work on different branches in `lynference` that correspond to different
experiments, it is cumbersome to keep all scripts working and up to date over
there. Hence, I created this package to centralize the developement of the scripts
and focus on their implementation in the pipelines in separate repository.
"""
import argparse

from ._version import version
from .helpers import report

__version__ = version
__description__ = "Package containing scripts used in lynference pipelines"
__author__ = "Roman Ludwig"
__email__ = "roman.ludwig@usz.ch"
__uri__ = "https://github.com/rmnldwg/lyscripts"

# nopycln: file

def exit_lyscripts(args: argparse.Namespace):
    """Exit the cmd line tool"""
    if args.version:
        report.print("lyscripts ", __version__)
    else:
        report.print("No command chosen. Exiting...")
