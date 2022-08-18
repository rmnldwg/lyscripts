"""
.. include:: ../README.md
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

def _exit(args: argparse.Namespace):
    """Exit the cmd line tool"""
    if args.version:
        report.print("lyscripts ", __version__)
    else:
        report.print("No command chosen. Exiting...")
