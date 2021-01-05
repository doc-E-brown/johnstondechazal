"""Top-level package for johnstondechazal."""

__author__ = """Ben Johnston"""
__email__ = 'ben.johnston@sydney.edu.au'

from johnstondechazal._version import get_versions

__version__ = get_versions()['version']
del get_versions

from .groundtruth import FindGrouthTruth
