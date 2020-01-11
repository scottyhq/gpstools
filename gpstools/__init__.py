"""
Package for working with GPS time series
"""

__all__ = ["io", "plot", "analysis", "auxfiles"]

import pkg_resources

# https://github.com/python-poetry/poetry/issues/144#issuecomment-559793020
def get_version():
    try:
        distribution = pkg_resources.get_distribution("gpstools")
    except pkg_resources.DistributionNotFound:
        return "dev"  # or "", or None
        # or try with importib.metadata (py>3.8)
        # or try reading pyproject.toml
    else:
        return distribution.version


__version__ = get_version()


from . import io

# from . import ungl,panga,jpl,sopac
from . import plot
from . import analysis
from . import auxfiles
