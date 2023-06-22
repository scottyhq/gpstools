# gpstools

Python routines for working with GPS time series from a variety of processing centers.

![Action Status](https://github.com/scottyhq/gpstools/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/scottyhq/gpstools/branch/master/graph/badge.svg)](https://codecov.io/gh/scottyhq/gpstools)
[![PyPI version](https://badge.fury.io/py/gpstools.svg)](https://badge.fury.io/py/gpstools)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/scottyhq/gpstools/binder?urlpath=lab)

Each processing center distributes data in different formats (typically .csv text files with different column headings). This package loads a variety of those files into a standard pandas dataframe for consistent analysis and comparison.

This is a python package that can be used in scripts via an import statement:

### Basic usage

```
pip install https://github.com/scottyhq/gpstools.git
import gpstools
# See example notebooks in `examples` directory
```



### Development

```
git clone https://github.com/scottyhq/gpstools.git
cd gpstools
conda env create -f environment.yml
conda activate gpstools
pip install -e ".[dev]"
```
