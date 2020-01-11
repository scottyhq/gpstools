# gpstools

Python routines for working with GPS time series from a variety of processing centers. 

![Action Status](https://github.com/scottyhq/gpstools/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/scottyhq/gpstools/branch/master/graph/badge.svg)](https://codecov.io/gh/scottyhq/gpstools)
[![PyPI version](https://badge.fury.io/py/gpstools.svg)](https://badge.fury.io/py/gpstools)


Each processing center distributes data in different formats (typically .csv text files with different column headings). This package loads a variety of those files into a standard pandas dataframe for consistent analysis and comparison.

This is a python package that can be used in scripts via an import statement:

```
import gpstools as gps
```

The general structure is:
gpstools
    data
    io
    auxfiles
    analysis
    plot

note to retrigger build
