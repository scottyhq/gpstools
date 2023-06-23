"""
Functions for working with SOPAC GPS time series
http://sopac-csrc.ucsd.edu

http://sopac-csrc.ucsd.edu/index.php/measures-2/

http://garner.ucsd.edu/pub/timeseries/readme.txt

http://garner.ucsd.edu/pub/measuresESESES_products/Timeseries/Current/Clean_TrendNeuTimeSeries_jpl_20230601/husbCleanTrend.neu.Z
"""

import os
import pandas as pd

from pathlib import Path

import datetime
import pandas as pd
import numpy as np
import os.path
import urllib
from pathlib import Path
import sys
import fsspec
import aiohttp

datadir = os.path.join(sys.modules['gpstools'].DATADIR, 'sopac')
if not os.path.isdir(datadir):
    os.makedirs(datadir)
auxdir = os.path.join(sys.modules['gpstools'].AUXDIR, 'sopac')

URL = "http://garner.ucsd.edu/pub/measuresESESES_products/Timeseries/Current"

def configure_session():
    """
    authenticate to sopac server, return session
    """
    fs = fsspec.filesystem('http',
                           client_kwargs = {'auth': aiohttp.BasicAuth('anonymous', 'anonymous@email.com')})
    fs.stat(URL)
    return fs    

def load_timeseries(site, product='Raw', processor='sopac', fs=None):
    """
    type = 'Raw', 'Clean', 'Filter'
    processor = 'sopac', 'jpl', 'comb'
    Read site timeseries directly from URL
    #Dec Yr   Yr  DayOfYr     N         E        U       N sig    E sig    U sig   CorrNE   CorrNU   CorrEU
    #2001.4562  2001  167     14.23    -17.56     25.14     2.86     3.87     8.21  -0.0348  -0.1213   0.0169
    """
    if not fs:
        fs = configure_session()
    
    # Unfortunately, date changes on these subdirectories!
    subdirs = fs.ls(URL, detail=False)
    baseurl= [x for x in subdirs if product in x if processor in x][0]

    #chaining, read remote zipped data
    url = f'{baseurl}{site.lower()}{product}Trend.neu.Z'
    print(url)
    with fsspec.open(f'zip://*neu::{url}',
                     http=dict(client_kwargs = {'auth': aiohttp.BasicAuth('anonymous', 'anonymous@email.com')}),
                     ) as f:
        
        names = ['decyr','year','dayofyear','north','east','up','sig_n','sig_e','sig_u','corr_en','corr_nu','corr_eu']
        if processor == 'comb':
            names += 'chi_squared'

        df = pd.read_csv(f, 
                         names = names,
                         comment='#', 
                         delim_whitespace=True)

    convert = [
        "east",
        "north",
        "up",
        "sig_e",
        "sig_n",
        "sig_u",
        "corr_en",
        "corr_eu",
        "corr_nu",
    ]
    df[convert] = df[convert] * 1e3
    datestr =  df[['year','dayofyear']].astype(str).agg('-'.join, axis=1)
    df.index = pd.to_datetime(datestr, format="%Y-%j")

    return df