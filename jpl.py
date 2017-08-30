"""
Functions for working with JPL GPS time series
"""

import os
import pandas as pd



jpl_dir = os.path.abspath('./jpl')
jpl_data = os.path.abspath('./jpl/data')

def load_timeseries(site):
    ''' julsec = time in seconds past J2000 '''
    df = pd.read_csv(os.path.join(jpl_data, site + '.series'),
                        header=None,
                        names=['decyear','east','north','up',
                        'sig_e', 'sig_n', 'sig_u',
                        'corr_en', 'corr_eu', 'corr_nu',
                        'julsec','year','month','day','hour','min','sec'],
                        delim_whitespace=True,
                        )

    # Convert units from [m] to [mm]
    convert = ['east','north','up','sig_e','sig_n','sig_u','corr_en','corr_eu','corr_nu']
    df[convert] = df[convert]*1e3

    # Convert to python timestamps
    df.index = pd.to_datetime(df[['year','month','day']])

    return df


def download_timeseres(site):
    ''' ftp://sideshow.jpl.nasa.gov/pub/usrs/mbh/point/ '''
    print('todo')
