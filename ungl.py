# -*- coding: utf-8 -*-
"""
Functions for working with GPS data

Created on Fri Apr  1 14:52:00 2016

@author: scott
"""

import pandas as pd
import numpy as np
import os
import urllib

import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.signal import sawtooth
from scipy import stats

# Need to set this environment variable for each computer
ungl_data = os.environ['UNGL_DATA']

# ---------------------------------------------------------
#    Functions for Loading UNGL Data
# ---------------------------------------------------------
def load_stations(file=os.path.join(ungl_data,'DataHoldings.txt'), station=None):
    '''
    Get station names and positions from UNGL file
    '''
    df = pd.read_csv(file,
                #names=['site','lat','lon','height','sx','y','z','dtbeg','dtend','dtmod','nsol','origname'],
                names=['site','lat','lon','height','start','end'],
                usecols = [0,1,2,3,7,8],
                skiprows=1,
                parse_dates = ['start','end'],
                delim_whitespace=True,
                engine='python',
                )
    if station: #NOTE: would be more efficient to grep for specific line, and read only that
        df = df[df.site == station]

    return df

def load_stations_old(file=os.path.join(ungl_data,'llh')):
    '''
    Get station names and positions from UNGL file
    '''
    df = pd.read_csv(file,
                 names=['site','lat','lon','height'],
                 delim_whitespace=True,
                 engine='python',
                 )
    return df


def load_steps(ungl_data, station):
    '''Load Unevada shifts for give 4 character station name '''
    os.system('grep {0} steps.txt > station_steps.txt'.format(station))
    os.system("""awk -F" " '$3 == "1" { print $1,$2,$3,$4}' station_steps.txt > steps_code1.txt""")
    os.system("""awk -F" " '$3 == "2" { print $1,$2,$3,$4,$5,$6,$7 }' station_steps.txt > steps_code2.txt""")

    dateparse = lambda x: pd.datetime.strptime(x, '%y%b%d')
    df1 = pd.read_csv(os.path.join(ungl_data,'steps_code1.txt'),
                     names=['site', 'date', 'code', 'note'],
                     delim_whitespace=True,
                     parse_dates = ['date'],
                     date_parser=dateparse,
                     engine='python'
                    )
    df2 = pd.read_csv(os.path.join(ungl_data,'steps_code2.txt'),
                     names=['site', 'date', 'code','thresh_d','distance','mag','id'],
                     delim_whitespace=True,
                     parse_dates = ['date'],
                     date_parser=dateparse,
                     engine='python'
                    )
    return df1,df2


def load_steps_code2(station=None):
    '''Load Unevada shifts for give 4 character station name '''
    #NOTE: isloate just EQ-related steps:
    #awk -F" " '$3 == "2" { print $1,$2,$3,$4,$5,$6,$7 }' steps.txt > steps_code2.txt
    df = pd.read_csv(os.path.join(ungl_data,'steps_code2.txt'),
                     names=['site', 'date', 'code','thresh_d','distance','mag','id'],
                     sep=r"\s*",
                     engine='python'
                    )

    #df.set_index('date')
    df.index = pd.to_datetime(df.date, format='%y%b%d')

    if station:
        df = df[df.site == station] #if not specific station, return whole database

    return df


def load_midas(station=None):
    '''
    Midas is automatic UNevada GPS velocity solution

    Blewitt, G., C. Kreemer, W.C. Hammond, J. Gazeaux, 2016, MIDAS robust trend estimator
    for accurate GPS station velocities without step detection, Journal of Geophysical Research
    doi: 10.1002/2015JB012552.

    (See http://geodesy.unr.edu/NGLStationPages/decyr.txt for translation to YYMMMDD format)
    '''
    #!grep {station} /Volumes/OptiHDD/data/GPS/unevada/midas.IGS08.txt > midas.IGS08.station.txt  #just 1 station
    df = pd.read_csv(os.path.join(ungl_data,'midas.IGS08.txt'),
                     header=None,
                     names=['site', 'version', 'start', 'end', 'years', 'epochs', 'epochs_good', 'pairs',
                        'east', 'north', 'up', 'err_e', 'err_n', 'err_u', 'e0', 'n0', 'u0',
                        'out_e', 'out_n', 'out_u', 'sig_e', 'sig_n', 'sig_u', 'nsteps'],
                     sep=r"\s*",
                     engine='python',
                    )

    #df.set_index('date')
    #df.index = pd.to_datetime(df.date, format='%y%b%d')
    if station:
        df = df[df.site == station] #if not specific station, return whole database

    return df


def download_data(station, overwrite=False, url='http://geodesy.unr.edu/gps_timeseries/tenv3/IGS08'):
    procede = True
    localfile = station + '.IGS08.tenv3'
    if os.path.exists(localfile):
        if overwrite:
            print('Overwriting ' + station)
        else:
            print(station + ' already dowloaded... skipping')
            procede = False

    if procede:
        url = '{}/{}.IGS08.tenv3'.format(url, station)
        print('Downloading {} ...'.format(url))
        savefile = os.path.basename(url)
        try:
            localfile, result = urllib.request.urlretrieve(url, savefile)
        except Exception as e:
            print(station, e)
            return None

    return localfile


def load_tenv3(envfile):
    '''Load GPS timeseries into pandas dataframe with timestamps as index '''
    #http://geodesy.unr.edu/gps_timeseries/README_tenv3.txt
    df = pd.read_csv(envfile,
                     skiprows=1,
                     header=None,
                     names=['site', 'date', 'decyear', 'mjd', 'week', 'day',
                        'reflon', 'e0', 'east', 'n0', 'north', 'u0', 'up', 'ant',
                        'sig_e', 'sig_n', 'sig_u', 'corr_en', 'corr_eu', 'corr_nu'],
                     sep=r"\s*",
                     engine='python',
                    )

    #df.set_index('date')
    df.index = pd.to_datetime(df.date, format='%y%b%d')

    return df


def decyear2date(decyear, inverse=False):
    '''
    Pretabulated conversions from decimal year to datestr through 2030 from Unevada

    Examples:
    In[0]: decyear2date(1990.0068)
    Out[0]: Timestamp('1990-01-03 00:00:00')
    In[1]: decyear2date('90JAN01', inverse=True)
    Out[1]: 1990.0014000000001

    '''
    df = pd.read_csv(os.path.join(ungl_data,'decyr.txt'),
                     names=['date','decyear'],
                     sep=' ',
                     )
    df['datetime'] = pd.to_datetime(df.date, format='%y%b%d')

    if inverse:
        converted = df.decyear[df.date == decyear].values[0]
    else:
        #converted = df.datetime[df.decyear == decyear].values[0] #weird, seems to convert to local timezone via numpy
        converted = df.datetime[df.decyear == decyear].iloc[0] #Keep as pandas timestamp

    return converted



def add_midas(df, dfMidas):
    '''
    Add MIDAS estimates to dataframe
    '''
    start = decyear2date(dfMidas.start.iloc[0])
    end = decyear2date(dfMidas.end.iloc[0])
    t = df.ix[start:end, 'decyear'] - dfMidas.start.iloc[0]

    E0 = df.ix[start,'east'] + dfMidas.e0.iloc[0]
    N0 = df.ix[start,'north'] + dfMidas.n0.iloc[0]
    U0 = df.ix[start,'up'] + dfMidas.u0.iloc[0]

    rate_east = dfMidas.east.iloc[0]
    rate_north = dfMidas.north.iloc[0]
    rate_up = dfMidas.up.iloc[0]
    E = E0 + (rate_east * t)
    N = N0 + (rate_north * t)
    U = U0 + (rate_up * t)
    df.ix[start:end,'midas_east'] = E
    df.ix[start:end, 'midas_north'] = N
    df.ix[start:end, 'midas_up'] = U

    rate_east = dfMidas.east.iloc[0] + dfMidas.err_e.iloc[0]
    rate_north = dfMidas.north.iloc[0] + dfMidas.err_n.iloc[0]
    rate_up = dfMidas.up.iloc[0] + dfMidas.err_u.iloc[0]
    E = E0 + (rate_east * t)
    N = N0 + (rate_north * t)
    U = U0 + (rate_up * t)
    df.ix[start:end,'midas_east_ub'] = E
    df.ix[start:end, 'midas_north_ub'] = N
    df.ix[start:end, 'midas_up_ub'] = U

    rate_east = dfMidas.east.iloc[0] - dfMidas.err_e.iloc[0]
    rate_north = dfMidas.north.iloc[0] - dfMidas.err_n.iloc[0]
    rate_up = dfMidas.up.iloc[0] - dfMidas.err_u.iloc[0]
    E = E0 + (rate_east * t)
    N = N0 + (rate_north * t)
    U = U0 + (rate_up * t)
    df.ix[start:end,'midas_east_lb'] = E
    df.ix[start:end, 'midas_north_lb'] = N
    df.ix[start:end, 'midas_up_lb'] = U

    return df
