"""
Functions for working with GPS data from University of Nevada Geodetic Library
"""

import datetime
import pandas as pd
import numpy as np
import os.path
import urllib

from pathlib import Path

datadir = os.path.join(Path(__file__).parent.parent, "data/ungl")
auxdir = os.path.join(Path(__file__).parent.parent, "auxfiles/ungl")
#print(datadir)

# ---------------------------------------------------------
#    Functions for Loading UNGL Data
# ---------------------------------------------------------
def load_stations(file=os.path.join(auxdir, "DataHoldings.txt"), station=None):
    """
    Get station names and positions from UNGL file
    """
    df = pd.read_csv(
        file,
        # names=['site','lat','lon','height','sx','y','z','dtbeg','dtend','dtmod','nsol','origname'],
        names=["site", "lat", "lon", "height", "start", "end"],
        usecols=[0, 1, 2, 3, 7, 8],
        skiprows=1,
        parse_dates=["start", "end"],
        delim_whitespace=True,
    )
    # convert longitude from (0 to 360) to (-180 to 180)
    df["lon"] = np.where(df.lon > 180.0, df.lon - 360, df.lon)

    if (
        station
    ):  # NOTE: would be more efficient to grep for specific line, and read only that
        df = df[df.site == station]

    return df


def load_steps(station):
    """
    Load nearby earthquakes and equipment related changes
    It is a ragged CSV, so parsing is awkward!
    """
    df = pd.read_csv(os.path.join(auxdir, "steps.txt"),
                     names=["site", "date", "code", "note","distance", "mag", "id"],
                     delim_whitespace=True,
                     parse_dates=["date"],
                     date_format="%y%b%d"
    )
    df = df[df.site == station]
    df1 = df.loc[df.code == 1, ["site", "date", "code", "note"]]
    df2 = df[df.code == 2].rename(columns={'note':'thresh_d'})
    
    return df1, df2


def load_midas(station=None, refframe="IGS14"):
    """
    Midas is automatic UNevada GPS velocity solution

    Blewitt, G., C. Kreemer, W.C. Hammond, J. Gazeaux, 2016, MIDAS robust trend estimator
    for accurate GPS station velocities without step detection, Journal of Geophysical Research
    doi: 10.1002/2015JB012552.

    (See http://geodesy.unr.edu/NGLStationPages/decyr.txt for translation to YYMMMDD format)
    """
    #!grep {station} /Volumes/OptiHDD/data/GPS/unevada/midas.IGS08.txt > midas.IGS08.station.txt  #just 1 station
    df = pd.read_csv(
        os.path.join(auxdir, "midas.{}.txt".format(refframe)),
        header=None,
        names=[
            "site",
            "version",
            "start",
            "end",
            "years",
            "epochs",
            "epochs_good",
            "pairs",
            "east",
            "north",
            "up",
            "err_e",
            "err_n",
            "err_u",
            "e0",
            "n0",
            "u0",
            "out_e",
            "out_n",
            "out_u",
            "sig_e",
            "sig_n",
            "sig_u",
            "nsteps",
            "latitude",
            "longitude",
            "height",
        ],
        delim_whitespace=True,
    )

    # Convert units from [m] to [mm]
    convert = [
        "e0",
        "n0",
        "u0",
        "east",
        "north",
        "up",
        "sig_e",
        "sig_n",
        "sig_u",
        "err_e",
        "err_n",
        "err_u",
    ]
    df[convert] = df[convert] * 1e3

    # df.set_index('date')
    # df.index = pd.to_datetime(df.date, format='%y%b%d')
    if station:
        df = df[df.site == station]  # if not specific station, return whole database

    return df


def download_data(
    station,
    refframe,  # 'IGS14' or 'NA'
    overwrite=False,
    outdir=datadir,
    loadpredictions=False,
    url="http://geodesy.unr.edu/gps_timeseries/tenv3",
):
    """ 24 hour final solutions:
        http://geodesy.unr.edu/gps_timeseries/tenv3_loadpredictions/HUSB.tenv3 
        http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/HUSB.tenv3
        http://geodesy.unr.edu/gps_timeseries/tenv3/plates/NA/HUSB.NA.tenv3
    """
    procede = True
    if refframe == 'NA':
        url = f"{url}/plates/{refframe}/{station}.{refframe}.tenv3"
    else:
        url = f"{url}/{refframe}/{station}.tenv3"
        if loadpredictions:
            url = url.replace('tenv3/IGS14', 'tenv3_loadpredictions')
        
    localfile = os.path.join(outdir, os.path.basename(url))
    if os.path.exists(localfile):
        if overwrite:
            print("Overwriting " + station)
        else:
            print(station + " already dowloaded... skipping")
            procede = False

    if procede:
        print(f"Downloading {url} ...")
        # savefile = os.path.basename(url)
        try:
            localfile, result = urllib.request.urlretrieve(url, localfile)
        except Exception as e:
            print(station, e)
            return None

    return localfile


def load_tenv3(envfile, loadpredictions=False):
    """Load GPS timeseries into pandas dataframe with timestamps as index 
    
    Renames default column headings:
    site YYMMMDD yyyy.yyyy __MJD week d reflon _e0(m) __east(m) ____n0(m) _north(m) u0(m) ____up(m) _ant(m) sig_e(m) sig_n(m) sig_u(m) __corr_en __corr_eu __corr_nu __ee_ntal __nn_ntal __uu_ntal __ee_ntol __nn_ntol __uu_ntol __ee_hydl __nn_hydl __uu_hydl __ee_masc __nn_masc __uu_masc __ee_rese __nn_rese __uu_rese __ee_wus_ __nn_wus_ __uu_wus_ __ee_wusa __nn_wusa __uu_wusa
    """
    # http://geodesy.unr.edu/gps_timeseries/README_tenv3.txt
    # http://geodesy.unr.edu/gps_timeseries/README_tenv3load.txt
    # Overwrite
    names=[
        "site",
        "date",
        "decyear",
        "mjd",
        "week",
        "day",
        "reflon",
        "e0",
        "east",
        "n0",
        "north",
        "u0",
        "up",
        "ant",
        "sig_e",
        "sig_n",
        "sig_u",
        "corr_en",
        "corr_eu",
        "corr_nu"]

    loadnames = [
        'ee_ntal',
        'nn_ntal',
        'uu_ntal',
        'ee_ntol',
        'nn_ntol',
        'uu_ntol',
        'ee_hydl',
        'nn_hydl',
        'uu_hydl',
        'ee_masc',
        'nn_masc',
        'uu_masc',
        'ee_rese',
        'nn_rese',
        'uu_rese',
        'ee_wus',
        'nn_wus',
        'uu_wus',
        'ee_wusa',
        'nn_wusa',
        'uu_wusa']
    
    if loadpredictions:
        names += loadnames
    else:
        names += ["latitude", "longitude", "height"]

    df = pd.read_csv(
        envfile,
        skiprows=1,
        header=None,
        names=names,
        delim_whitespace=True,
    )

    # Convert units from [m] to [mm]
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

    df.index = pd.to_datetime(df.date, format="%y%b%d")

    return df


def decyear2date(decyear, inverse=False):
    """
    Pretabulated conversions from decimal year to datestr through 2030 from Unevada

    Examples:
    In[0]: decyear2date(1990.0068)
    Out[0]: Timestamp('1990-01-03 00:00:00')
    In[1]: decyear2date('90JAN01', inverse=True)
    Out[1]: 1990.0014000000001

    """
    df = pd.read_csv(
        os.path.join(auxdir, "decyr.txt"),
        skiprows=1,
        names=['date', 'decimalyr', 'year', 'mm', 'dd', 'hh', 'day', 'mjday', 'week', 'd',  'J2000_sec'],
        delim_whitespace=True,
    )
    df["datetime"] = pd.to_datetime(df.date, format="%y%b%d")

    if inverse:
        converted = df.decimalyr[df.date == decyear].values[0]
    else:
        converted = df.datetime[df.decimalyr == decyear].iloc[0]

    return converted


def add_midas(df, dfMidas):
    """
    Add MIDAS estimates to dataframe
    """
    start = decyear2date(dfMidas.start.iloc[0])
    end = decyear2date(dfMidas.end.iloc[0])
    t = df.loc[start:end, "decyear"] - dfMidas.start.iloc[0]

    E0 = df.loc[start, "east"] + dfMidas.e0.iloc[0]
    N0 = df.loc[start, "north"] + dfMidas.n0.iloc[0]
    U0 = df.loc[start, "up"] + dfMidas.u0.iloc[0]

    rate_east = dfMidas.east.iloc[0]
    rate_north = dfMidas.north.iloc[0]
    rate_up = dfMidas.up.iloc[0]
    E = E0 + (rate_east * t)
    N = N0 + (rate_north * t)
    U = U0 + (rate_up * t)
    df.loc[start:end, "midas_east"] = E
    df.loc[start:end, "midas_north"] = N
    df.loc[start:end, "midas_up"] = U

    rate_east = dfMidas.east.iloc[0] + dfMidas.err_e.iloc[0]
    rate_north = dfMidas.north.iloc[0] + dfMidas.err_n.iloc[0]
    rate_up = dfMidas.up.iloc[0] + dfMidas.err_u.iloc[0]
    E = E0 + (rate_east * t)
    N = N0 + (rate_north * t)
    U = U0 + (rate_up * t)
    df.loc[start:end, "midas_east_ub"] = E
    df.loc[start:end, "midas_north_ub"] = N
    df.loc[start:end, "midas_up_ub"] = U

    rate_east = dfMidas.east.iloc[0] - dfMidas.err_e.iloc[0]
    rate_north = dfMidas.north.iloc[0] - dfMidas.err_n.iloc[0]
    rate_up = dfMidas.up.iloc[0] - dfMidas.err_u.iloc[0]
    E = E0 + (rate_east * t)
    N = N0 + (rate_north * t)
    U = U0 + (rate_up * t)
    df.loc[start:end, "midas_east_lb"] = E
    df.loc[start:end, "midas_north_lb"] = N
    df.loc[start:end, "midas_up_lb"] = U

    return df
