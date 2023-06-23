"""
Functions for working with PANGA GPS data
http://www.geodesy.cwu.edu/data/bysite/

#NOTE: about 400MB for all stations, updated daily:
http://www.geodesy.cwu.edu/panga/officialresults/archives/panga_raw.zip

Data acknowledgement should read "GPS time series provided by the Pacific Northwest Geodetic Array, Central Washington University."
"""

import os
import urllib
import pandas as pd
import datetime as DT
import sys
from pathlib import Path

datadir = os.path.join(sys.modules['gpstools'].DATADIR, 'panga')
if not os.path.isdir(datadir):
    os.makedirs(datadir)
auxdir = os.path.join(sys.modules['gpstools'].AUXDIR, 'panga')


def load_stations(file=os.path.join(auxdir, "sites.csv"), station=None):
    """
    Get station names and positions from UNGL file
    """
    df = pd.read_csv(
        file,
        names=["site", "description", "lat", "lon", "height", "start", "end"],
        skiprows=1,
        # parse_dates = ['start','end'],
    )
    # NOTE: would be more efficient to grep for specific line, and read only that
    if station:
        df = df[df.site == station]

    return df


def download_data(
    station,
    product="raw",
    overwrite=False,
    outdir=datadir,
    baseurl="http://www.geodesy.org/cgi-bin/timeseries_data.pl",
):
    procede = True
    names = dict(lon="e", lat="n", rad="u")
    for key, val in names.items():
        query = "?n=panga&s={}&p={}&f=daily&c={}".format(station, product, key)
        localfile = os.path.join(outdir, "{}{}.csv".format(station, val))
        if os.path.exists(localfile):
            if not overwrite:
                procede = False

        if procede:
            url = "{0}/{1}".format(baseurl, query)
            print("Downloading {} ...".format(url))
            # savefile = os.path.basename(url)
            try:
                localfile, result = urllib.request.urlretrieve(url, localfile)
            except Exception as e:
                print(station, e)


def decyear2datetime(atime):
    """
    https://stackoverflow.com/questions/19305991/convert-fractional-years-to-a-real-date-in-python
    """
    year = int(atime)
    remainder = atime - year
    boy = DT.datetime(year, 1, 1)
    eoy = DT.datetime(year + 1, 1, 1)
    seconds = remainder * (eoy - boy).total_seconds()

    return boy + DT.timedelta(seconds=seconds)


def datetime2decyear(adatetime):
    """
    https://stackoverflow.com/questions/19305991/convert-fractional-years-to-a-real-date-in-python
    """
    year = adatetime.year
    boy = DT.datetime(year, 1, 1)
    eoy = DT.datetime(year + 1, 1, 1)

    return year + ((adatetime - boy).total_seconds() / ((eoy - boy).total_seconds()))


def load_panga_fit_info(filepath):
    """ parse panga processing fit data from header
    (only lines starting with #)
    """
    from itertools import takewhile

    with open(filepath, "r") as fobj:
        headiter = takewhile(lambda s: s.startswith("#"), fobj)
        header = list(headiter)

    print("TODO- convert to JSON/dictionary metadata?")
    return metadata


def load_panga(site):
    """Load GPS timeseries into pandas dataframe with timestamps as index """
    # http://www.geodesy.cwu.edu/data/bysite/   'east', 'n0', 'north', 'u0', 'up'
    def load_csv(path):
        print(path)
        tmp = pd.read_csv(
            path,
            comment="#",
            header=None,
            names=["decyear", "comp", "error"],
            delim_whitespace=True,
        )
        return tmp

    df = load_csv(os.path.join(datadir, "{}e.csv".format(site)))
    if df.decyear[0] == "<pre>":
        print("No timeseries data for {}".format(site))
        return

    df.columns = ["decyear", "east", "err_e"]  #'north', 'up', 'err_e', 'err_n', 'err_u'

    tmp = load_csv(os.path.join(datadir, "{}n.csv".format(site)))
    df[["north", "err_n"]] = tmp[["comp", "error"]]

    tmp = load_csv(os.path.join(datadir, "{}u.csv".format(site)))
    df[["up", "err_u"]] = tmp[["comp", "error"]]

    # df['just_date'] = df['dates'].dt.date # Get rid of hours, minutes, seconds
    # complicated, but results in pandas DateTimeIndex...
    # query nearest index value:
    # https://github.com/pandas-dev/pandas/issues/8845
    df["date"] = pd.to_datetime(df.decyear.apply(decyear2datetime).dt.date)
    df.set_index("date", inplace=True)

    # NOTE: occaisionally Panga has decimal year values falling on same days
    # pandas requires unique index values, so remove the duplicates
    # This takes mean of duplicate dates and fills in missing dates with NaN
    df = df.resample("D").mean()

    return df
