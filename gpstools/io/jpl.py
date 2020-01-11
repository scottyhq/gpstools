"""
Functions for working with JPL GPS time series
"""

import os
import pandas as pd

from pathlib import Path

datadir = os.path.join(Path(__file__).parent.parent, "data/jpl")
auxdir = os.path.join(Path(__file__).parent.parent, "auxfiles/jpl")


def load_timeseries(site):
    """ julsec = time in seconds past J2000 """
    df = pd.read_csv(
        os.path.join(datadir, site + ".series"),
        header=None,
        names=[
            "decyear",
            "east",
            "north",
            "up",
            "sig_e",
            "sig_n",
            "sig_u",
            "corr_en",
            "corr_eu",
            "corr_nu",
            "julsec",
            "year",
            "month",
            "day",
            "hour",
            "min",
            "sec",
        ],
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

    # Convert to python timestamps
    df.index = pd.to_datetime(df[["year", "month", "day"]])

    return df


def download_timeseres(site):
    """ ftp://sideshow.jpl.nasa.gov/pub/usrs/mbh/point/ """
    print("todo")
