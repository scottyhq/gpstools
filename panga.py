"""
Functions for working with PANGA GPS data
http://www.geodesy.cwu.edu/data/bysite/

#NOTE: about 400MB for all files, updated daily
http://www.geodesy.cwu.edu/panga/officialresults/archives/panga_raw.zip

http://www.geodesy.cwu.edu/panga/timeseries_data.php?n=panga&s=TPW2&p=cleaned&f=daily&c=lat
http://www.geodesy.cwu.edu/panga/timeseries_data.php?n=panga&s=TPW2&p=cleaned&f=daily&c=lon
http://www.geodesy.cwu.edu/panga/timeseries_data.php?n=panga&s=TPW2&p=cleaned&f=daily&c=rad

http://www.geodesy.cwu.edu/panga/timeseries_data.php?n=panga&s=TPW2&p=raw&f=daily&c=lat
http://www.geodesy.cwu.edu/panga/timeseries_data.php?n=panga&s=TPW2&p=raw&f=daily&c=rad



Data acknowledgement should read "GPS time series provided by the Pacific Northwest Geodetic Array, Central Washington University."
"""

import os
import urllib
import pandas as pd
import datetime as DT

def download_data(station,
                product='raw',
                overwrite=False,
                outdir='./',
                baseurl='http://www.geodesy.org/panga/timeseries_data.php'):
    procede = True
    names = dict(lon='e', lat='n', rad='u')
    for key,val in names.items():
        query='?n=panga&s={}&p={}&f=daily&c={}'.format(station,product,key)
        localfile = os.path.join(outdir, '{}{}.csv'.format(station,val))
        if os.path.exists(localfile):
            if overwrite:
                print('Overwriting ' + localfile)
            else:
                print(station + ' already dowloaded... skipping')
                procede = False

        if procede:
            url = '{0}/{1}'.format(baseurl, query)
            print('Downloading {} ...'.format(url))
            #savefile = os.path.basename(url)
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


def load_panga_fit_info(filename, datadir='./'):
    ''' parse panga processing fit data from header
    (only lines starting with #)
    '''
    from itertools import takewhile
    with open(filename, 'r') as fobj:
        headiter = takewhile(lambda s: s.startswith('#'), fobj)
        header = list(headiter)

    print('TODO- convert to JSON/dictionary metadata?')
    return metadata


def load_panga(site, datadir='./'):
    '''Load GPS timeseries into pandas dataframe with timestamps as index '''
    #http://www.geodesy.cwu.edu/data/bysite/   'east', 'n0', 'north', 'u0', 'up'
    def load_csv(path):
        tmp = pd.read_csv(path,
                         comment='#',
                         header=None,
                         names=['decyear','comp','error'],
                         delim_whitespace=True,
                        )
        return tmp
    df = load_csv(os.path.join(datadir, '{}e.csv'.format(site)))
    df.columns = ['decyear', 'east', 'err_e'] #'north', 'up', 'err_e', 'err_n', 'err_u'

    tmp = load_csv(os.path.join(datadir, '{}n.csv'.format(site)))
    df[ ['north','err_n']]= tmp[ ['comp','error']]

    tmp = load_csv(os.path.join(datadir, '{}u.csv'.format(site)))
    df[ ['up','err_u']]= tmp[ ['comp','error']]

    #df['just_date'] = df['dates'].dt.date # Get rid of hours, minutes, seconds
    # complicated, but results in pandas DateTimeIndex...
    #query nearest index value:
    #https://github.com/pandas-dev/pandas/issues/8845
    df['date'] = pd.to_datetime(df.decyear.apply(decyear2datetime).dt.date)
    df.set_index('date', inplace=True)

    return df
