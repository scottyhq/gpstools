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


def load_panga(site):
    '''Load GPS timeseries into pandas dataframe with timestamps as index '''
    #http://www.geodesy.cwu.edu/data/bysite/   'east', 'n0', 'north', 'u0', 'up'
    dfe = pd.read_csv(site + '.lon',
                     comment='#',
                     header=None,
                     names=['decyear','east','sig_e'],
                     delim_whitespace=True,
                    )
    dfn = pd.read_csv(site + '.lat',
                     comment='#',
                     header=None,
                     names=['decyear','north','sig_n'],
                     delim_whitespace=True,
                    )
    dfu = pd.read_csv(site + '.rad',
                     comment='#',
                     header=None,
                     names=['decyear','up','sig_u'],
                     delim_whitespace=True,
                    )

    #df.set_index('date')
    #df.index = pd.to_datetime(df.date, format='%y%b%d')

    return df
