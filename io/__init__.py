"""
Input/Output routines for each processing center
"""

__all__ = ['ungl','panga','jpl','sopac']
__version__ = 0.1
__author__ = 'scottyh@uw.edu'
__lastupdate__ = '10/2017'

from . import ungl,panga,jpl,sopac

import geopandas as gpd
from shapely.geometry import Point

def toGeoDataFrame(df, lat='lat', lon='lon', epsg=4326):
    ''' Convert pandas dataframe with point positions to geopandas '''
    points = df.apply(lambda row: Point([row.lon, row.lat]), axis=1)
    gf = gpd.GeoDataFrame(df, geometry=points, crs={'init' :'epsg:{}'.format(epsg)})

    return gf
