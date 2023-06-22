import gpstools as gps
from pathlib import Path
import os.path
import pandas
import numpy as np

ROOTDIR = Path(__file__).parent.parent
TESTDATA = str(ROOTDIR / 'tests/data')
AUXDIR = str(ROOTDIR / 'gpstools/auxfiles')
DATADIR = str(ROOTDIR / 'gpstools/data')

#STATION = 'UTUR'
STATION = 'HUSB'

def test_version_string():
    assert isinstance(gps.__version__, str)


class TestUNGL:
    def test_update_auxfiles(self):
        auxdir = os.path.join(AUXDIR, 'ungl')
        auxfiles = ['decyr.txt','midas.IGS14.txt','midas.NA12.txt', 'DataHoldings.txt', 'steps.txt']
        gps.auxfiles.ungl.update()
        for file in auxfiles:
            assert file in os.listdir(auxdir)
        
    def test_load_stations(self):
        df = gps.io.ungl.load_stations()
        assert isinstance(df, pandas.DataFrame)
        assert isinstance(df.index, pandas.RangeIndex)
        assert list(df.columns) == ['site', 'lat', 'lon', 'height', 'start', 'end']
        assert len(df) != 0

    def test_download_single_station(self):
        tenv3 = gps.io.ungl.download_data(STATION, 'IGS14', outdir=TESTDATA)
        assert os.path.isfile(tenv3)

    def test_read_tenv3(self):
        df = gps.io.ungl.load_tenv3(f'{TESTDATA}/{STATION}.tenv3')
        assert isinstance(df, pandas.DataFrame)
        assert len(df.columns) == 23
        assert isinstance(df.index, pandas.DatetimeIndex)

    def test_decyear2date(self):
        # Pisaqua Earthquake
        eq = pandas.Timestamp('2014-04-01 23:46:47', tz='UTC')
        tj = gps.io.ungl.decyear2date(eq.strftime('%y%b%d').upper(), inverse=True)
        assert tj == 2014.2478

    def test_read_midas(self):
        df = gps.io.ungl.load_midas(STATION)
        assert isinstance(df, pandas.DataFrame)
        assert len(df.columns) == 24
        assert isinstance(df.index, pandas.Index)
        assert df.version.iloc[0] == 'MIDAS4'

    def test_load_steps(self):
        dfCh, dfEq = gps.io.ungl.load_steps(station=STATION)
        assert isinstance(dfCh, pandas.DataFrame)
        assert list(dfCh) == ['site', 'date', 'code', 'note'] 
        assert isinstance(dfEq, pandas.DataFrame)
        assert list(dfEq) == ['site', 'date', 'code', 'thresh_d', 'distance', 'mag', 'id']


# class TestPanga:
#     def test_download_single_station(self):
#         tenv3 = gps.io.ungl.download_data('TPW2', 'ITRF2008', outdir=TESTDATA)
#         assert os.path.isfile(tenv3)


class TestAnalysis:
    tenv3 = gps.io.ungl.download_data(STATION, 'IGS14', outdir=TESTDATA)
    DF = gps.io.ungl.load_tenv3(f'{TESTDATA}/{STATION}.tenv3')

    def test_cull_outliers(self):
        components = ['east','north','up']
        df = gps.analysis.cull_outliers(self.DF, cols=components)
        assert len(df) != len(self.DF)

    def test_linear_fit(self):
        df = gps.analysis.fit_linear(self.DF)
        assert isinstance(df, pandas.DataFrame)
        assert np.isin(['fit_up', 'fit_east', 'fit_north'], df.columns).all()

    def test_add_midas(self):
        dfM = gps.io.ungl.load_midas(STATION)
        df = gps.io.ungl.add_midas(self.DF, dfM)
        assert isinstance(df, pandas.DataFrame)
        assert np.isin(['midas_up', 'midas_east', 'midas_north'], df.columns).all()

    def test_osu_fit(self):
        df = self.DF
        comp = 'up'

        # Initial Guess Parameters
        t0 = df.decyear[0] # intial time [yr]
        x0 = df[comp][0] # initial position [m]
        v = (df[comp][-1] - df[comp][0]) / (df.decyear[-1] - df.decyear[0])  # trend [m/yr]
        tj = 2014.2478 # known time of jump (earthquake)
        b = 0.0 # step jump offset amplitude [m]
        s1 = s2 = c1 = c2 = 0.001 # fourier coefficients [m]
        T1 = 1 # fourier period 1 [yr]
        T2 = 0.5 #fourier period 2 [yr]
        guess = [t0,x0,v,b,tj,s1,c1,s2,c2,T1,T2]
        fit = gps.analysis.osu(df.decyear.values, *guess)

        assert isinstance(fit, np.ndarray)
        assert len(fit) == len(df)