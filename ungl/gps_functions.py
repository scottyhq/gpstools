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

import matplotlib.pyplot as plt
import matplotlib.dates as pltdate

import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.signal import sawtooth
from scipy import stats


# ---------------------------------------------------------
#    Functions for Loading UNGL Data
# ---------------------------------------------------------
def load_stations(file='/Volumes/OptiHDD/data/GPS/unevada/llh'):
    '''
    Get station names and positions from UNGL file
    '''
    df = pd.read_csv(file,
                 names=['site','lat','lon','height'],
                 sep='  ',
                 engine='python',
                 )
    return df


def load_steps(station=None):
    '''Load Unevada shifts for give 4 character station name '''
    #NOTE: isloate just EQ-related steps:
    #awk -F" " '$3 == "2" { print $1,$2,$3,$4,$5,$6,$7 }' steps.txt > steps_code2.txt
    df = pd.read_csv('/Volumes/OptiHDD/data/GPS/unevada/steps_code2.txt',
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
    df = pd.read_csv('/Volumes/OptiHDD/data/GPS/unevada/midas.IGS08.txt',
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
    df = pd.read_csv('/Volumes/OptiHDD/data/GPS/unevada/decyr.txt',
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


# ---------------------------------------------------------
#    Plotting Functions
# ---------------------------------------------------------
def plot_all(df, dfSteps=None, dfMidas=None, dfFit=None, columns=['north', 'east','up'], axhline=False, title=''):
    #Plot daily positions
    fig, (ax,ax1,ax2) =  plt.subplots(3,1,sharex=True,figsize=(8.5,11))
    ax.plot(df.index, df[columns[0]], 'b.', label='NS')
    #ax.set_title('NS')

    ax1.plot(df.index, df[columns[1]], 'g.', label='EW')
    #ax1.set_title('EW')

    ax2.plot(df.index, df[columns[2]], 'r.', label='Z')
    #ax2.set_title('Z')

    # Add midas velocities
    if isinstance(dfMidas, pd.DataFrame):
        X = [decyear2date(dfMidas.start.iloc[0]), decyear2date(dfMidas.end.iloc[0])]
        dt = dfMidas.years.values

        E0 = (df.east.iloc[0] + dfMidas.e0).values
        N0 = (df.north.iloc[0] + dfMidas.n0).values
        U0 = (df.up.iloc[0] + dfMidas.u0).values

        E = [E0, E0 + dfMidas.east.values * dt] #dfMidas.e0.values not sure about this value
        N = [N0, N0 + dfMidas.north.values * dt]
        U = [U0,  U0 + dfMidas.up.values * dt]
        ax.plot(X, N, 'k-' , lw=2, label='MIDAS')
        ax1.plot(X, E, 'k-', lw=2 )
        ax2.plot(X, U, 'k-', lw=2 )

    # Add discontinuities
    if isinstance(dfSteps, pd.DataFrame):
        for step in dfSteps.index.intersection(df.index):
            for axes in (ax,ax1,ax2):
                axes.axvline(step, color='k', linestyle='dashed')

    # Add Function Fits
    if isinstance(dfFit, pd.DataFrame):
        ax.plot(dfFit.index, dfFit.fit_north, 'k--' , lw=2, label='L2')
        ax1.plot(dfFit.index.values, dfFit.fit_east.values, 'k--', lw=2)
        ax2.plot(dfFit.index.values, dfFit.fit_up.values, 'k--', lw=2 )

    if axhline:
        for axes in (ax,ax1,ax2):
            axes.axhline(color='k',lw=1)

    ax.legend(loc='upper left')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    plt.suptitle(title, fontsize=16)
    plt.ylabel('Position [m]')

    months = pltdate.MonthLocator()
    years = pltdate.YearLocator()
    for axes in (ax,ax1,ax2):
        axes.xaxis.set_major_locator(years)
        axes.xaxis.set_minor_locator(months) #too much
        axes.fmt_xdata = pltdate.DateFormatter('%Y-%m-%d')
        axes.grid(True, axis='x')

    fig.autofmt_xdate()



# ---------------------------------------------------------
#    Analysis Functions
# ---------------------------------------------------------
def get_extrema(df, component, output=True):
    # Get all mins and max
    # since function is analytic, can find zeros of second derivative explicitly
    # http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
    # Approach depends on noisiness of data! basic approach for smooth function below
    data = df[component]

    ind_extrema = np.diff(np.sign(np.diff(data))).nonzero()[0] + 1
    ind_minima = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1
    ind_maxima = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1

    # Timestamps of mins and max of fit function
    ts_max = df.index[ind_maxima]
    ts_min = df.index[ind_minima]

    # Date Difference with pandas (days between maxima / Period)
    periods = np.diff(df.index[ind_extrema].to_pydatetime())

    if output:
        print('Minima: ', df.date[ind_minima].values)
        print('Maxima: ', df.date[ind_maxima].values)
        print('Periods (days): ', [x.days for x in periods])
        #df.plot(x=df.index, y=[component,'seasonal'],style=['k.','r-'], lw=2, figsize=(11,4))
        #plt.figure(figsize=(11,4))
        #plt.plot(df.index, df[component],'k.')
        #plt.plot(df.index, df.seasonal, 'r-', lw=2)
        #plt.plot(df.index[ind_minima], df.seasonal[ind_minima], 'bo', ms=10)
        #plt.plot(df.index[ind_maxima], df.seasonal[ind_maxima], 'ro', ms=10)

    return ind_minima, ind_maxima

def add_midas(df, dfMidas):
    '''
    Add MIDAS estimates to dataframe
    '''
    t = (df.decyear - df.decyear.iloc[0]).values
    E0 = (df.east.iloc[0] + dfMidas.e0).values
    N0 = (df.north.iloc[0] + dfMidas.n0).values
    U0 = (df.up.iloc[0] + dfMidas.u0).values

    E = E0 + (dfMidas.east.values * t)
    N = N0 + (dfMidas.north.values * t)
    U = U0 + (dfMidas.up.values * t)

    df['midas_east'] = E
    df['midas_north'] = N
    df['midas_up'] = U

    return df


# Fit sinusoid, get date of peaks and troughs!
def do_detrend(df, col='up', start=0, end=-1):
    '''
    remove linear trend from GPS data, note start and end either need to be integers or dates/timestamps
    '''
    #between range of dates
    df = df.ix[start:end]

    df['ints'] = df.index.asi8
    df['elapsed_s'] = (df.ints - df.ints[0])/1e9 #sec

    y = df[col]  # response
    X = df.elapsed_s  # predictor
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    model = sm.OLS(y, X)
    est = model.fit()
    rate = est.params[1]*3.1536e10
    print('{} Rate [mm/yr]={}'.format(col, rate))
    #print(est.summary()) # Big stats-rich summary!

    df['linear_{}'.format(col)] = est.predict(X)
    df['detrend1_{}'.format(col)] = y - est.predict(X)

    calc_RMSE(y, est.predict(X))

    return df


def detrend(df, col='up'):
    '''
    remove linear trend from GPS data
    '''
    df['ints'] = df.index.asi8
    df['elapsed_s'] = (df.ints - df.ints[0])/1e9 #sec

    y = df[col]  # response
    X = df.elapsed_s  # predictor
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    model = sm.OLS(y, X)
    est = model.fit()
    rate = est.params[1]*3.1536e10
    print('Rate [mm/yr]={}'.format(rate))
    #print(est.summary()) # Big stats-rich summary!
    df['linear_fit'] = est.predict(X)
    df['detrend'] = y - df.linear_fit

    return df


def calc_RMSE(data, fit):
    ''' formula for root mean square error'''
    residuals = data - fit
    rmse = np.sqrt((np.sum(residuals**2) / residuals.size))
    print('RMSE = ', rmse)
    return rmse

def fit_linear(df, cols=['up','east','north']):
    df['ints'] = df.index.asi8
    df['elapsed_s'] = (df.ints - df.ints[0])/1e9 #sec

    for col in cols:
        y = df[col]  # response
        X = df.elapsed_s  # predictor
        X = sm.add_constant(X)  # Adds a constant term to the predictor
        model = sm.OLS(y, X)
        est = model.fit()
        rate = est.params[1]*3.1536e10
        df[col+'_fit'] = est.predict(X)

    # Add 95% confidence interval for regression
    #http://markthegraph.blogspot.com.co/2015/05/using-python-statsmodels-for-ols-linear.html
    #y_hat = est.predict(X)
    #mean_x = X.T[1].mean()
    #n = len(X)
    #dof = n - est.df_model - 1

    return df



def cull_outliers(df, cols=['detrend1_north', 'detrend1_east', 'detrend1_up'], nstd=3):
    ''' Remove points greater than 3std (**from detrended signal - first subtract linear trend)'''
    for col in cols:
        thresh = df[col].std()*nstd
        errors = np.abs(df[col] - df[col].mean()) #mean effectively zero in this case...

        #df = df.query('errors >= @thresh') #return dataframe w/ omissions
        ind = (errors >= thresh)
        print('{}: Culled {} points'.format(col, np.sum(ind)))
        df.ix[ind, col] = np.nan #return original dataframe with values set to nans


    return  df.dropna() #fitting functions don't like nans...


#%qtconsole
def remove_eq1(df, eq, dt=90, showplot=False):
    '''Remove coseismic EQ offset by equalizing means for "dt" days around "eq"'''
    cols = ['up','north','east']
    d = pd.Timedelta(1,'D')
    df_after = df.ix[eq-d:eq-d+(d*dt), cols]
    df_before = df.ix[eq-(d*dt)-d:eq-d, cols]

    ua,ea,na = df_after.mean()
    ub,eb,nb = df_before.mean()

    if showplot:
        fig, axes = plt.subplots(3,1, figsize=(6,11), sharex=True)
        for col,ax in zip(cols,axes):
            ax.plot(df_before.index, df_before[col],'k.')
            ax.plot(df_after.index, df_after[col],'k.')
            ax.axhline(df_before[col].mean(), color='red', xmax=0.5)
            ax.axhline(df_after[col].mean(), color='red', xmin=0.5)
            ax.axvline(eq,color='k',linestyle='dashed')
            ax.set_title(col)
        fig.autofmt_xdate()

    # Before EQ - After EQ
    du = ub - ua
    de = eb - ea
    dn = nb - na
    print('EQ Offsets [mm]:\n u={}, e={}, n={}'.format(du, de, dn))

    #df_shift = df.copy()
    #df_shift[eq:].u += du
    #df_shift[eq:].e += de
    #df_shift[eq:].n += dn

    return du, de, dn



#%qtconsole
def remove_eq2(df, eq, dt=120, showplot=False):
    '''Remove coseismic EQ offset by finding linear intercept on data before and after'''
    cols = ['up','north','east']
    d = pd.Timedelta(1,'D')
    df_after = df.ix[eq-d:eq-d+(d*dt), cols]
    df_before = df.ix[eq-(d*dt)-d:eq-d, cols]

    dfb = fit_linear(df_before)
    dfa = fit_linear(df_after)

    if showplot:
        fig, axes = plt.subplots(3,1, figsize=(6,11), sharex=True)
        for col,ax in zip(cols,axes):
            ax.plot(df_before.index, df_before[col],'k.')
            ax.plot(df_after.index, df_after[col],'k.')
            ax.plot(df_before.index, df_before[col+'_fit'], 'r-')
            ax.plot(df_after.index, df_after[col+'_fit'], 'r-')
            ax.axvline(eq,color='k',linestyle='dashed')
            ax.set_title(col)
        fig.autofmt_xdate()

    # Before EQ - After EQ
    #print(df_before['up_fit'][-1], df_after['up_fit'][0])
    du = df_after['up_fit'][0] - df_before['up_fit'][-1]
    de = df_after['east_fit'][0] - df_before['east_fit'][-1]
    dn = df_after['north_fit'][0] - df_before['north_fit'][-1]
    offsets_mm = np.array([du,de,dn])*1e3
    print('EQ Offsets [mm]:\n u={0:.2f}, e={1:.2f}, n={2:.2f}'.format(*offsets_mm))

    return du, de, dn


def myfit(x, y, F, guess):
    '''
    Add column with best-fit information to dataframe
    '''
    popt, pcov = curve_fit(F, x, y, guess)
    result = F(x, *popt)

    residuals = y - result
    rmse = np.sqrt((np.sum(residuals**2) / residuals.size))

    print('myfit:', popt)

    # pretty-print OSU model params
    print('OSU Fit:\n---------')
    print('RMSE [mm] = {:.3e}'.format(rmse*1e3))
    print('X0 [m] = {:.3f}'.format(popt[1]))
    print('V [mm/yr] = {:.3f}'.format(popt[2]*1e3 ))
    print('Step [mm] = {:.3f}'.format(popt[3]*1e3 ))
    print('s1,c1,T1 [mm, mm, yr] = {:.3f}, {:.3f}, {:.3f}'.format(popt[5]*1e3,
          popt[6]*1e3,
          popt[9]))
    print('s2,c2,T2 [mm, mm, yr] = {:.3f}, {:.3f}, {:.3f} '.format(popt[7]*1e3,
      popt[8]*1e3,
      popt[10]))

    return result

# ---------------------------------------------------------
#    Fitting Functions
# ---------------------------------------------------------

def sinfunc(t,a,f):
    return a * np.sin((2*np.pi*f)*t)

def sinfunc2(t,a1,f1,a2,f2):
    return a1*np.sin((2*np.pi*f1)*t) + a2*np.sin((2*np.pi*f2)*t)

def sawfunc(t,a,f,width):
    ''' Sawtooth / Triangle Wave from SciPy Signal'''
    return  a * sawtooth((2*np.pi*f)*t, width)

def sawfunc2(t, a1,f1,w1, a2,f2,w2):
    ''' Sawtooth / Triangle Wave from SciPy Signal'''
    return  a1 * sawtooth((2*np.pi*f1)*t, w1) + a2 * sawtooth((2*np.pi*f2)*t, w2)

def heaviside(t):
    ''' Heavidside step function'''
    return 0.5 * (np.sign(t) + 1)

def osu(t,t0,x0,v,b,tj,s1,c1,s2,c2,T1,T2):
    '''
    Preferred general model for OSU processing (See Bevis 2014 eq 5)
    #http://www.mathworks.com/matlabcentral/fileexchange/27783-fitting-data-with-a-sudden-discontinuity/content/html/exampleShift.html
    2 frequencies = 4 fourier series parameters sn, cn)
    # T1=365.25,T2=182.625
    '''
    w1 = (2 * np.pi) / T1
    w2 = (2 * np.pi) / T2

    t = t - t0
    tj = tj - t0

    '''
    d2s = 86400
    y2s = 31557600
    #convert from decyear to seconds
    t = (t-t0) * y2s
    tj = (tj-t0) * y2s
    w1 = w1 / d2s
    w2 = w2 / d2s
    '''

    # How to fit n heaviside step functions to data?
    #bj = [] #need to solve for B
    #for step in tj:
    #    b*heaviside(t-tj)

    f = ( x0 +
          v * t +
          b * heaviside(t-tj) +
          s1 * np.sin(w1*t) + c1 * np.cos(w1*t) + #)#
          s2 * np.sin(w2*t) + c2 * np.cos(w2*t) )

    return f

def scott(t,t0,x0,v,b,tj,s1,c1,s2,c2,T1,T2):
    '''
    Preferred general model for OSU processing (See Bevis 2014 eq 5)
    #http://www.mathworks.com/matlabcentral/fileexchange/27783-fitting-data-with-a-sudden-discontinuity/content/html/exampleShift.html
    2 frequencies = 4 fourier series parameters sn, cn)
    # T1=365.25,T2=182.625
    '''
    w1 = (2 * np.pi) / T1
    w2 = (2 * np.pi) / T2

    t = t - t0
    tj = tj - t0

    '''
    d2s = 86400
    y2s = 31557600
    #convert from decyear to seconds
    t = (t-t0) * y2s
    tj = (tj-t0) * y2s
    w1 = w1 / d2s
    w2 = w2 / d2s
    '''

    # How to fit n heaviside step functions to data?
    #bj = [] #need to solve for B
    #for step in tj:
    #    b*heaviside(t-tj)

    f = ( x0 +
          v * t +
          b * heaviside(t-tj) +
          sawfunc(t,a,f,width) )

    return f
