"""
Functions for analyzing gps time series
* linear fits
* multiparamter fits
* general stats
"""


import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.signal import sawtooth
from scipy import stats
import numpy as np



# ---------------------------------------------------------
#    Analysis Functions
# ---------------------------------------------------------
def find_nearest_trench_point(value):
    path = '/Volumes/OptiHDD/data/plates/PLATES/PLATES_PlateBoundary_ArcGIS/trench.txt'
    tlon,tlat = np.loadtxt(path,unpack=True)
    idx = (np.abs(tlat-value)).argmin()
    return np.array([tlat[idx], tlon[idx]])

def get_geocentric_radius(latitude, a=6378.1370, b=6356.7523):
    '''a is equitorial radius, b is polar radius, WGS84 Ellipsoid'''
    latr = np.radians(latitude)
    num = (a**2 * np.cos(latr))**2 + (b**2 * np.sin(latr))**2
    denom = (a * np.cos(latr))**2 + (b * np.sin(latr))**2
    radius = np.sqrt(num/denom)
    return radius


def add_trench_distance(df):
    distance = np.zeros_like(df.lon)
    for i in range(df.shape[0]):
        gps_point = np.array( [ df.lat[i], df.lon[i] ])
        trench_point = find_nearest_trench_point(gps_point[0])
        distance[i] = spherical_dist(trench_point, gps_point)

    df['distance'] = distance/1e3 #in km

    return df


# Fit sinusoid, get date of peaks and troughs!
def do_detrend(df, col='up', start=0, end=-1):
    '''
    remove linear trend from GPS data, note start and end either need to be integers or dates/timestamps
    '''
    #between range of dates
    #print(type(start))
    if type(start) is int:
        df = df.iloc[start:end]
    else:
        df = df.loc[start:end]

    df['ints'] = df.index.asi8
    df['elapsed_s'] = (df.ints - df.ints[0])/1e9 #sec

    y = df[col]  # response
    X = df.elapsed_s  # predictor
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    model = sm.OLS(y, X)
    est = model.fit()
    rate = est.params[1]*3.1536e7
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
    rate = est.params[1]*3.1536e7
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
        rate = est.params[1]*3.1536e7
        df['fit_'+col] = est.predict(X)

    # Add 95% confidence interval for regression
    #http://markthegraph.blogspot.com.co/2015/05/using-python-statsmodels-for-ols-linear.html
    #y_hat = est.predict(X)
    #mean_x = X.T[1].mean()
    #n = len(X)
    #dof = n - est.df_model - 1

    return df


#http://stackoverflow.com/questions/19413259/efficient-way-to-calculate-distance-matrix-given-latitude-and-longitude-data-in
def spherical_dist(pos1, pos2, r=6371e3):
    ''' positions given as (lat,lon), default is average earth radius'''
    pos1 = pos1 * np.pi / 180
    pos2 = pos2 * np.pi / 180
    cos_lat1 = np.cos(pos1[..., 0])
    cos_lat2 = np.cos(pos2[..., 0])
    cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
    cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))
# checked that it matches calculation from proj4/cartopy:
# Distances with cartopy
#from cartopy.geodesic import Geodesic
#geod = Geodesic() #default ellipsoid
#dutur = np.asarray(geod.inverse( (-70.769, -19.610), (-67.205542,-22.242005) )).T #distance and azimuths
# distance in meters, azimuth is degrees clockwise from north


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


def invert_osu(x, y, tj, guess):
    '''
    Invert OSU, fixing many parameters:
    t0, tj, T1, T2
    '''
    #osu(t,t0,x0,v,b,tj,s1,c1,s2,c2,T1,T2)
    #guess = [t0,x0,v,b,tj,s1,c1,s2,c2,T1,T2]
    t0 = x[0]
    tj = tj
    T1 = 1
    T2 = 0.5
    wrapper = lambda t,x0,v,b,s1,c1,s2,c2: osu(t,t0,x0,v,b,tj,s1,c1,s2,c2,T1,T2)
    popt, pcov = curve_fit(wrapper, x, y, guess)
    #print('Parameter stdev estimates from covariance:\nx0,v,b,s1,c1,s2,c2')
    #print(np.sqrt(np.diag(pcov))) #NOTE: seem too small...
    result = wrapper(x, *popt)
    residuals = y - result
    rmse = np.sqrt((np.sum(residuals**2) / residuals.size))

    return result, popt, rmse


def myfit(x, y, F, guess, printresult=False):
    '''
    Add column with best-fit information to dataframe
    NOTE: need to look into fixing parameters...
    '''
    popt, pcov = curve_fit(F, x, y, guess)
    result = F(x, *popt)
    residuals = y - result
    rmse = np.sqrt((np.sum(residuals**2) / residuals.size))

    if printresult:
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

    return result, popt, rmse

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
    fix t0 and tj in lambda function in myfit()
    #http://www.mathworks.com/matlabcentral/fileexchange/27783-fitting-data-with-a-sudden-discontinuity/content/html/exampleShift.html
    2 frequencies = 4 fourier series parameters sn, cn)
    # T1=365.25,T2=182.625
    '''
    w1 = (2 * np.pi) / T1
    w2 = (2 * np.pi) / T2
    t = t - t0
    tj = tj - t0
    f = ( x0 +
          v * t +
          b * heaviside(t-tj) +
          s1 * np.sin(w1*t) + c1 * np.cos(w1*t) + #)#
          s2 * np.sin(w2*t) + c2 * np.cos(w2*t) )

    return f



def osu_bak(t,t0,x0,v,b,tj,s1,c1,s2,c2,T1,T2):
    '''
    NOTE: as written, allows t0 and tj to vary... but we want to fix them!
    Preferred general model for OSU processing (See Bevis 2014 eq 5)
    #http://www.mathworks.com/matlabcentral/fileexchange/27783-fitting-data-with-a-sudden-discontinuity/content/html/exampleShift.html
    2 frequencies = 4 fourier series parameters sn, cn)
    # T1=365.25,T2=182.625
    '''
    w1 = (2 * np.pi) / T1
    w2 = (2 * np.pi) / T2
    t = t - t0
    tj = tj - t0
    f = ( x0 +
          v * t +
          b * heaviside(t-tj) +
          s1 * np.sin(w1*t) + c1 * np.cos(w1*t) + #)#
          s2 * np.sin(w2*t) + c2 * np.cos(w2*t) )

    return f
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
