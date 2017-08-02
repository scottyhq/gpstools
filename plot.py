"""
Figure generation functions for GPS time series
"""
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as pltdate



# ---------------------------------------------------------
#    Plotting Functions
# ---------------------------------------------------------
def components(df, dfSteps=None, dfMidas=None, dfFit=None,
             columns=['east', 'north','up'], axhline=False, title=''):
    #Plot daily positions
    fig, (ax,ax1,ax2) =  plt.subplots(3,1,sharex=True,figsize=(8.5,11))
    ax.plot(df.index, df[columns[0]], 'k.', label='EW')
    #ax.set_title('NS')

    ax1.plot(df.index, df[columns[1]], 'k.', label='NS')
    #ax1.set_title('EW')

    ax2.plot(df.index, df[columns[2]], 'k.', label='Z')

    ax.legend(loc='upper left',frameon=True)
    ax1.legend(loc='upper left',frameon=True)
    ax2.legend(loc='upper left',frameon=True)
    plt.suptitle(title, fontsize=16)
    ax1.set_ylabel('Position [m]')

    months = pltdate.MonthLocator()
    years = pltdate.YearLocator()
    for axes in (ax,ax1,ax2):
        axes.xaxis.set_major_locator(years)
        axes.xaxis.set_minor_locator(months) #too much
        axes.fmt_xdata = pltdate.DateFormatter('%Y-%m-%d')
        axes.grid(True)



def all(df, dfSteps=None, dfMidas=None, dfFit=None,
             columns=['east', 'north','up'], axhline=False, title=''):
    #Plot daily positions
    fig, (ax,ax1,ax2) =  plt.subplots(3,1,sharex=True,figsize=(8.5,11))
    ax.plot(df.index, df[columns[0]], 'k.', label='EW')
    #ax.set_title('NS')

    ax1.plot(df.index, df[columns[1]], 'k.', label='NS')
    #ax1.set_title('EW')

    ax2.plot(df.index, df[columns[2]], 'k.', label='Z')
    #ax2.set_title('Z')

    # Add MIDAS velocities
    if isinstance(dfMidas, pd.DataFrame):
        dfM = add_midas(df, dfMidas)
        ax.plot(dfM.index.values, dfM.midas_east.values, 'm-' , lw=2, label='MIDAS')
        ax1.plot(dfM.index.values, dfM.midas_north.values, 'm-', lw=2 )
        ax2.plot(dfM.index.values, dfM.midas_up.values, 'm-', lw=2 )

        # Show error bounds
        # NOTE: what exatly are MIDAS error bounds? note 95% confidence limits...
        #ax.fill_between(dfM.index.values, dfM.midas_east_lb.values, dfM.midas_east_ub.values, color='m', alpha=0.5)
        #ax1.fill_between(dfM.index.values, dfM.midas_north_lb.values, dfM.midas_north_ub.values, color='m', alpha=0.5)
        #ax2.fill_between(dfM.index.values, dfM.midas_up_lb.values, dfM.midas_up_ub.values, color='m', alpha=0.5)

    # Add discontinuities
    if isinstance(dfSteps, pd.DataFrame):
        for step in dfSteps.index.intersection(df.index):
            for axes in (ax,ax1,ax2):
                axes.axvline(step, color='k', linestyle='dashed')

    # Add Function Fits
    if isinstance(dfFit, pd.DataFrame):
        ax.plot(dfFit.index, dfFit.fit_east, 'c-' , lw=3, label='Fit')
        ax1.plot(dfFit.index.values, dfFit.fit_north.values, 'c-', lw=3)
        ax2.plot(dfFit.index.values, dfFit.fit_up.values, 'c-', lw=3)

    if axhline:
        for axes in (ax,ax1,ax2):
            axes.axhline(color='k',lw=1)

    ax.legend(loc='upper left',frameon=True)
    ax1.legend(loc='upper left',frameon=True)
    ax2.legend(loc='upper left',frameon=True)
    plt.suptitle(title, fontsize=16)
    ax1.set_ylabel('Position [m]')

    months = pltdate.MonthLocator()
    years = pltdate.YearLocator()
    for axes in (ax,ax1,ax2):
        axes.xaxis.set_major_locator(years)
        axes.xaxis.set_minor_locator(months) #too much
        axes.fmt_xdata = pltdate.DateFormatter('%Y-%m-%d')
        axes.grid(True)

    plt.tick_params(axis='x', which='minor', length=5, top=False, bottom=True)
    plt.tick_params(axis='x', which='major', length=10, top=False, bottom=True)
    fig.autofmt_xdate()
