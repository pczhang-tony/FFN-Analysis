# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 16:27:44 2018

Code for semi-automatic processing of FFN traces.

Traces were all extracted from Time Series Analyzer plugin in ImageJ.

The program will process the traces to extract information about FFN release
kinetics as well as vesicle type.

To enable dynamic plotting in Spyder, go to:
    
    Tools \ Preferences \ IPython console \ Graphics
    
    Under "Support for graphics (Matplotlib)", uncheck "Activate support"
    
    If this is not done, you will NOT see any plots during dynamic plotting


Versions of components (in case future updates break the program):
    Operating System: Windows 10
    Anaconda: 5.2.0
    Python: 3.6.6
    Spyder: 3.3.1
    IPython: 6.4.0
    
    matplotlib: 2.2.3
    numpy: 1.15.1
    scipy: 1.1.0
    pandas: 0.23.4
    pywt: 0.5.2
    lmfit: 0.9.11
    statsmodels: 0.9.0
    
    This program uses pickle protocol 4 to save to file

@author: Peng Cheng Zhang
"""

# Set backend (to enable dynamic plotting with multiprocessing in Spyder)
## See above comments to correctly configure IDE
import matplotlib
matplotlib.use('Qt5Agg')
# Import built-in libraries
import os # For file manipulation
import multiprocessing # For parallel programming (used for dynamic plotting)
# Import third-party packages
import pandas as pd # For handling dataframes
import numpy as np # For performing vector operations
import matplotlib.pyplot as plt # For plotting
from mpl_toolkits.axes_grid1.inset_locator import inset_axes # For creating insets
import pywt # For performing continuous wavelet transform
from statsmodels.stats import power, weightstats # For performing statsitical tests
from scipy import stats # Import stats module from scipy
from scipy import interpolate # For creating and manipulating BSpline objects
from scipy.optimize import brentq # For root solving
import lmfit # For non-linear least squares curve-fitting
import pickle # For saving object to file
from tkinter import Tk # To access clipboard

matplotlib.rcParams['figure.raise_window']=False # Set raise_window to False so that newly generated plots are in background windows

###############################################################################
####################### Functions for file manipulation #######################
###############################################################################

def get_files (path):
    '''
    Function to get list of file names
    
    Argument:
        path: string of directory where data files are located
        
    Returns:
        tuple of two lists (trace, measure):
            trace: list of 'Time trace(s)' files (containing the actual traces)
            measure: list of 'Measurement' files (containing the slice information)
    '''
    os.chdir(path) # Change directory to specified path
    files = os.listdir(path) # Generate of list with all file names
    timetrace = [] # Create empty list
    measure = [] # Create empty list
    
    for e in files:
        if 'Time Trace(s)' in e:
            timetrace.append(e)
        elif 'Measurement' in e:
            measure.append(e)
    return (timetrace, measure)

def matcher (measurement):
    '''
    From given measurement file, find its corresponding time trace file
    Argument:
        measurement: string, name of measurement file. **Assumes file name has format 'Measurement_##.csv'
        
    Returns:
        string, name of the corresponding timetrace file with format 'Time Trace(s)_##.csv'
    '''
    name = measurement.split('_')
    return 'Time Trace(s)_' + name[-1]

def get_slices (measurement):
    '''
    For a given measurement file, generate a pandas Series object with the Slice information for each ROI
    
    Argument:
        measurement: string, name of measurment file (.csv file)
    
    Returns:
        pandas Series object containing slices of each ROI, using ROI name as index
    '''
    slice_info = pd.read_csv (measurement, index_col = 0, usecols = [' ', 'Slice'], squeeze = True) # Generate a pandas Series object from measurement file. Using the first column as index, and the 'Slice' column as values
    columns = pd.read_csv(matcher(measurement), index_col = 0, nrows = 0).columns
    slice_info.index = columns[1:-2]
    return slice_info

def column_generator (timetrace):
    '''
    Generator that produces one column at a time from given timetrace file
    
    Arguments:
        timetrace: string. Name of csv file to process.
        
    Yields:
        pandas.Series object (one column of the timetrace file)
    '''
    columns = pd.read_csv(timetrace, index_col = 0, nrows = 0).columns
    for n in range(columns.size - 2):
        yield pd.read_csv(timetrace, index_col = 0, usecols = [0, n + 1], squeeze = True)

###############################################################################
######################## Function for dynamic plotting ########################
###############################################################################

def plot(datax,datay,Spikes,Slice,title = None,indices = None, flag = None):
    '''
    Versatile plot function, has three modes depending on the number of
    arguments passed in addtion to datax, datay, Spikes, Slice, and title:
        
        NO arguments or ONLY indices: simple plot showing general overview
                                      of trace data.
        
        ONLY flag: dynamic plot (via multiprocessing) for verification of
                   automatic spike detection
        
        indices & flag: dynamic plot (via multiprocessing) for verification
                        of automatic baseline detection
    
    
    Arguments:
        datax: pandas Series object, containing x values to be plotted
        datay: pandas Series object, containing y values to be plotted
        Spikes: pandas Series object, boolean array marking spike points
        Slice: int, ROI slice
        title: str, title of graph
        indices: multiprocessing.Array object containing four ints,
                 optional argument, only needed for dynamic plotting
        flag: multiprocessing.Value object, optional argument, only needed 
              for dynamic plotting
    '''
    datax = datax.copy()
    datay = datay.copy()
    if Spikes is not None:
        Spikes = Spikes.copy()
    # If the argument "flag" is given
    if flag is not None:
        plt.ion() # Turn on interactive mode (for dynamic plotting)
        fig,(ax,ax2) = plt.subplots(nrows=1,ncols=2,num=title,figsize=(16,9),clear=True)
    else:
        fig,ax = plt.subplots(nrows=1,ncols=1,num=title,figsize=(16,9),clear=True)

    fig.suptitle(title)
    ax.plot(datax,datay,'b.',label = 'Raw Data')
    ax.plot(datax.loc[Slice],datay.loc[Slice],'ko',label = 'ROI Slice')
    ax.set_xlim(left = datax.iloc[0],right = datax.iloc[-1]) # Set the limits of the x axis (start and end point of datax)
    ax.set_title(label = 'Overview')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('A.F.U.')
    # If the argument "flag" is not given, show plot
    if flag is None:
        ax.legend(loc='best') # Show figure legend
        plt.show()
    # Otherwise, if the argument "flag" is given
    else:
        ax2.plot(datay,'b.',label = 'Raw Data')
        ax2.plot(Slice,datay.loc[Slice],'ko',label='ROI Slice')
        ax2.set_xlabel('Index Number')
        # If indices are given, plot baselines for dynamic baseline verification
        if indices is not None:
            ax2.set_title(label = 'Baseline Detection')
            a,b,c,d = indices
            if np.isfinite(a):
                left = max(a - 80,datay.index[0])
            else:
                left = datay.index[0]
            if np.isfinite(d):
                right = min(d + 80,datay.index[-1])
            else:
                right = datay.index[-1]
            ax2.set_xlim(left = left, right = right) # Set limits of x axis
            # Define pre-event baseline segment
            pre = datay.loc[a:b]
            # Define post-event baseline segment
            post = datay.loc[c:d]
            # Try to find the index of event peak
            try:
                peakidx = datay.loc[a:d].idxmax()
            # If ValueError is raised, baseline was not determined, define peakidx as np.nan
            except ValueError:
                peakidx = np.nan
            # Plot pre-event baseline and related elements
            bl_pre, = ax2.plot(pre,'c.',label='Pre-event Baseline')
            bl_pre_main, = ax.plot(datax.loc[pre.index],pre,'c.',label = 'Pre-event Baseline') # Plot pre-event baseline on overview plot as well
            pre_est = pre.mean() # Estimate pre-event background
            pre_below = datay.loc[b+1:peakidx] < pre_est # Find the datapoints between pre-event baseline and peak that are below pre-event background
            pre_help_line, = ax2.plot([a,peakidx],[pre_est,pre_est],'c:',label = '_nolegend_') # Draw a line corresponding to estimated pre-event background
            pre_help, = ax2.plot(datay.loc[b+1:peakidx][pre_below],'g.',label = '_nolegend_') # Mark datapoints below the line in green
            try:
                pre_help_line_main, = ax.plot(datax.loc[[a,peakidx]],[pre_est,pre_est],'c:',label = '_nolegend_') # Draw line on overview plot as well
            except KeyError:
                pre_help_line_main, = ax.plot([np.nan,np.nan],[pre_est,pre_est],'c:',label='_nolegend_')
            pre_help_main, = ax.plot(datax.loc[b+1:peakidx][pre_below],datay.loc[b+1:peakidx][pre_below],'g.',label = '_nolegend_') # Mark datapoints on overview plot as well
            # Plot post-event baseline and related elements
            bl_post, = ax2.plot(post,'m.',label='Post-event Baseline')
            bl_post_main, = ax.plot(datax.loc[post.index],post,'m.',label='Post-event Baseline') # Plot post-event baseline on overview plot as well
            post_est = post.mean() # Estimate post-event background
            post_below = datay.loc[peakidx:c-1] < post_est # Find the datapoints between post-event baseline and peak that are below pre-event background
            post_help_line, = ax2.plot([peakidx,d],[post_est,post_est],'m:',label = '_nolegend_') # Draw a line corresponding to estimated post-event background
            post_help, = ax2.plot(datay.loc[peakidx:c-1][post_below],'r.',label = '_nolegend_') # Mark datapoints below the line in red
            try:
                post_help_line_main, = ax.plot(datax.loc[[peakidx,d]],[post_est,post_est],'m:',label='_nolegend_') # Draw line on overview plot as well
            except KeyError:
                post_help_line_main, = ax.plot([np.nan,np.nan],[post_est,post_est],'m:',label='_nolegend_')
            post_help_main, = ax.plot(datax.loc[peakidx:c-1][post_below],datay.loc[peakidx:c-1][post_below],'r.',label = '_nolegend_') # Mark datapoints on overview plot as well
        # Otherwise, if indices are not given, plot spikes for spike verification
        else:
            ax2.set_title(label = 'Spike Detection')
            left = max (datay.index[0],Slice - 80)
            right = min(datay.index[-1],Slice + 80)
            ax2.set_xlim(left = left, right = right)
            if np.any(Spikes):
                ax2.plot(datay[Spikes],'r.',label = 'Spikes')
                ax.plot(datax[Spikes],datay[Spikes],'r.',label = 'Spikes')

        # Pause (to display plot), automatically updates plot
        h,l = ax2.get_legend_handles_labels() # Get handles and labels from ax2
        fig.legend(h,l,loc='upper right') # Add legend to figure

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        return fig
        
        """
        # Start while loop (to dynamically show the plot)
        while flag.value:
            # If indices are given, dynamically update the pre-event and post-event baselines based on given input
            if indices is not None:
                # Save the old values
                old_a,old_b,old_c,old_d = [a,b,c,d]
                old_peakidx = peakidx
                # Get updated values from indices
                a,b,c,d = indices
                ## Update related elements
                # If a or d is updated
                if [a,d] != [old_a,old_d]:
                    # Update new peakidx
                    try:
                        peakidx = datay.loc[a:d].idxmax()
                    except ValueError:
                        peakidx = np.nan
                # If a or b is updated, OR peakidx is changed
                if [a,b] != [old_a,old_b] or peakidx != old_peakidx:
                    # If a or b is updated, update pre-event baseline and related elements
                    if [a,b] != [old_a,old_b]:
                        pre = datay.loc[a:b] # Update pre-event baseline segment
                        pre_est = pre.mean() # Update pre-event background
                        bl_pre.set_data(pre.index,pre.values) # Update plot
                        bl_pre_main.set_data(datax.loc[pre.index],pre) # Update overview plot
                    pre_help_line.set_data([a,peakidx],[pre_est,pre_est]) # Update line corresponding to pre-event background
                    try:
                        pre_help_line_main.set_data(datax.loc[[a,peakidx]],[pre_est,pre_est]) # Update line in overview plot
                    except KeyError:
                        pre_help_line_main.set_data([np.nan,np.nan],[pre_est,pre_est])
                    pre_below = datay.loc[b+1:peakidx] < pre_est # Find the datapoints between pre-event baseline and peak that are below new line
                    pre_help.set_data(datay.loc[b+1:peakidx][pre_below].index,datay.loc[b+1:peakidx][pre_below].values) # Update the plot
                    pre_help_main.set_data(datax.loc[b+1:peakidx][pre_below],datay.loc[b+1:peakidx][pre_below]) # Update overview plot
                    update_plot = True
                # If c or d is updated, OR peakidx is changed
                if [c,d] != [old_c,old_d] or peakidx != old_peakidx:
                    # If c or d is updated, update post-event baseline and related elements
                    if [c,d] != [old_c,old_d]:
                        post = datay.loc[c:d] # Update post-event baseline segment
                        post_est = post.mean() # update post-event background
                        bl_post.set_data(post.index,post.values) # Update plot
                        bl_post_main.set_data(datax.loc[post.index],post) # Update overview plot
                    post_help_line.set_data([peakidx,d],[post_est,post_est]) # Update line corresponding to post-event background
                    try:
                        post_help_line_main.set_data(datax.loc[[peakidx,d]],[post_est,post_est]) # Update line in overview plot
                    except KeyError:
                        post_help_line_main.set_data([np.nan,np.nan],[post_est,post_est])
                    post_below = datay.loc[peakidx:c-1] < post_est # Find the datapoints between peak and post-event baseline that are below the new line
                    post_help.set_data(datay.loc[peakidx:c-1][post_below].index,datay.loc[peakidx:c-1][post_below].values) # Update the plot
                    post_help_main.set_data(datax.loc[peakidx:c-1][post_below],datay.loc[peakidx:c-1][post_below]) # Update overview plot
            
            # Pause (to display plot), automatically updates plot
            h,l = ax2.get_legend_handles_labels() # Get handles and labels from ax2
            fig.legend(h,l,loc='upper right') # Add legend to figure
    #       plt.pause(0.5) # Show plot dynamically
            fig.canvas.draw_idle() 
            fig.canvas.start_event_loop(0.5) 
        """

###############################################################################
############# Functions for constructing models for curve fitting #############
###############################################################################

def lin_lin (x, T_max, bg1, alpha, delta, eta, iota):
    '''
    Function used to fit data. Linear rise phase, linear decay phase.
    It is a combination of four segments:
        When x < T_rise, a flat line equal to pre-event baseline
        When T_rise <= x < T_max, a straight line (linear rise phase)
        When T_max <= x < T_decay, a straight line (linear decay phase)
        When x >= T_decay, a flat line equal to post-event baseline
    
    Arguments:
        x: independent variable
        T_max: parameter, boundary point between rise phase and decay phase
        bg1: parameter, amplitude of pre-event basline
        alpha: parameter, difference between T_rise (boundary point between pre-event baseline and linear rise phase) and T_max, must be non-positive
        delta: parameter, difference between amplitude of signal peak and pre-event baseline, must be non-negative
        eta: parameter, difference between T_decay (boundary point between decay phase and post-event baseline) and T_max, must be non-negative
        iota: parameter, difference between post-event baseline and pre-event baseline, must be non-positive
    '''
    # Compute relevant parameters to construct function
    Peak = bg1 + delta
    T_rise = T_max + alpha
    T_decay = T_max + eta
    bg2 = bg1 + iota
    slope1 = - delta / alpha
    slope2 = (iota - delta) / eta
    # Construct separate x arrays for each segment (compatible with scalar inputs), ensuring that their values are within appropriate bounds
    x2 = x * ((T_rise <= x) & (x < T_max)) + T_rise * (x < T_rise) + T_max * (x >= T_max) # Ensures that T_rise <= x2 <= T_max (avoid NaNs during calculations)
    x3 = x * ((T_max <= x) & (x < T_decay)) + T_max * (x < T_max) + T_decay * (x >= T_decay) # Ensures that  T_max <=x3 <= T_decay (avoid NaNs during calculations)
    return bg1 * (x < T_rise) + (slope1 * (x2 - T_rise) + bg1)  * ((T_rise <= x) & (x < T_max)) + (slope2 * (x3 - T_max) + Peak) * ((T_max <= x) & (x < T_decay)) + bg2 * ( x >= T_decay)

def lin_1p (x, T_max, bg1, alpha, delta, k_slow, iota):
    '''
    Function used to fit data. Linear rise phase, one phase exponential decay.
    It is a combination of three segments:
        When x < T_rise, a flat line euqal to bg1 (pre-event baseline)
        When T_rise <= x < T_max, a straight line (linear rise phase)
        When x >= T_max, a one-phase exponential decay with final plateau at post-event baseline
    
    Arguments:
        x: independent variable
        T_max: parameter, boundary point between rise phase and decay phase
        bg1: parameter, amplitude of pre-event baseline
        alpha: parameter, difference between T_rise (boundary point between pre-event baseline and linear rise phase) and T_max, must be non-positive
        delta: parameter, difference between amplitude of signal peak and pre-event baseline, must be non-negative
        k_slow: parameter, rate constant of decay component
        iota: parameter, difference between post-event baseline and pre-event baseline, must be non-positive
    '''
    # Compute relevant parameters to construct function
    T_rise = T_max + alpha
    bg2 = bg1 + iota
    slope = - delta / alpha
    # Construct separate x arrays for each segment (compatible with scalar inputs), ensuring that their values are within appropriate bounds
    x2 = x * ((T_rise <= x) & (x < T_max)) + T_rise * (x < T_rise) + T_max * (x >= T_max) # Ensure that T_rise <= x2 <= T_max (avoid NaNs during calculations)
    x3 = x * (x >= T_max) + T_max * (x < T_max) # Ensures that x3 >= T_max (avoid NaNs during calculations)
    return bg1 * (x < T_rise) + (slope * (x2 - T_rise) + bg1) * ((T_rise <= x) & (x < T_max)) + ((delta - iota) * np.exp (-k_slow * (x3 - T_max)) + bg2) * (x >= T_max)

def lin_2p (x, T_max, bg1, alpha, delta, iota, percent_fast, k_fast, k_slow):
    '''
    Function used to fit data. Linear rise phase, two phase exopential decay.
    It is a combination of three segments:
        When x < T_rise, a flat line equal to bg1 (pre-event baseline)
        When T_rise <= x < T_max, a straight line (linear rise phase)
        When x >= T_max, a two-phase exponential decay with final plateau at bg2
    
    Arguments:
        x: independent variable
        T_max: parameter, boundary point between rise phase and decay phase
        bg1: parameter, specifies pre-event baseline
        alpha: parameter, difference between T_rise (boundary point between pre-event baseline and linear rise phase) and T_max, must be non-positive
        delta: parameter, difference between amplitude of signal peak and pre-event baseline, must be non-negative
        iota: parameter, difference between post-event baseline and pre-event baseline, must be non-positive
        percent_fast: parameter, specifies contribution of fast component in two-phase decay
        k_fast: parameter, specifies the rate of decay of fast component
        k_slow: parameter, specifies the rate of decay of slow component
    '''
    # Compute relevant parameters to construct function
    bg2 = bg1 + iota
    T_rise = T_max + alpha
    slope = - delta / alpha
    # Construct separate x arrays for each segment (compatible with scalar inputs), ensuring that their values are within appropriate bounds
    x2 = x * ((T_rise <= x) & (x < T_max)) + T_rise * (x < T_rise) + T_max * (x >= T_max) # Ensures that T_rise <= x2 <= T_max (avoid NaNs during calculations)
    x3 = x * (x >= T_max) + T_max * (x < T_max) # Ensures that x3 >= T_max (avoid NaNs during calculations)
    return bg1 * (x < T_rise) + (slope * (x2 - T_rise) + bg1) * ((T_rise <= x) & (x < T_max)) + ((delta - iota) * (percent_fast * np.exp (- k_fast * (x3 - T_max)) + (1 - percent_fast) * np.exp (-k_slow * (x3 - T_max))) + bg2) * (x >= T_max)

def exp_lin (x, T_max, bg1, delta, iota, eta, k_rise):
    '''
    Function used to fit data. Exponential rise phase, linear decay phase.
    It is a combination of two segments:
        When x < T_max, exponential rise with initial plateau at bg1
        When T_max <= x < T_decay, a straight line (linear decay phase)
        When x >= T_decay, a flat line (post event background)
        
    Arguments:
        x: independent variable
        T_max: parameter, boundary point between rise phase and decay phase
        bg1: parameter, specifies pre-event baseline
        delta: parameter, difference between amplitude of signal peak and pre-event baseline, must be non-negative
        iota: parameter, difference between post-event baseline and pre-event baseline, must be non-positive
        eta: parameter, difference between T_decay (boundary point between decay phase and post-event baseline) and T_max, must be non-negative
        k_rise: parameter, rate constant of rise component
    '''
    # Compute relevant parameters to construct function
    T_decay = T_max + eta
    bg2 = bg1 + iota
    Peak = bg1 + delta
    slope = (iota - delta) / eta
    # Construct separate x arrays for each segment (compatible with scalar inputs), ensuring that their values are within appropriate bounds
    x1 = x * (x < T_max) + T_max * (x >= T_max) # Ensures that x1 <= T_max (avoid NaNs during calculations)
    x2 = x * ((T_max <= x) & (x < T_decay)) + T_max * (x < T_max) + T_decay * (x >= T_decay) # Ensures that T_max <= x2 <= T_decay (avoid NaNs during calculations)
    return (delta * np.exp (k_rise * (x1 - T_max)) + bg1) * (x < T_max) + (slope * (x2 - T_max) + Peak) * ((T_max <= x) & (x < T_decay)) + bg2 * (x >= T_decay)

def exp_1p (x, T_max, bg1, delta, iota, k_slow, k_rise):
    '''
    Function used to fit data. Exponential rise phase, one phase exponential decay.
    It is a combination of two segments:
        When x < T_max, exponential rise with initial plateau at bg1
        When x >= T_max, a one-phase exponential decay with final plateau at bg2
        
    Arguments:
        x: independent variable
        T_max: parameter, boundary point between rise phase and decay phase
        bg1: parameter, specifies pre-event baseline
        delta: parameter, difference between amplitude of signal peak and pre-event baseline, must be non-negative
        iota: parameter, difference between post-event baseline and pre-event baseline, must be non-positive
        k_slow: parameter, rate constant of decay component
        k_rise: parameter, rate constant of rise component
    '''
    # Compute relevant parameters to construct function
    bg2 = bg1 + iota
    # Construct separate x arrays for each segment (compatible with scalar inputs), ensuring that their values are within appropriate bounds
    x1 = x * (x < T_max) + T_max * (x >= T_max) # Ensures that x1 <= T_max (avoid NaNs during calculations)
    x2 = x * (x >= T_max) + T_max * (x < T_max) # Ensures that x2 >= T_max (avoid NaNs during calculations)
    return (delta * np.exp(k_rise * (x1 - T_max)) + bg1) * (x < T_max) + ((delta - iota) * np.exp (-k_slow * (x2 - T_max)) + bg2) * (x >= T_max)

def exp_2p (x, T_max, bg1, delta, iota, percent_fast, k_slow, k_fast, k_rise):
    '''
    Function used to fit data. Exponential rise phase, two phase exponential decay.
    It is a combination of two segments:
        When x < T_max, exponential rise with initial plateau at bg1
        When x >= T_max, a two phase exponential decay with final plateau at bg2
        
    Arguments:
        x: independent variable
        T_max: parameter, boundary point between rise phase and decay phase
        bg1: parameter, specifies pre-event baseline
        delta: parameter, difference between amplitude of signal peak and pre-event baseline, must be non-negative
        iota: parameter, difference between post-event baseline and pre-event baseline, must be non-positive
        percent_fast: parameter, specifies contribution of fast component in two-phase decay
        k_fast: parameter, specifies the rate of decay of fast component
        k_slow: parameter, specifies the rate of decay of slow component
        k_rise: parameter, rate constant of rise component
    '''
    # Compute relevant parameters to construct function
    bg2 = bg1 + iota
    # Construct separate x arrays for each segment (compatible with scalar inputs), ensuring that their values are within appropriate bounds
    x1 = x * (x < T_max) + T_max * (x >= T_max) # Ensures that x1 <= T_max (avoid NaNs during calculations)
    x2 = x * (x >= T_max) + T_max * (x < T_max) # Ensures that x2 >= T_max (avoid NaNs during calculations)
    return (delta * np.exp (k_rise * (x1 - T_max)) + bg1) * (x < T_max) + ((delta - iota) * (percent_fast * np.exp (- k_fast * (x2 - T_max)) + (1 - percent_fast) * np.exp (-k_slow * (x2 - T_max))) + bg2) * (x >= T_max)

###############################################################################
################# Solvers to solve for x value given y value ##################
###############################################################################

def solver_linear (slope, x0, intercept, y):
    '''
    Given y value, solves the x value which satisfies:
        
        y = slope * (x - x0) + intercept
    
    
    Arguments:
        slope: float. Slope of line.
        intercept: float. Y intercept of line.
        x0: float. Position of X intercept
        y: float. Value for which we wish to find corresponding x value.
        
    Returns:
        float, the x value satisfying y = slope * (x - x0) + intercept
    '''
    return (y - intercept) / slope + x0

def solver_one_phase (span, x0, k, plateau, y):
    '''
    Given y value, solves the x value which satisfies:
        
        y = span * exp (- k * (x - x0)) + plateau
    
    
    Arguments:
        k: float. Rate constant.
        x0: float. Position of y asymptote.
        span: float. Span of exponential decay.
        plateau: float. Amplitude of x asymptote.
        y: float. Value for which we wish to find corresponding x value.
        
    Returns:
        float, the x value satisfying y = span * exp (- k* (x - x0)) + plateau
    '''
    return (-np.log((y - plateau) / span) / k) + x0

def solver_two_phase (span, x0, percent_fast, k_fast, k_slow, plateau, y):
    '''
    Given y value, solves the x value which satisfies:
        y = span * percent_fast * exp (- k_fast * (x - x0)) + span * (1 - percent_fast) * exp (-k_slow * (x - x0)) + plateau
    
    Arguments:
        percent_fast: float. Contribution of fast component.
        k_fast: float. Rate constant of fast component.
        k_slow: float. Rate constant of slow component.
        x0: float. Position of y asymptote.
        span: float. Span of exponential decay.
        plateau: float. Amplitude of x asymptote.
        y: float. Value for which we wish to find corresponding x value.
        
    Returns:
        float, the x value satisfying y = span * percent_fast * exp (- k_fast * (x - x0)) + span * (1 - percent_fast) * exp (-k_slow * (x - x0)) + plateau
    '''
    def function (x, span, x0, percent_fast, k_fast, k_slow, plateau, y):
        return span * (percent_fast * np.exp (- k_fast * (x - x0)) + (1 - percent_fast) * np.exp (-k_slow * (x - x0))) + plateau - y
    a = np.log(span / (y - plateau)) / k_fast + x0
    b = np.log(span / (y - plateau)) / k_slow + x0
    fa = function (a, span, x0, percent_fast, k_fast, k_slow, plateau, y)
    fb = function (b, span, x0, percent_fast, k_fast, k_slow, plateau, y)
    # Check that fa and fb have opposite signs
    if fa * fb < 0:
        return brentq (function, a, b, (span, x0, percent_fast, k_fast, k_slow, plateau, y))
    # If not, then either a or b is already very close to the actual root
    # The one which evaluates to least absolute value is taken as root
    else:
        if abs(fa) < abs(fb):
            return a
        else:
            return b

def solver_spline (spline, y):
    '''
    Given y value, solves the x value which satisfies:
        
        Spline_function(x) = y
        
    Spline_function is specified by knot vector, control points, and degree
    
    Arguments:
        spline: scipy BSpline object, the spline for which we wish to solve y.
        y: float, the value for which we wish to find corresponding x value.
        
    Returns:
        numpy array of floats, the x value(s) where the spline evaluates to y
    '''
    t,c,k = spline.tck # Get knot vector, control points, and polynomial degree from spline object
    c = c[:t.size - (k+1)] # Ensure that control points have the appropriate number of elements (Generating splines using the derivative method of BSpline objects introduces extra zeros at the end of control point array)
    c_adjust = c - y # Shift control points by y
    return solve_spline_roots(t,c_adjust,k)

###############################################################################
###############  Helper functions for finding roots of splines  ###############
###############################################################################

def average_knots (Knot_vec, degree):
    '''
    Helper function to generate average knots (for computing the control
    polygon)
    
    Arguments:
        Knot_vec: iterable, the knot vector defining basis splines
        degree: int, the degree of polynomial for each spline segment
        
    Returns:
        Knot_avg: numpy array containing the average knots (the corresponding x
                  values of each control point)
    '''
    # Construct a 2D calculation array to store values:
    # Knot_vec = [t0, t1, t2, ..., t_n-2, t_n-1, t_n]
    # Row 1: (row index = 0) [t1, t2, t3, ..., t_n-1, t_n, 0]
    # Row 2: (row index = 1) [t2, t3, t4, ..., t_n, 0, 0]
    # ...
    # Row degree (row index = degree - 1) [t_degree, t_order, t_degree + 2, ..., t_n, 0, ..., 0, 0]
    # Then take the averages along each column (axis = 0), and leave out the last degree + 1 elements
    Calc = np.zeros((degree,Knot_vec.size-degree-1),dtype = np.float_)
    for i in range(degree):
        Calc[i,:] = Knot_vec[i+1:-degree+i]
    return Calc.mean(axis = 0)

def insert_knot (x, Knot_vec, Ctrlp, degree):
    '''
    Helper function to correctly insert a knot point without changing the shape
    of the spline
    
    Arguments:
        x: float, new knot point to be inserted
        Knot_vec: numpy array, the knot vector into which the new knot will be
                  inserted
        Ctrlp: numpy array, the original control points of the spline
        degree: int, the degree of polynomial
        
    Returns:
        (k, c):
            k: numpy array, new knot vector with the inserted knot
            c: numpy array, adjusted control point array
    '''
    pre = Knot_vec[Knot_vec <= x] # Find the original knots that are no greater than x
    post = Knot_vec[Knot_vec > x] # Find the original knots that are greater than x
    # Insert knot value into knot vector
    k = np.append(pre,x)
    k = np.append(k,post)
    c = np.zeros(k.size - degree - 1) # Create numpy array of length len(k) - (degree + 1)
    # Compute new control points
    mu = pre.size # Find the number of knots <= x
    # Keep the first mu - degree knots
    c[:mu-degree] = Ctrlp[:mu-degree]
    # Alter the control point values near the inserted knot points
    mu_i = (x - Knot_vec[mu-degree:mu])/(Knot_vec[mu:mu+degree] - Knot_vec[mu-degree:mu])
    c[mu-degree:mu] = (1-mu_i) * Ctrlp[mu-degree-1:mu-1] + mu_i * Ctrlp[mu-degree:mu]
    # Shift the remaining control points into the remaining empty slots
    c[mu:] = Ctrlp[mu-1:]
    return (k,c)

def solve_spline_roots (Knot_vec,Ctrlp,degree,eps = 10**-15):
    '''
    Helper function to find all roots of a given spline
    
    Arguments:
        Knot_vec: numpy array, knot vector of spline
        Ctrlp: numpy array, control points of spline
        degree: int, polynomial degree
        eps: float, error tolerance, default value 10^-15
        max_iter: int, maximum number of iterations, default value 20
        
    Returns:
        numpy array containing all roots of spline (within defined region), if
        no roots, an empty array will be returned
    '''
    f = interpolate.BSpline(Knot_vec,Ctrlp,degree,extrapolate=False) # Construct callable BSpline object
    t = Knot_vec # Initialize t to original knot vector
    c = Ctrlp # Initialize c to original control points
    kavg = average_knots(t,degree) # generate average knot vector (the corresponding x values of the control points to construct control polygon)
    # Find control points where the sign changes
    c_idx = c >= 0 # Generate boolean array, "True" where control point is >=0, "False" otherwise (<0)
    sign_change = np.diff(c_idx.astype(int)) # Differentiate the boolean array as type int
    idxs = np.where(sign_change != 0)[0] # Positions where values are non-zero are right before a sign change
    # If there are no control point sign changes, return an empty array
    if idxs.size == 0:
        return np.array([])
    # Estimate zero points based on the contol points around sign change
    x = (c[idxs+1] * kavg[idxs] - c[idxs] * kavg[idxs+1]) / (c[idxs+1] - c[idxs])
    # Evaluate function values of x
    y = f(x)
    err_test = abs(y) > eps # Test whether function values are within error tolerance
    while err_test.any(): # While there are any function values that fail error tolerance test
        x_old = x.copy() # Store old values
        x_insert = x[err_test] # Mark x values whose function values fail error test for insertion
        # Insert each insertion point iteratively into the knot vector and update control points
        for pt in x_insert:
            t,c = insert_knot(pt,t,c,degree)
        kavg = average_knots(t,degree) # Update average knot vectors
        # Find control points where the sign changes
        c_idx = c >= 0 # Generate boolean array, "True" where control point is >=0, "False" otherwise (<0)
        sign_change = np.diff(c_idx.astype(int)) # Differentiate the boolean array as type int
        idxs = np.where(sign_change != 0)[0] # Positions where values are non-zero are right before a sign change
        # If there are no control point sign changes, return an empty array
        if idxs.size == 0:
            return np.array([])
        # Estimate zero points based on the contol points around sign change
        x = (c[idxs+1] * kavg[idxs] - c[idxs] * kavg[idxs+1]) / (c[idxs+1] - c[idxs])
        # If x_old and x have same dimensions, compare the two arrays
        if x_old.size == x.size:
            # If updated values offer no improvement, exit loop
            if np.all(x_old == x):
                break
        y = f(x)
        err_test = abs(y) > eps # Test whether function values are within error tolerance
    return x

###############################################################################
###############  Trace class, deals with individual FFN traces  ###############
###############################################################################

class trace(object):
    def __init__(self,x,y,Slice):
        '''
        Instantiating trace class requires the following:
            x values: pandas.Series object containing x values (timestamps)
            y values: pandas.Series object containing y values (raw data)
            Slice: int, the index at which the ROI was placed
        '''
        # Check that x and y contain the same number of elements
        if x.size != y.size:
            raise ValueError('x and y values must be the same length!')
        self.size = y.size # Define attribute "size" to store the number of data points
        self.x = x # Define attribute "x" to store x values
        self.y = y # Define attribute "y" to store y values
        self.Slice = Slice # Define attribute "Slice" to store Slice information
        self.name = y.name # Define attribute "name" to store name of trace
        self.index = self.x.index # Define attribute "index" to store index of data points
        self.exposure = x.diff().dropna().mean() # Define attribute "exposure" to store (average) exposure time per frame
        self.fs = 1 / self.exposure # Define attribute "fs" to store (average) sampling frequency
#        self.fs = 1000 / self.exposure
        self.remark = '' # Initialize remark attribute to empty string
        self.fit_method = None # Initialize fit_method attribute to NoneType
        
    def set_values(self,startidx=None,pre_endidx=None,post_startidx=None,endidx=None,spike=None,spike_start=None,spike_end=None,process=None,remark=None,ves_type=None,fit_method=None,**extras):
        '''
        Method for directly setting values to the following attributes:
            startidx: int, index of start of pre-event baseline
            pre_endidx: int, index of end of pre-event baseline
            post_startidx: int, index of start of post-event baseline
            endidx: int, index of end of post-event baseline
            spike: bool, whether there was a peak associated with event
            spike_start: int, index of start of event spike
            spike_end: int, index of end of event spike
            process: bool, whether trace should be further processed by least
                     squares fitting to extract kinetic parameters
            remark: str, remarks about the trace
            ves_type: str, vesicle type
            fit_method: str, specifies type of least squares fitting method for
                        extracting kinetic parameters
            **extras: for storing extra arguments when unpacking dictionaries
        '''
        # Set remark attribute if remark is given
        if remark is not None:
            self.remark = remark
        # Set startidx attribute if startidx is given
        if startidx is not None:
            # Try casting startidx as int, if applicable
            try:
                self.startidx = int(startidx)
            except ValueError:
                pass
        # Set pre_endidx attribute if pre_endidx is given
        if pre_endidx is not None:
            # Try casting pre_endidx as int, if applicable
            try:
                self.pre_endidx = int(pre_endidx)
            except ValueError:
                pass
        # Set post_startidx attribute if post_startidx is given
        if post_startidx is not None:
            # Try casting post_startidx as int, if applicable
            try:
                self.post_startidx = int(post_startidx)
            except ValueError:
                pass
        # Set endidx attribute if endidx is given
        if endidx is not None:
            # Try casting endidx as int, if applicable
            try:
                self.endidx = int(endidx)
            except ValueError:
                pass
        # Set spike attribute if spike is given
        if spike is not None:
            # If input type is bool
            if type(spike) == bool:
                self.spike = spike
            # If input type is str, accept 'True' as True and 'False' as False
            elif type(spike) == str:
                if spike == 'True':
                    self.spike = True
                elif spike == 'False':
                    self.spike = False
        # Set spike_start attribute if spike_start is given
        if spike_start is not None:
            # Try casting spike_start as int, if applicable
            try:
                self.spike_start = spike_start
            except ValueError:
                pass
        # Set spike_end attribute if spike_end is given
        if spike_end is not None:
            # Try casting spike_end as int, if applicable
            try:
                self.spike_end = spike_end
            except ValueError:
                pass
        # Set process attribute if process is given
        if process is not None:
            # If input type is bool
            if type(process) == bool:
                self.process = process
            # If input type is str, accept 'True' as True and 'False' as False
            elif type(process) == str:
                if process == 'True':
                    self.process = True
                elif process == 'False':
                    self.process = False
        # Set ves_type attribute if ves_type is given
        if ves_type is not None:
            self.ves_type = ves_type
        # Set fit_method attribute if fit_method is given
        if fit_method is not None:
            # Try to check if fit_method is np.nan (empty)
            try:
                np.isnan(fit_method)
            # If fit_method is not empty, it is a string, which will raise a TypeError when np.isnan is applied
            except TypeError:
                self.fit_method = fit_method
                # When fit_method is a string, the pre- and post-event baselines are defined
                self.pre = self.y.loc[self.startidx:self.pre_endidx]
                self.post = self.y.loc[self.post_startidx:self.endidx]
        
    def detect_spikes(self,wname,Wid = [1,100], Ns = 50, L = 0, out = False):
        '''
        Function to detect spikes using continuous wavelet transform.
        Based on methods developed in:
        
        Nenadic Z and Burdick JW, Spike detection using the continuous wavelet
        transform, IEEE Trans. Biomed. Eng., vol.52, issue 1, pp. 74-87, 2005.
        
        Adapted from original matlab code.
        
        Arguments:
        wname: string. The name of wavelet used for fitting.
               Relevant options: 'gaus1', first derivative of Gaussian wavelet.
                                 'mexh', negative second derivative of Gaussian wavelet.
        Wid: length 2 iterable of expected minimum (first element) and maximum 
             (second element) width [# of frames] of spike to be detected.
             Wid = [Wmin, Wmax]. (Default: Wmin = 1, Wmax = 100)
        Ns: Int. The number of scales to use in detection (Nx >=2)
            (Default value 50)
        L: Float. Factor that multiplies [cost of comission] / [cost of omission]
            -0.2 <= L <= 0.2
            For unsupervised detection, the suggested value of L is close to 0.
            (Default value is 0)
        out: bool, whether to return results, default set to "False"
            
        Returns:
            Spikes: boolean array marking data points judged to be part of a spike
        '''
        Nt = (self.y).size # Get the number of data points
        cor_data = self.y - (self.y).mean() # Make sure signal is zero-mean
        # The following block of code is used for determining the scale for use with continuous wavelet transform
        Signal = np.zeros(1000)
        
        if wname == 'gaus1':
            Signal[374] = 1
            Signal[624] = -1
        
        elif wname == 'mexh':
            Signal[499] = 1
            Signal[333] = -0.4
            Signal[666] = -0.4
            
        Width = np.linspace(Wid[0], Wid[1], Ns) # Create a linespace vector "Width" spaced evenly between minimum and maximum
        
        Eps = 10**(-15) # Ifninitesimally small number
        
        Scales = np.linspace(2, 24, 23) # Generate Scales array [2, 3, 4, ... , 24] (24 = 8 * 3, 8 is from 800 Hz)
        WidthTable = np.zeros(Scales.size) # Generate zero vector with same length as Scales
        
        c, f = pywt.cwt(Signal, Scales, wname)
        for i in range(Scales.size):
            IndPos = (c[i] > 0).astype(int) # Indicators of positive coefficients
            IndDer = np.diff(IndPos) # indicators of derivative
            IndZeroCross = (IndDer == -1).ravel().nonzero()[0] # Find the non-zero indicators and make into array
            Ind_max = min(IndZeroCross[IndZeroCross > 499]) + 1
            Ind_min = max(IndZeroCross[IndZeroCross < 499])
            WidthTable[i] = Ind_max - Ind_min # 0.125 = 1/8
        
        WidthTable += (Scales - 1)*Eps
        Scale = np.around(np.interp(Width, WidthTable, Scales))
        
        ct = np.zeros((Ns, Nt)) # Create numpy array of zeros with dimensions Ns * Nt (matrix of thresholded coefficients)
        
        coef, freq = pywt.cwt(cor_data, Scale, wname) # Get all coefficients for data, using scale calculated above
        
        L = L * 36.7368 # 36.7368 is the maximum allowed by current machine precision
        
        I0 = np.zeros(Nt) # Initialize the vector of spike indicators, 0-no spike, 1-spike
        
        
        for i in range(Ns): # For each row of the resulting coefficient matrix
            Sigmaj = np.median(abs(coef[i, ::int(Scale[i])] - coef[i].mean())) / 0.6745 # Take only coefficients spaced Scale[i] apart
            Thetaj = Sigmaj * ((2 * np.log(Nt))**0.5) # Compute hard threshold
            index = (abs(coef[i]) > Thetaj).ravel().nonzero()[0] # Find indices of coefficients greater than Thetaj threshold
            if index.size == 0:
                Mj = Thetaj # If no coefficients are greater than threshold, assume at least one spike
                PS = 1/Nt
                PN = 1 - PS
            else:
                Mj = abs(coef[i, index]).mean()
                PS = index.size/Nt
                PN = 1- PS
            
            DTh = Mj / 2 + Sigmaj * Sigmaj / Mj * (L + np.log(PN/PS)) # Compute decision threshold
            DTh = abs(DTh) * (DTh >= 0) # Make DTh >=0
            ind = (abs(coef[i]) > DTh).ravel().nonzero()[0]
            ct[i, ind] = coef[i, ind]
            Index = (ct[i] != 0) # Find non-zero indices
            Index = Index + I0 - Index*I0 # Combine common elements from Index and I0 vectors
            I0 = Index # Set I0 to the current Index vector, for use in next iteration
        
        self.Spikes = pd.Series(Index.astype(bool), index = self.index)
        
        if out:
            return self.Spikes
    
    def parse_spikes(self,block = 9,out = False):
        '''
        Helper method to parse the Spikes generated by detect_spikes method
        and judge whether an event has a spike or not
        
        Argument:
            block: int, specifies size of window region. Default value 9.
                   (window length = 2 * block + 1)
            out: bool, whether to return results, default set to "False"
            
        Returns:
            (spike,spike_start,spike_end):
                spike: bool, True if a spike is judged to be associated with
                       the event
                spike_start: int, index corresponding to start of event spike
                spike_end: int, index corresponding to end of event spike
        '''
        center = self.Slice
        while center <= self.Slice + block: # Allow the center frame to drift between Slice and block
            start = max(center - block, 1)
            stop = min(center + block, self.Spikes.size)
            if not self.Spikes.loc[start:stop].any():
                center += 1
            else:
                break
        # If there are no "1"s in Index
        if not self.Spikes.astype(int).loc[start:stop].any():
            self.spike = False
            self.spike_start = np.nan
            self.spike_end = np.nan
            self.remark += 'Detected no spike associated with event; ' # Update remark
        else:
            pos = self.Spikes.astype(int).to_numpy().nonzero()[0]+1 # Get list of non-zero positions within the interval
            left = pos[pos >= start].min() # Get left time-point
            right = pos[pos <= stop].max() # Get right time-point
            temp = self.Spikes.astype(int).diff() # Get first-order derivative of Index. There will be 1 followed by -1 for each spike
            rise = (temp == 1).to_numpy().nonzero()[0] # Find positions of "1"s in temp
            fall = (temp == -1).to_numpy().nonzero()[0] # Find positions of "-1"s in temp
            initium = max(rise[rise <= left]) # Find the maximum position of "1" where it is smaller or equal to left edge of interval
            postremum = min(fall[fall >= right]) # Find the mimum position of "-1" where it is greater or equal to right edge of interval
            if postremum - initium <= 2:
                self.spike = False
                self.spike_start = np.nan
                self.spike_end = np.nan
                self.remark += 'Detected no spike associated with event; ' # Update remark
            else:
                self.spike = True
                self.spike_start = initium
                self.spike_end = postremum
                self.remark += f'Detected a spike associated with event, starting index: {self.spike_start}, ending index: {self.spike_end}; ' # Update remark
        # If out set to "True"
        if out:
            return (self.spike,self.spike_start,self.spike_end)
    
    def detect_flats(self,window,alpha = 0.05, beta = 0.2,out = False):
        '''
        Method for detecting regions of flatness
        
        Arguments:
            window: int, window length for scanning
            alpha: float, type I error rate, 0<alpha<1, default value 0.05
            beta: float, type II error rate, 0<beta<1, default value 0.2
            out: bool, whether to return results. Default value "False"
        
        Returns:
            pandas.Series object containing boolean values, True indicating that
            the corresponding position in the original trace is considered flat
        '''
        roll = self.y.rolling(window = window)
        avg = roll.mean()
        diffs = avg.diff(periods = window)
        std = roll.std()
        shift = np.r_[np.ones(window)*np.nan,std]
        index_shift = np.r_[self.index,np.arange(self.size+1,self.size+window+1,1)]
        std_shift = pd.Series(shift,index=index_shift).iloc[0:self.size]
        std_pool = np.sqrt((std**2 + std_shift**2)/2)
        # Compute bounds for equivalence test
        d = power.tt_ind_solve_power(nobs1 = window, alpha=alpha, power= 1-beta)
        bounds = d * std_pool * (2*window - 2.25)/(2*window - 3)*np.sqrt(2*window/(2*window-2)) # Correct for small sample size
        # Calculate t statistics
        #Equivalency test: Null Hypotheses: Mean1 - Mean2 < -bounds or Mean1- Mean2 > bound
        t_upper = ((diffs - bounds)*np.sqrt(2*window)/std_pool).dropna()
        t_lower = ((diffs + bounds)*np.sqrt(2*window)/std_pool).dropna()
        
        p_upper = stats.t.cdf(t_upper,df=2*(window-1))
        p_lower = stats.t.sf(t_lower,df=2*(window-1))
        
        p_vals = np.maximum(p_upper,p_lower)
        
        ends = pd.Series(np.r_[np.ones(self.size-p_vals.size)*np.nan,p_vals],index=self.index)
        
        # Mark positions where p value is less than alpha as "True"
        self.Flats = ends<alpha
        
        if out:
            return self.Flats
    
    def parse_flats(self,window, mode,interval = 100, alpha = 0.05, beta = 0.2, out = False):
        '''
        Method for processing self.flats to identify pre-event and post-event
        baselines, as well as vesicle type
        
        Arguments:
            window: int, should be the same value as the "window" value used to
                    compute flats (from baseline_scan)
            mode: string, specifies mode of anlaysis
            interval: int, specifies window length used to compute "density" of flat regions
            alpha: float, type I error rate, 0<alpha<1, default value 0.05
            beta: float, type II error rate, 0<beta<1, default value 0.2
            out: bool, return results if True. Default is "False"
                    
        Returns:
            (startidx,pre_endidx,Slice,post_startidx,endidx)
                startidx: int, index position of the start of event
                pre_endidx: int, index position of the end of pre-event baseline
                Slice: int, same as input
                post_startidx: int, index position of the start of post-event baseline
                endidx: int, index position of the end of event
        '''
        # Initialize results
        self.startidx,self.pre_endidx,self.post_startidx,self.endidx = (np.nan,np.nan,np.nan,np.nan)
        # Make copy of self.flats
        flats_info = self.Flats.copy()
        # Make sure that the first and last element in flats_info are "False"
        flats_info.iloc[0] = False
        flats_info.iloc[-1] = False
        # Take the rolling mean of boolean Series "flats" to compute the "density" of flat regions in a given interval
        roll = flats_info.astype(int).rolling(window = interval, center = True)
        density = roll.mean()
        # Only consider the regions which have a density of 95% or higher
        regions = density >= 0.95
        # Perform discrete differentiation on regions to find start and end points of contiguous intervals
        posit = regions.astype(int).diff()
        starts = posit[posit == 1].index # Values 1 mark starting points
        ends = posit[posit == -1].index # Values -1 mark ending points
        length = ends - starts
        valid = length > np.ceil(interval/2)
        valid_s = starts[valid] + int(np.ceil(interval*0.05))
        valid_e = ends[valid] - int(np.ceil(interval*0.05))-1

        # For fast traces:
        if mode == 'fast':
            # Find start and end positions of flat segments prior to Slice
            pre_ends = valid_e[valid_e < self.Slice]
            pre_starts = valid_s[valid_s < self.Slice]
            # Find start and end positions of flat segments after Slice
            post_starts = valid_s[valid_s > self.Slice]
            # If pre_ends is not empty
            if pre_ends.size:
                if pre_starts[-1]-50 > 4000: # Make sure that it is post-stimulus
                    self.pre_endidx = pre_ends[-1] # Find the largest end-point smaller than Slice
                    self.startidx = valid_s[np.argmax(valid_e == self.pre_endidx)] # Find the start-point corresponding to the end-point in previous step
                    self.pre = self.y.loc[self.startidx:self.pre_endidx]
                    self.remark += f'Pre-event baseline detected, starting index: {self.startidx}, ending index: {self.pre_endidx}; ' # Update remark attribute
            # If post_starts is not empty
            if post_starts.size:
                self.post_startidx = post_starts[0] # Find the smallest start-point greater than Slice
                self.endidx = valid_e[np.argmax(valid_s == self.post_startidx)] # Find the end-point corresponding to the start-point in previous step
                self.post = self.y.loc[self.post_startidx:self.endidx]
                self.remark += f'Post-event baseline detected, starting index: {self.post_startidx}, ending index: {self.endidx}; ' # Update remark attribute
        
        elif mode == 'slow':
            if valid.sum() > 3:
                i = 0
                j = i + 1
                # Scan for pre-event baseline: 
                while j in range(valid.sum()):
                    # If region is before Slice
                    if 4500 < valid_e[i] < self.Slice:
                        # Determine the bounds for the two regions under consideration
                        pre = self.y.loc[valid_s[i]:valid_e[i]]
                        follow = self.y.loc[valid_s[j]:valid_e[j]]
                        pool_std = np.sqrt((pre.std()**2 * (pre.size-1) + follow.std()**2 * (follow.size -1)) / (pre.size + follow.size -2))
                        d = power.tt_ind_solve_power(nobs1 = pre.size, alpha=alpha, power= 1-beta, ratio = follow.size/pre.size)
                        bound = d * pool_std * (pre.size + follow.size - 2.25)/(pre.size + follow.size - 3)*np.sqrt((pre.size + follow.size)/(pre.size + follow.size - 2)) # Correct for small sample size
                        # Test for equivalency by two one-sided T-tests
                        p_v = weightstats.ttost_ind(pre,follow,-bound,bound)[0]
                        # If results are significant (the two regions are equivalent)
                        if p_v <= alpha:
                            # If the second segment is before Slice
                            if valid_e[j] < self.Slice:
                                # If the mean of second segment is smaller
                                if follow.mean() < pre.mean():
                                    # Reset the starting point (Make the second segment the first segment, and move on the scan the next segment, repeating the above process)
                                    i = j
                                    j = i + 1
                                # Otherwise (if the mean of the second segment is larger), keep the first segment and compare against the next segment
                                else:
                                    j += 1
                            # Otherwise if the second segment is after Slice
                            else:
                                # Keep the first segment, and compare agains the next segment
                                j += 1
                        # If the results are not significant (the two regions are not equivalent)
                        else:
                            # If the mean of the second segment is smaller
                            if follow.mean() < pre.mean():
                                # If the second segment is before Slice
                                if valid_e[j] <= self.Slice:
                                    # Reset the starting point (Make the second segment the first segment and repeat scanning process)
                                    i = j
                                    j = i + 1
                                # Otherwise (if the second segment is after Slice)
                                else:
                                    # Define the first segment as pre-event baseline
                                    self.startidx = valid_s[i]
                                    self.pre_endidx = valid_e[i]
                                    break # Exit loop
                            # Otherwise, if the mean of the second segment is larger
                            else:
                                # Keep the first segemnt, compare against the next segment
                                j+=1
                    # Otherwise, if the first segment is before stimulus, slide along
                    elif valid_e[i] <= 4500:
                        i = j
                        j = i + 1
                    # Otherwise, if the first segment is after stimulus
                    else:
                        break # Exit loop
            # If pre-event baseline was identified
            if np.isfinite(self.startidx * self.pre_endidx):
                # Store pre-event baseline data in attribute named "pre"
                self.pre = self.y.loc[self.startidx:self.pre_endidx]
                self.remark += f'Pre-event baseline detected, starting index: {self.startidx}, ending index: {self.pre_endidx}; ' # Update remark attribute
                # Define a third segment to compare with the second segment from above
                m = j + 1
                # Scan for post-event baseline
                while m in range(valid.sum()): # Lee: should be while m < valid.sum()
                    # Determine bounds for testing statistical equivalence
                    follow = self.y.loc[valid_s[j]:valid_e[j]]
                    post = self.y.loc[valid_s[m]:valid_e[m]]
                    # Lee: Should write a function to perform equivalence test
                    pool_std = np.sqrt((post.std()**2 * (post.size-1)+ follow.std()**2 * (follow.size - 1)) / (post.size + follow.size - 2))
                    d = power.tt_ind_solve_power (nobs1 = follow.size, alpha = alpha, power = 1-beta, ratio = post.size/follow.size)
                    bound = d * pool_std * (post.size + follow.size - 2.25)/(post.size + follow.size - 3)*np.sqrt((post.size + follow.size)/(post.size + follow.size - 2)) # Correct for small sample size
                    # Test for equivalence using two one-sided T-tests
                    p_v = weightstats.ttost_ind(follow,post,-bound,bound)[0]
                    # If results are significant (the segments are statistically equivalent)
                    if p_v <= alpha:
                        # If the mean of the third segment is larger
                        if post.mean() > follow.mean():
                            # Define the third segment as post event baseline
                            self.post_startidx = valid_s[m]
                            self.endidx = valid_e[m]
                            break # Exit loop
                        # Otherwise, consider the next segment
                        else:
                            m += 1
                    # Otherwise, if the results are not significant (the two segments are not statistically equivalent)
                    else:
                        # If the mean of the third segment is smaller
                        if follow.mean() > post.mean():
                            # Reset the second segment to the third segment, and compare against the next segment
                            j = m
                            m = j + 1
                        # Otherwise, if the mean of the third segment is larger
                        else:
                            # Define the region between second and third segment
                            inter = self.y.loc[valid_e[j] + 1: valid_s[m] - 1]
                            # If the lowest 25% quantile of the inter segment is larger than that of the second segment
                            if inter.quantile(q=0.25,interpolation = 'lower') >= follow.quantile(q=0.25,interpolation = 'lower'):
                                # Define second segment as post-event baseline
                                self.post_startidx = valid_s[j]
                                self.endidx = valid_e[j]
                                break # exit while loop, post event baseline undetermined
                            # Otherwise, if the 25% quantile of the inter segment is smaller
                            else:
                                break # post event baseline is disturbed, no post event baseline
            # If post-event baseline was identified
            if np.isfinite(self.post_startidx * self.endidx):
                # Store post-event baseline data in attribute named "post"
                self.post = self.y.loc[self.post_startidx:self.endidx]
                self.remark += f'Post-event baseline detected, starting index: {self.post_startidx}, ending index: {self.endidx}; ' # Update remark attribute
        if np.isnan(self.startidx * self.pre_endidx):
            self.remark += 'Pre-event baseline not found; '
        if np.isnan(self.post_startidx * self.endidx):
            self.remark += 'Post-event baseline not found; '
        # If out set to "True"
        if out:
            return (self.startidx,self.pre_endidx,self.post_startidx,self.endidx)
    
    def parse_vestype(self,out = False):
        '''
        Helper method to detect vesicle type
        
        Argument:
            out: bool, if "True", returns result
        
        Returns:
            (ves_type,process):
                ves_type: str, type of vesicle
                process: bool, True if data is to be processed to extract parameters
        '''
        # Initialize attributes
        self.ves_type = 'Not Determined'
        self.process = False
        # If both start and stop have values
        if (not np.isnan(self.startidx)) and (not np.isnan(self.endidx)):
            if weightstats.ttest_ind(self.pre,self.post)[1] < 0.05:
                self.ves_type = 'Docked'
            else:
                self.ves_type = 'New Arrival'
            self.process= True
        # If out set to "True"
        if out:
            return (self.ves_type,self.process)
    
    def plot(self):
        '''
        Simple plot method to get general overview of data shape
        '''
        fig,ax = plt.subplots(nrows=1,ncols=1,num=self.name,figsize=(16,9),clear=True)
        fig.suptitle(self.name)
        ax.set_title('Overview')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('A.F.U.')
        ax.set_xlim(left = self.x.iloc[0],right = self.x.iloc[-1])
        ax.plot(self.x,self.y,'b.',label = 'Raw Data')
        ax.plot(self.x.loc[self.Slice],self.y.loc[self.Slice],'ko',label = 'ROI Slice')
        plt.show()

    def plot_features(self,save_plot = False,show_plot = True,rasterize = True,dpi = 150,out = False,save_path = None):
        '''
        Method for plotting trace information, including event features:
            Slice
            Baseline (pre & post event, where applicable)
            Spikes
        
        Resulting plot will contain three subplots:
            top subplot will show overall feature of the curve, including all 
            features
            
            middle subplot will show spike information (if spike is detected), 
            or more detailed information of the region near Slice (if no spike
            is detected)
            
            bottom subplot will show baseline information (if baseline was
            detected), or region surrounding Slice (if no baseline is detected)
        
        Arguments:
            save_plot: bool, if "True", saves plot to file.
                       Default value "False"
            show_plot: bool, if "True", shows plot. Default value "True"
            rasterize: bool, if "True", rasterizes image. Default value "True"
            dpi: int, image resolution
            out: bool, if "True", returns figure object. Default value "False"
            save_path: (optional) str, file path of file to be saved. If not
                       given, defaults to the current working directory
            
        Returns:
            subplot: (fig1, (ax1,ax2,ax3))
        '''
        fig, (ax1,ax2,ax3) = plt.subplots(nrows = 1, ncols = 3,sharey = True,num = self.name,figsize = (16,9),dpi = dpi,clear = True)
        # Set figure title
        fig.suptitle(t = self.name)
        ## Plot Axis 1 (top subplot)
        # Set x-axis bounds
        ax1.set_xlim(left = self.x.iloc[0],right = self.x.iloc[-1]) # Set the limits of the x axis (start and end point of self.x)
        ax1.set_xlabel(xlabel = 'Time (s)') # Set x axis label
        ax1.set_ylabel(ylabel = 'A.F.U.') # Set y axis label
        ax1.set_title(label = 'Raw Data') # Set (sub)plot title
        # Plot the entire trace
        ax1.plot(self.x,self.y,'b.',rasterized = rasterize, label = 'Raw data')
        # Mark positions of spikes if any
        if self.Spikes.any():
            ax1.plot(self.x[self.Spikes],self.y[self.Spikes],'r.',rasterized = rasterize, label = 'Spikes')
        # Mark position of Slice
        ax1.plot(self.x.loc[self.Slice],self.y.loc[self.Slice],'ko',rasterized = rasterize, label = 'ROI Slice')
        # Show associated spike (if applicable)
        try:
            ax1.plot(self.x.loc[[self.spike_start,self.spike_end]],self.y.loc[[self.spike_start,self.spike_end]],'ro',rasterized = rasterize, label = 'Event Spike')
        except KeyError:
            pass
        # Highlight pre-event baseline if it was determined
        try:
            ax1.plot(self.x.loc[self.startidx:self.pre_endidx],self.pre,'c.',rasterized = rasterize, label = 'Pre-Event Baseline')
        except AttributeError:
            pass
        # Highlight post-event baseline if it was determined
        try:
            ax1.plot(self.x.loc[self.post_startidx:self.endidx],self.post,'m.',rasterized = rasterize, label = 'Post-Event Baseline')
        except AttributeError:
            pass
        
        ## Plot Axis 2 (middle subplot)
        # Set subplot title and x- & y-axis labels
        ax2.set_title(label = 'Spike Detection')
        ax2.set_xlabel(xlabel = 'Index No.')
        ax2.set_ylabel(ylabel = 'A.F.U.')
        left = max(self.index[0],self.Slice - self.fs//10) # Set left boundary point ~ 800 points to the left of Slice (within bounds of index)
        right = min(self.Slice + self.fs//10,self.index[-1]) # Set right boundary point ~ 80 points to the right of ending index (within bounds of index)
        ax2.set_xlim(left = left, right = right)
        ax2.plot(self.y,'b.',rasterized = rasterize, label = 'Raw Data')
        # Mark positions of spikes if any are within plot range
        if self.Spikes.loc[left:right].any():
            ax2.plot(self.y[self.Spikes],'r.',rasterized = rasterize, label = 'Spikes')
        # Mark position of Slice
        ax2.plot(self.Slice,self.y.loc[self.Slice],'ko',rasterized = rasterize, label = 'ROI Slice')
        # Show associated spike (if applicable)
        try:
            ax2.plot(self.y.loc[[self.spike_start,self.spike_end]],'ro',rasterized = rasterize, label = 'Event Spike')
        except KeyError:
            pass
        
        ## Plot Axis 3 (bottom subplot)
        ax3.set_title(label = 'Baseline Detection')
        ax3.set_xlabel(xlabel = 'Index No.')
        ax3.set_ylabel(ylabel = 'A.F.U.')
        # If startidx is NaN (No pre-event baseline)
        if np.isnan(self.startidx):
            left = max(self.index[0],self.Slice - self.fs//1) # Set left boundary point ~ 800 points to the left of Slice (within bounds of index)
        # Otherwise
        else:
            left = max(self.index[0],self.startidx - self.fs//10) # Set left boundary point ~ 80 points to the left of starting index (within bounds of index)
        # If endidx is NaN (No post-event baseline)
        if np.isnan(self.endidx):
            right = min(self.Slice + self.fs//1,self.index[-1]) # Set right boundary point ~ 800 points to the right of Slice (within bounds of index)
        # Otherwise
        else:
            right = min(self.endidx + self.fs//10,self.index[-1]) # Set right boundary point ~ 80 points to the right of ending index (within bounds of index)
        ax3.set_xlim(left = left, right = right)
        ax3.plot(self.y,'b.',rasterized = rasterize, label = 'Raw Data')
        # Mark position of Slice
        ax3.plot(self.Slice,self.y.loc[self.Slice],'ko',rasterized = rasterize, label = 'ROI Slice')
        # Highlight pre-event baseline if it was determined
        try:
            ax3.plot(self.pre,'c.',rasterized = rasterize, label = 'Pre-Event Baseline')
        except AttributeError:
            pass
        # Highlight post-event baseline if it was determined
        try:
            ax3.plot(self.post,'m.',rasterized = rasterize, label = 'Post-Event Baseline')
        except AttributeError:
            pass
        
        ## Construct figure legend (shared across entire figure)
        handles,labels = ax1.get_legend_handles_labels() # Get handles & lables from the top subplot (contains everything)
        fig.legend(handles,labels,loc='lower right')
        # If out is set to "True", return subplots
        if out:
            return (fig, (ax1,ax2,ax3))
        # If save_plot is "True", save plot to pdf file
        if save_plot:
            # If save_path is not given, save file in current directory
            if save_path is None:
                fig.savefig(f'{self.name}.pdf', dpi = 'figure', format = 'pdf')
            # Otherwise, save file in designated path
            else:
                fig.savefig(f'{save_path}\\{self.name}.pdf', dpi = 'figure', format = 'pdf')
        # Show figure if show_plot is set to "True"
        if show_plot:
            fig.show()
        # If show_plot is set to "False", close figure (release from memory)
        else:
            plt.close(fig)
        
        
    def pre_process(self, out = False):
        '''
        Method to pre-process the trace for least-squares fitting
        
        First detect spikes by continuous wavelet transform.
            - This step will decide the window size used for the next step
        
        Next detect baseline by comparing rolling averages of given window size
        (determined in the first step)
        
        Argument:
            out: bool, if "True", returns pre-p0rocessing results.
                 Default value "False"
         
        Returns:
            dict object containing:
                (int): startidx,pre_endidx,Slice,post_startidx,endidx,spike_start,spike_end
                (bool): spike,process
                (str): remark,ves_type,fit_method
        '''
        ## Detect whether the event has "spikes" (to determine method for baseline detection)
        self.detect_spikes(wname = 'gaus1') # Detect spikes
        self.parse_spikes() # Parse spikes
        """
        # Dynamically plot results of spike detection for manual verification (by multiprocessing)
        flag_spike = multiprocessing.Value('i')
        flag_spike.value = 1
        """
        # Create copies of variables to send to new Process
        datax = self.x.copy()
        datay = self.y.copy()
        Spikes = self.Spikes.copy()
        Slice = self.Slice
        title = self.name
        """
        p_s = multiprocessing.Process(target = plot,args = (datax,datay,Spikes,Slice,title,None,flag_spike)) # Create multiprocessing.Process object to start new process
        p_s.start() # Start the plot method in a separate process (this should generate graph)
        """
        fig = plot(datax, datay, Spikes, Slice, title, None, 1)

        # Prepare screen prompt (in main console)
        if self.spike:
            prompt = 'Detected a spike associated with event, agree? Y/N\n'
        else:
            prompt = 'Did NOT detect a spike associated with event, agree? Y/N\n'
        # Wait for user input
        while True:
            key = input(prompt)
            # If agree with automatic classification
            if key == 'Y' or key == 'y':
                self.remark += 'Confirmed by manual inspection; ' # Make remark
                break # Break out of loop
            # If disagree with automatic classification
            elif key == 'N' or key == 'n':
                self.remark += 'Overturned by manual inspection: ' # Make remark
                self.spike = not self.spike # Change the result
                # Make appropriate notes
                if self.spike:
                    self.remark += 'A spike was judged to be associated with event; '
                else:
                    self.remark += 'NO spike was judged to be associated with event; '
                break # Break out of loop
            # If input is not accepted input values
            else:
                print('Unrecognized command. If detection is correct, press "Y"; otherwise press "N".\n')

        """
        # Set value of flag_spike to 0, thus exiting the while loop in self.plot, this will close the figure
        flag_spike.value = 0
        p_s.join() # Terminate the other process
        """
        plt.close(fig)
        
        ## Detect baseline, which determines vesicle type and whether trace will be processed for fitting
        # Determine whether to fit and method of fitting
        # Determine vesicle type
        window = int(self.fs//16) # Set window to equivalent length of 1/16 of a second
        # If event is marked as "fast"
        if self.spike:
            # Parse baseline
            self.detect_flats(window)
            self.parse_flats(window,mode = 'fast')
        # Otherwise, if event is marked as "slow"
        else:
            # Parse baseline
            self.detect_flats(window)
            self.parse_flats(window,mode = 'slow')
        # Define helper function to retrieve input
        def retrieve_num (prompt):
            '''
            Helper function to retrieve int from correct input
            
            Argument:
                prompt: str, input prompt
            
            Returns:
                int, converted from keyboard input
            '''
            res = input(prompt)
            if not res.isdigit():
                res = retrieve_num ('Invalid input, please provide integer: ')
            else:
                res = int(res)
                if res > self.size:
                    res = retrieve_num(f'Out of Range! Must be between {self.index[0]} & {self.index[-1]}: ')
                elif res == 0:
                    res = np.nan
            return res
        ## Manual verification of baseline detection
        # Dynamically plot results of baseline detection for manual verification (by multiprocessing)
        
        """
        flag_bl = multiprocessing.Value('i')
        flag_bl.value = 1 # Initialize Value to 1
        flag_bl = multiprocessing.Value('i')
        flag_bl.value = 1 # Initialize Value to 1
        indices = multiprocessing.Array('d',4)
        indices[:] = [self.startidx,self.pre_endidx,self.post_startidx,self.endidx] # Initialize Array with indices determined by auto-detection
        p_bl = multiprocessing.Process(target = plot,args=(datax,datay,None,Slice,title,indices,flag_bl)) # Create multiprocessing.Process object to start new process
        p_bl.start() # Run plot method in separate process (this should generate graph)
        """
        indices = [self.startidx, self.pre_endidx, self.post_startidx, self.endidx]
        fig = plot(datax, datay, None, Slice, title, indices, 1)
        # Prepare input prompts:
        # If startidx AND pre_endidx are both not NaN
        if np.isfinite(self.startidx * self.pre_endidx):
            prompt1 = f'Detected pre-event baseline, start: {self.startidx}, end: {self.pre_endidx}\n'
        # Otherwise (if either one is NaN)
        else:
            prompt1 = 'Did NOT detect pre-event baseline\n'
        prompt1 += 'Modify baseline detection results? Y/N\n'
        # If endidx AND post_startidx are both not NaN
        if np.isfinite(self.post_startidx * self.endidx):
            prompt2 = f'Detected post-event baseline, start: {self.post_startidx}, end: {self.endidx}\n'
        # Otherwise (if either one is NaN)
        else:
            prompt2 = 'Did NOT detect post-event baseline\n'
        prompt2 += 'Modify baseline detection results? Y/N\n'
        # Verify pre-event baseline
        while True:
            key1 = input(prompt1)
            if key1 == 'Y' or key1 == 'y':
                while True:
                    indices[0] = retrieve_num ('Specify index of pre-event baseline starting point (to remove baseline, enter 0): ') # Manually set startidx
                    fig.clear()
                    plot(datax, datay, None, Slice, title, indices, 1)
                    indices[1] = retrieve_num('Specify index of pre-event baseline ending point (to remove baseline, enter 0): ') # Manually set pre_endidx
                    fig.clear()
                    plot(datax, datay, None, Slice, title, indices, 1)
                    key = input('Press "Y" to continue. Otherwise, press "N" to modify your changes: ')
                    if key == 'Y' or key == 'y':
                        break
                # Try to convert values to int, if fail, then store as NaN
                try:
                    self.startidx = int(indices[0])
                except ValueError:
                    self.startidx = np.nan
                try:
                    self.pre_endidx = int(indices[1])
                except ValueError:
                    self.pre_endidx = np.nan
                # Update pre attribute
                self.pre = self.y.loc[self.startidx:self.pre_endidx]
                # If startidx OR pre_endidx attribute is NaN, delete pre attribute
                if np.isnan(self.startidx * self.pre_endidx):
                    del self.pre
                    self.remark += f'Pre-event baseline was manually removed; ' # Add remark
                else:
                    self.remark += f'Pre-event baseline was manually specified, starting index: {self.startidx}, ending index: {self.pre_endidx}; ' # Add remark
                break
            elif key1 == 'N' or key1 == 'n':
                break
            else:
                print('Unrecognized command. Press "Y" if pre-event baseline endpoints needs adjusting. Otherwise press "N".\n')
        # Verify post-event baseline
        while True:
            key2 = input(prompt2)
            if key2 == 'Y' or key2 == 'y':
                while True:
                    indices[2] = retrieve_num('Specify index of post-event baseline starting point (to remove baseline, enter 0): ') # Manually set post_startidx
                    fig.clear()
                    plot(datax, datay, None, Slice, title, indices, 1)

                    indices[3] = retrieve_num('Specify index of post-event baseline ending point (to remove baseline, enter 0): ') # Manually set endidx
                    fig.clear()
                    plot(datax, datay, None, Slice, title, indices, 1)

                    key = input('Press "Y" to continue. Otherwise, press "N" to modify your changes: ')
                    if key == 'Y' or key == 'y':
                        break
                # Try to convert values to int, if fail, then store as NaN
                try:
                    self.post_startidx = int(indices[2])
                except ValueError:
                    self.post_startidx = np.nan
                try:
                    self.endidx = int(indices[3])
                except ValueError:
                    self.endidx = np.nan
                # Update post attribute
                self.post = self.y.loc[self.post_startidx:self.endidx]
                # If endidx OR post_startidx attribute is NaN, delete post attribute
                if np.isnan(self.endidx * self.post_startidx):
                    del self.post
                    self.remark += 'Post-event baseline was manually removed; ' # Add remark
                else:
                    self.remark += f'Post-event baseline was manually specified, starting index: {self.post_startidx}, ending index: {self.endidx}; ' # Add remark
                break
            elif key2 == 'N' or key2 == 'n':
                break
            else:
                print('Unrecognized command. Press "Y" if post-event baseline endpoints needs adjusting. Otherwise press "N".\n')

        """
        # Set value of flag_bl to 0, thus exiting the while loop in self.plot, this will close the figure
        flag_bl.value = 0
        p_bl.join() # Terminate the other process
        """
        plt.close(fig)

        ## Check vesicle type
        self.parse_vestype()
        ## Determine fit method
        # If value of process attribute is "True" (trace is marked for further processing)
        if self.process:
            # If a spike was associated with event
            if self.spike:
                # Set fit_method attribute to "curve" - mark trace to be processed by least-squares curve fitting
                self.fit_method = 'curve'
            else:
                # Set fit_method attribute to "spline" - mark trace to be processed by least-squares spline fitting
                self.fit_method = 'spline'
        # Return pre_processing parameters if out is set to "True"
        if out:
            res = {}
            res['startidx'] = self.startidx
            res['pre_endidx'] = self.pre_endidx
            res['Slice'] = self.Slice
            res['post_startidx'] = self.post_startidx
            res['endidx'] = self.endidx
            res['spike'] = self.spike
            res['spike_start'] = self.spike_start
            res['spike_end'] = self.spike_end
            res['process'] = self.process
            res['remark'] = self.remark
            res['ves_type'] = self.ves_type
            res['fit_method'] = self.fit_method
            return res
        
    def est_params(self, out = False):
        '''
        Method for estimating initial parameters for curve fitting
        
        Argument:
            out: bool. If set to "True", return lmfit.Parameters object that
                 contains all the estimated parameters with initial values
        '''
        # Define the relevant data points to use for curve fitting
        self.x_val = self.x.loc[self.startidx:self.endidx]
        self.y_val = self.y.loc[self.startidx:self.endidx]
        # Extract features of the curve
        bg1 = self.pre.mean() # Estimate pre-event baseline
        bg2 = self.post.mean() # Estimate post-event baseline
        Peak = self.y_val.max() # Estimate peak value
        Peakidx = self.y_val.idxmax() # Find the peak index
        T_max = self.x_val.loc[Peakidx] # Estimate corresponding time point of the Peak
        self.Peak_emp = Peak
        self.Tmax_emp = T_max # Empirical Peak time point (from raw data)
        # Estimate the region of the spike
        bumps1 = self.y_val > bg1 # Create boolean array, describing whether data point is greater than bg1
        bumps2 = self.y_val > bg2 # Create boolean array, describing whether data point is greater than bg2
        idx1 = bumps1[bumps1 == False].index # Get all indices of elements in bumps which are "False"
        idx2 = bumps2[bumps2 == False].index # Get all indices of elements in bumps which are "False"
        spike_startidx = idx1[idx1 < Peakidx][-1] # Define spike start index as the last element in idx which is smaller than the Peak index
        spike_endidx = idx2[idx2 > Peakidx][0] # Define the spike end index as the first element in idx which is greater than the Peak index
        # Estimate initial values for parameters
        T_rise = self.x_val.loc[spike_startidx] # Estimate time point of signal rise start
        T_decay = self.x_val.loc[spike_endidx] # Estmiate time point of signal decay end
        # Create spike_index boolean array for computing error of the peak (Only "True" within the spike region)
        N_spike = spike_endidx - spike_startidx + 1 # Compute the number of datapoints within spike area
        if N_spike <= 9: # If there are too few datapoints
            adj = np.ceil((10-N_spike)/2).astype(np.int_)
            spike_endidx += adj
            spike_startidx -= adj
        self.N_spike = spike_endidx - spike_startidx + 1
        self.spike_index = pd.Series(np.zeros(self.x_val.size,dtype = bool),index = self.x_val.index)
        self.spike_index.loc[spike_startidx:spike_endidx] = True
        # Estimate rate constant for exponential increase model
        k_rise = 1 / (T_max - T_rise)
        # Estimate rate constant for exponential decay models
        k_slow = 1 / (T_decay - T_max)
        k_fast = k_slow
        
        # Initialize Parameter objects for use in least-squares curve fitting
        self.bg1 = lmfit.Parameter(name='bg1',value = bg1) # pre-event baseline
        self.T_max = lmfit.Parameter(name='T_max',value = T_max) # Timepoint of peak
        self.iota = lmfit.Parameter(name='iota',value = min(bg2 - bg1,0), max = 0) # iota = bg2 - bg1, limit: iota <= 0
        self.delta = lmfit.Parameter(name='delta',value = max(Peak - bg1,0), min = 0) # delta = Peak - bg1, limit: delta > 0
        self.alpha = lmfit.Parameter(name='alpha',value = min(T_rise - T_max,0), max = 0) # alpha = T_start - T_max, limit: alpha < 0
        self.eta = lmfit.Parameter(name='eta',value = max(T_decay - T_max,0), min = 0) # eta = T_decay - T_max, limit: eta > 0
        self.k_rise = lmfit.Parameter(name='k_rise',value = k_rise, min = 0) # Rate constant for exponential rise, must be positive
        self.k_slow = lmfit.Parameter(name='k_slow',value = k_slow, min = 0) # Rate constant for exponential decay (one phase or slow component), must be positive
        self.k_fast = lmfit.Parameter(name='k_fast',value = k_fast, min = 0) # Rate constant for fast componenet of two-phase exponential decay, must be positive
        self.percent_fast = lmfit.Parameter(name='percent_fast',value = 0.5, min = 0, max = 1) # Contribution of fast component in two-phase exponential decay, must be between 0 and 1
        
        # If set to return results
        if out:
            res = lmfit.Parameters() # Create instance of lmfit.Parameters class
            res.add_many(self.bg1,self.T_max,self.iota,self.delta,self.alpha,self.eta,self.k_rise,self.k_slow,self.k_fast,self.percent_fast) # Add the estimated parameters to lmfit.Parameters
            return res
        
    def curve_fit(self, report = True, out = False):
        '''
        Method for performing curve fit
        
        Arguments:
            report: bool. Whether to report best fit result. Default value "True"
            out: bool, whether to return fit results. Default value "False"
        
        Returns:
            numpy array containing all fit results
        '''
        self.est_params() # Estimate initial parameters
        # Create lmfit.Model instances
        modA0 = lmfit.Model(lin_lin, name = 'A0: linear rise, linear decay')
        modA1 = lmfit.Model(lin_1p, name = 'A1: linear rise, one phase decay')
        modA2 = lmfit.Model(lin_2p, name = 'A2: linear rise, two phase decay')
        modB0 = lmfit.Model(exp_lin, name = 'B0: exponential rise, linear decay')
        modB1 = lmfit.Model(exp_1p, name = 'B1: exponential rise, one phase decay')
        modB2 = lmfit.Model(exp_2p, name = 'B2: exponential rise, two phase decay')
        # Create lmfit.Parameters objects for each model
        paramsA0 = lmfit.Parameters()
        paramsA1 = lmfit.Parameters()
        paramsA2 = lmfit.Parameters()
        paramsB0 = lmfit.Parameters()
        paramsB1 = lmfit.Parameters()
        paramsB2 = lmfit.Parameters()
        # Add appropriate lmfit.Parameter objects
        paramsA0.add_many(self.T_max, self.bg1, self.alpha, self.delta, self.eta, self.iota)
        paramsA1.add_many(self.T_max, self.bg1, self.alpha, self.delta, self.k_slow, self.iota)
        paramsA2.add_many(self.T_max, self.bg1, self.alpha, self.delta, self.iota, self.percent_fast, self.k_fast, self.k_slow)
        paramsB0.add_many(self.T_max, self.bg1, self.delta, self.iota, self.eta, self.k_rise)
        paramsB1.add_many(self.T_max, self.bg1, self.delta, self.iota, self.k_slow, self.k_rise)
        paramsB2.add_many(self.T_max, self.bg1, self.delta, self.iota, self.percent_fast, self.k_slow, self.k_fast, self.k_rise)
        # Perform fit (Using trust-region reflected least squares algorithm)
        fitA0 = modA0.fit(data = self.y_val, x = self.x_val, params = paramsA0, nan_policy = 'raise', method = 'Least_squares')
        fitA1 = modA1.fit(data = self.y_val, x = self.x_val, params = paramsA1, nan_policy = 'raise', method = 'Least_squares')
        fitA2 = modA2.fit(data = self.y_val, x = self.x_val, params = paramsA2, nan_policy = 'raise', method = 'Least_squares')
        fitB0 = modB0.fit(data = self.y_val, x = self.x_val, params = paramsB0, nan_policy = 'raise', method = 'Least_squares')
        fitB1 = modB1.fit(data = self.y_val, x = self.x_val, params = paramsB1, nan_policy = 'raise', method = 'Least_squares')
        fitB2 = modB2.fit(data = self.y_val, x = self.x_val, params = paramsB2, nan_policy = 'raise', method = 'Least_squares')
        # Construct numpy array containing the fit results
        fit_res = np.array([fitA0,fitA1,fitA2,fitB0,fitB1,fitB2])
        self.N_spike = self.spike_index.sum() # Count the number of points within spike region
        Curve_AICc = lambda x: self.N_spike * np.log((self.y_val[self.spike_index] - x.best_fit[self.spike_index]).dot(self.y_val[self.spike_index] - x.best_fit[self.spike_index]) / self.N_spike) + 2 * x.nvarys + 2 * x.nvarys * (x.nvarys + 1) / (self.N_spike - x.nvarys - 1)  # Compute the AICc value in the spike region
        curve_errs = np.array(list(map(Curve_AICc,fit_res))) # Get AICc value for curve region for all models
        min_err = curve_errs.min() # Find minimum error
        mods = fit_res[curve_errs == min_err] # Find fit result(s) with minimal fit error
        # If there is only one element in mods
        if mods.size == 1:
            self.best_fit = mods[0] # Define model as best fit model
        # Otherwise, if there are more than one element in mods (multiple models with same error)
        else:
            Count_vars = lambda x: x.nvarys
            Num_vars = np.array(list(map(Count_vars,mods)))
            idx = Num_vars.argmin()
            self.best_fit = mods[idx] # Define the model with the least number of parameters as best fit
        # Add remark
        self.remark += f'Least squares fitting by curve fitting, best fit model is {self.best_fit.model.name}; '
        # Save residuals to "residuals" attribute as a pandas Series object
        self.residuals = pd.Series(self.best_fit.residual,index = self.y_val.index)
        # If set to report
        if report:
            print('Best fit model for ' + self.name + ' is:',self.best_fit.model.name + '\n')
            print('Best fit parameters\n')
            self.best_fit.params.pretty_print()
        # If out is set to true, return all fit results
        if out:
            return fit_res
        
    def curve_func(self,x):
        '''
        Callable function of best fit model
        
        Argument:
            x: float or numpy array, parameter
            
        Returns:
            y, float or numpy array, the function value
        '''
        return self.best_fit.model.func(x, **self.best_fit.best_values)
            
    def spline_fit(self,fit_err = None,degree=3,max_smooth=2,min_angle=10,search_range=50,max_iter=100,ScanKnot=True):
        '''
        Method to perform spline regression, adapted from methods described in:
            
            Dung VT and Tjahjowidodo T, A direct method to solve optimal knots
            of B-spline curves: An application for non-uniform B-spline curves 
            fitting, PLoS One, vol. 12, issue 3, pp. e0173857, 2017.
            
        Arguments:
            fit_err: float, error tolerance level for spline regression, if not
                     specified, defaults to estimated noise level
            degree: int, degree of polynomials used for fit, default value 3
            max_smooth: int, maximum allowed continuity, default value 2
            min_angle: float, minimum join angle (in degrees) at multiple knots
            search_range: int, specificies range for knot optimization, default
                          value 50
            max_iter: int, maximum iterations for knot optimization, default
                      value 100
            ScanKnot: bool, whether to scan and optimize all knots, default
                      value "True"
        '''
        # Extend endpoints about one second beyond endpoints, within the limit of data range
        self.y_smooth = self.y.rolling(window=11,center=True).mean().fillna(method='ffill').fillna(method='bfill') # Get 11-window rolling average of y values & fill NaN values
        self.spline_startidx = max(self.index[0],self.startidx - self.fs//10)
        self.spline_endidx = min(self.index[-1],self.endidx + self.fs//10)
        self.spline_degree = degree
        self.spline_order = degree + 1
        # Define derivative operation matrix
        d_mat = np.diag(np.arange(1,self.spline_order,dtype = int),k=1)
        # Construct a list of derivative matrices (to calculate 0 - degree order derivatives)
        self.D_mats = [] # Initialize empty list to store the matrices
        for n in range(self.spline_order+1):
            self.D_mats.append(np.linalg.matrix_power(d_mat,n))
        # Define the fit region
        self.x_f = self.x.loc[self.spline_startidx:self.spline_endidx]
        self.y_f = self.y.loc[self.spline_startidx:self.spline_endidx]
        # Define the relevant data points of the event
        self.x_val = self.x.loc[self.startidx:self.endidx]
        self.y_val = self.y.loc[self.startidx:self.endidx]
        self.y_val_sm = self.y_smooth.loc[self.startidx:self.endidx]
        self.spline_interp = interpolate.CubicSpline(self.x_val,self.y_val_sm,extrapolate=False) # Create a Scipy.Interpolate.CubicSpline instance that interpolates through the smoothed curve
        Peak = self.y_val_sm.max() # Estimate peak value of smoothed curve
        Peakidx = self.y_val_sm.idxmax() # Find the peak index of smoothed curve
        T_max = self.x_val.loc[Peakidx] # Estimate corresponding time point of the Peak
        bg1 = self.pre.mean() # Estimate pre-event baseline
        bg2 = self.post.mean() # Estimate post-event baseline
        self.Peak_emp = Peak
        self.Span_emp = Peak - bg2 # Empirical Span value (from raw data)
        self.Tmax_emp = T_max # Empirical Peak time point (from raw data)
        self.half_max_r_emp = (Peak + bg2) / 2 # Empirical half-decay value
        self.half_max_l_emp = (Peak + bg1) / 2 # Empirical half-rise value
        right_emp = self.spline_interp.solve(self.half_max_r_emp) # Solve for the timepoint of half-decay (empirical)
        left_emp = self.spline_interp.solve(self.half_max_l_emp) # Solve for the timepoint of half-rise (empirical)
        self.r_50_emp = right_emp[right_emp>T_max][0] # Empirical half-decay timepoint is defined as the first timepoint greater than T_max where the smoothed curve reaches half decay
        self.l_50_emp = left_emp[left_emp<T_max][-1] # Empirical half-rise timepoint is defined as the last timepoint smaller than T_max where the smoothed curve reaches half rise
        # Initialize T matrix (in the form of [1,x,x^2,x^3,...,x^degree])
        self.T = pd.DataFrame(np.ones((self.x_f.size,self.spline_order)),index = self.x_f.index)
        for e in range(1,degree + 1):
            self.T[e] = self.T[e-1] * self.x_f
        # Estimate noise level (to determine fit_err) (Pooled standard deviation of pre-event and post-event baselines)
        self.noise = np.sqrt(((self.pre.std() ** 2) * (self.pre.size - 1) + (self.post.std() ** 2) * (self.post.size - 1)) / (self.pre.size + self.post.size - 2))
        # If fit_err is not specified, defaults to estimated noise level
        if fit_err is None:
            fit_err = self.noise
        # Perform parallel bisection to determine coarse knot positions
        self.par_bisect(fit_err)
        # Optimize knot placement
        self.optimize_knots(max_smooth,min_angle,search_range,max_iter,ScanKnot)
        # Use knot information to generate Bspline bases
        self.ANmat = self.NMat_construct(self.knot_vec)
        # Solve for control points to generate curve
        self.ctrl_pts = np.linalg.lstsq(self.ANmat,self.y_f,rcond=None)[0]
        # Compute the curve of the fitted spline
        self.spline_curve = self.ANmat.dot(self.ctrl_pts)
        # Generate a Bspline object
        self.spline_func = interpolate.BSpline(self.knot_vec,self.ctrl_pts,self.spline_degree,extrapolate = False)
        # Generate knot averages of the knot vector (for computing control polygons [x values of each control point])
        self.knot_avg = average_knots(self.knot_vec, self.spline_degree)
        # Add remark
        self.remark += 'Least squares fitting by spline regression; '
        # Save residuals to "residuals" attribute
        self.residuals = self.y_f - self.spline_curve
        
    def gen_kv(self,Knot_seq):
        '''
        Helper function to generate knot vector from knot sequence (repeats the end
        knot points degree + 1 times)
        
        Arguments:
            Knot_seq: iterable object containing knot sequence (with internal
                      multiple knots repeated approriate number of times)
        
        Returns:
            1D numpy ndarray object containing the knot vector (with end points
            repeated)
        '''
        # Check if Knot_seq is list, if not, convert to list
        if type(Knot_seq) != list:
            Knot_seq = list(Knot_seq)
        # Repeat the end points for the appropriate number of times
        return np.array([Knot_seq[0]] * self.spline_degree + Knot_seq + [Knot_seq[-1]] * self.spline_degree)
    
    def NewMatrix(self,Knot_vec):
        '''
        Helper method to generate Bspline basis matrices
        
        Argument:
            Knot_seq: iterable, knot vector (with repeated end points)
            degree: int, degree of polynomial used to construct basis functions
            
        Returns:
            Flattened array containing the coefficients to construct Bspline
            basis matrix
        '''
        Intervals = len(Knot_vec) - 1
        row_num = self.spline_order * np.arange(1,self.spline_order+1).sum()
        CoeNmat = np.zeros((row_num,Intervals))
        activeKnot = np.zeros(Intervals,dtype = bool)
        # Calculate Ni,0 for 0 degree
        for ii in range(Intervals):
            if Knot_vec[ii+1] - Knot_vec[ii] != 0:
                CoeNmat[0,ii] = 1
                activeKnot[ii] = True
        # Calculate Ni,j for higher degrees
        for ii in range(Intervals): # for each interval
            if activeKnot[ii]:
                for jj in range(1,self.spline_order): # for each degree
                    id0 = self.spline_order* np.arange(0,jj).sum()
                    id1 = self.spline_order* np.arange(0,jj+1).sum()
                    for kk in range(jj + 1): # each id member matrix at interval ii
                        id2 = ii - jj + kk # effective interval
                        id2Knot00 = id2 + jj
                        id2Knot01 = id2Knot00 + 1
                        if id2 > -1 and id2Knot01 <= len(Knot_vec)-1:
                            # Access previous data Ni-1, j-1 Ni,j-1 and Ni+1,j-1
                            id00 = id0 + (kk-1) * self.spline_order
                            id01 = id0 + kk * self.spline_order
                            if kk == 0: # first box of matrix
                                N0 = np.zeros(self.spline_order)
                                N1 = CoeNmat[id01:id01 + self.spline_order, ii]
                            elif kk == jj:
                                N0 = CoeNmat[id00:id00 + self.spline_order, ii]
                                N1 = np.zeros(self.spline_order)
                            else:
                                N0 = CoeNmat[id00:id00 + self.spline_order, ii]
                                N1 = CoeNmat[id01:id01 + self.spline_order, ii]
                            # calculate a1x + a0
                            aden = Knot_vec[id2Knot00] - Knot_vec[id2]
                            bden = Knot_vec[id2Knot01] - Knot_vec[id2+1]
                            a0 = 0
                            a1 = 0
                            if aden != 0:
                                a1 = 1/aden
                                a0 = - Knot_vec[id2] / aden
                            # calculate b1x + b0
                            b0 = 0
                            b1 = 0
                            if bden != 0:
                                b1 = -1/bden
                                b0 = Knot_vec[id2Knot01]/bden
                            #Multiplication
                            Acoef = np.zeros(self.spline_order)
                            N00 = a0 * N0
                            N01 = a1 * N0
                            N10 = b0 * N1
                            N11 = b1 * N1
                            Acoef[0] = N00[0] + N10[0]
                            for n in range(1,self.spline_order):
                                Acoef[n] = N00[n] + N10[n] + N01[n-1] + N11[n-1]
                            id11 = id1 + kk * self.spline_order
                            CoeNmat[id11:id11 + self.spline_order,ii] = Acoef
        # Store data
        id10 = self.spline_order * np.arange(0,self.spline_order).sum()
        return CoeNmat[id10:,:]
    
    def NMat_construct(self,Knot_vec):
        '''
        Helper method to construct basis spline matrix
        
        Arguments:
            Knot_vec: iterable, knot vector (with end points repeated)
            
        Returns:
            ANmat: Bspline matrix
        '''
        start = self.x_f[self.x_f>=Knot_vec[0]].index[0] # Find the index in x corresponding to left endpoint
        end = self.x_f[self.x_f<=Knot_vec[-1]].index[-1] # Find the index in x corresponding to right endpoint
        x = self.x_f.loc[start:end] # Find the corresponding region of x values
        data = self.y_f.loc[start:end] # Find the corresponding region of data values
        opt_knots, multiples = np.unique(Knot_vec,return_counts = True) # Find unique knot positions in given knot vector, as well as the counts of each unique knot position
        # Construct matrix 
        looptime = opt_knots[1:-1].size # Find the number of interior knots
        findidx = lambda u: x[x<u].index[-1] # Find the right boundary index of interior knots
        InteriorKnotIdx = list(map(findidx,opt_knots[1:-1])) # Map the function across all internal knots and generate list (active region: knot_i-1 <= x < knot_i)
        Nmat = self.NewMatrix(Knot_vec)
        ANmat = pd.DataFrame(np.zeros((data.size,len(Knot_vec)-self.spline_order)),index = data.index)
        Dknot = np.diff(Knot_vec)
        indexnonzero = np.where(Dknot!=0)[0]
        idx0 = start
        idxrow0 = 0
        # Iterate across all internal knots
        for k in range(looptime):
            idx1 = InteriorKnotIdx[k]
            idxrow1 = idxrow0 + self.spline_degree
            ANmat.loc[idx0:idx1,idxrow0:idxrow1] = np.array((self.T.loc[idx0:idx1,:]).dot(np.reshape(Nmat[:,indexnonzero[k]],(self.spline_order,self.spline_order),'F')))
            idx0 = idx1 + 1
            idxrow0 = idxrow0 + multiples[1:-1][k]
        
        idx1 = end
        idxrow1 = idxrow0 + self.spline_degree
        ANmat.loc[idx0:idx1,idxrow0:idxrow1] = np.array((self.T.loc[idx0:idx1,:]).dot(np.reshape(Nmat[:,indexnonzero[looptime]],(self.spline_order,self.spline_order),'F')))
        return ANmat
        
    def par_bisect(self,fit_err,seg_len = None, len_limit = None, VectorIn = None, out = False):
        '''
        Method to determine coarse knot locations by parallel bisection
        
        Arguments:
            fit_err: float, error tolerance level.
            seg_size: int, number of data points per segment in the initial
                      segmentation, optional. If not given, defaults to int(fs)
            len_limit: int, minimum number of datapoints to attempt fit.
                       Optional, if not given, defaults to fs//8
            VectorIn: Iterable object containing the indices of starting knots,
                      Optional
            out: bool, whether to return coarse knots. Default value "False"
        '''
        ## Initialize function parameters
        # If seg_len is not specified, default to fs (rounded down)
        if seg_len is None:
            seg_len = self.fs//1
        # If VectorIn is not specified, default to even division from start to finish of fitting region
        if VectorIn is None:
            KnotX1 = np.arange(self.spline_startidx,self.spline_endidx,seg_len) # Initialize Knot vector to specify index locations
            if KnotX1[-1] != self.spline_endidx: # If the last element of KnotX1 is not the last index of data segment
                KnotX1 = np.append(KnotX1,self.spline_endidx)
        # Otherwise, make sure that VectorIn also includes the start and ending index of fitting region
        else:
            KnotX1 = np.array(VectorIn)
            if self.spline_startidx not in VectorIn:
                KnotX1 = np.append(self.spline_startidx,KnotX1)
            if self.spline_endidx not in VectorIn:
                KnotX1 = np.append(KnotX1,self.spline_endidx)
        # If len_limit is not specified, default to fs/8 (rounded down)
        if len_limit is None:
            len_limit = self.fs//8
        
        VectorUX = np.zeros((3,KnotX1.size-1)) # Initialize numpy array to store bisection results
        VectorUX[0] = KnotX1[:-1] # First row to store indices of starting points of each sub-segment
        VectorUX[1] = np.append(KnotX1[1:-1] + 1, KnotX1[-1]) # Second row to store indices of ending sub-segment
        VectorUX[-1] = fit_err + 1
        
        # Parallel bisection: fit segment with one piece 
        while True:
            # Perform piece-wise single-piece Bspline fit
            for k in range(KnotX1.size - 1):
                # get start and end points of the current segment under consideration
                left = VectorUX[0,k]
                right = VectorUX[1,k]
                # If current segment does not satisfy error tolerance test, and segment length is above limit, perform one-piece Bspline fit on segment
                if right - left + 1 >= len_limit and VectorUX[-1,k] > fit_err:
                    k_seq = self.x_f.loc[[left,right]] # Generate knot sequence for current piece
                    k_vec = self.gen_kv(k_seq) # Generate knot vector from knot sequence
                    ANmat = self.NMat_construct(k_vec) # Construct Bspline matrix from knot vector
                    y_seg = self.y_f.loc[left:right] # Define current segment of data points
                    ctrlp = np.linalg.lstsq(ANmat,y_seg,rcond=None)[0] # Solve for control points by (linear) least squares method
                    curve = ANmat.dot(ctrlp) # Compute values of fitted curve
                    error = abs(y_seg - curve) # Compute errors of fit
                    VectorUX[-1,k] = error.max() # Store max error in result
                # Otherwise if current segment length is below limit
                elif right - left + 1 < len_limit:
                    VectorUX[-1,k] = -1 # Mark error value as "-1"
            
            # Perform error tolerance test
            DivideNeedX = VectorUX[-1] > fit_err
            # If there are True values in DivideNeedX
            if DivideNeedX.any():
                Knots_temp = [] # Initialize empty list to store knot indices
                Errs = [] # Initialize empty list to store fitting errors
                # Iterate through each segment (from left to right)
                for j in range(KnotX1.size - 1):
                    # get start and end points of the current segment under consideration
                    left = VectorUX[0,j]
                    right = VectorUX[1,j]
                    # If current segment failed error tolerance test
                    if DivideNeedX[j]:
                        Knots_temp.append(left) # First append the original knot point
                        bisect_end = (left+right)//2 # Find the new endpoint of bisected fragment to the left
                        Errs.append(fit_err + 1)
                        Knots_temp.append(bisect_end + 1) # Append the new startpoint of bisected fragment to the right to temporary knot list
                        Errs.append(fit_err + 1)
                    # OTherwise (the segment satisfies requirement [or is too short to attempt fit])
                    else:
                        Knots_temp.append(left) # add starting point of segment into new list
                        Errs.append(VectorUX[-1,j]) # Append original fit error to storage list
                Knots_temp.append(VectorUX[1,-1]) # After iterating through all pieces, append the end point of the last segment
                # Update parameters for next iteration
                KnotX1 = np.array(Knots_temp)
                Errs = np.array(Errs)
                VectorUX = np.zeros((3,KnotX1.size-1)) # Initialize numpy array to store bisection results
                VectorUX[0] = KnotX1[:-1] # First row to store indices of starting points of each sub-segment
                VectorUX[1] = np.append(KnotX1[1:-1] + 1, KnotX1[-1]) # Second row to store indices of ending sub-segment
                VectorUX[-1] = Errs # Store fit errors in last row
            else:
                break
    
        # Join adjacent segments that passed the error test (and were long enough to fit)
        JoinNeedX = (VectorUX[-1] < fit_err) & (VectorUX[-1] >= 0) # Mark segments which passed Error test AND are longer than len_limit
        ## Check that there are consecutvie segments which passed the error tolerance test to be joined
        calc = np.append(np.insert(JoinNeedX.astype(int),0,0),0) # append "0" to the front and end of JoinNeedX (as type int)
        cons = np.diff(calc) # Take discrete differential of calc
        starts = np.where(cons == 1)[0] # the start of a consecutive sequence is marked by "1"
        ends = np.where(cons == -1)[0] # the end of a consecutive sequence is marked by "-1"
        cons_lens = ends - starts # Compute the length of each consecutive sequence of segments that satisfy the condition
        check_len = cons_lens >= 2
        # If there are any consecutive lengths that are 2 or above
        if check_len.any():
            Knots_temp = [] # Initiate temporary list to store end point information
            Errs = [] # Initiate empty list to store fit errors
            discard = [] # Initiate temporary list to store discard/skipped points
            # Iterate through all segments
            for i in range (KnotX1.size - 1):
                # get start and end points of the current segment under consideration
                left = VectorUX[0,i]
                right = VectorUX[1,i]
                # If left point is not in discard list
                if left not in discard:
                    # If current segment can be joined
                    if JoinNeedX[i]:
                        Knots_temp.append(left) # Append left point of current segment to temporary knot list
                        m = 1
                        temp = [] # Initiate temporary list to store fit results
                        # while i + m is not out of index range
                        while i + m <= KnotX1.size - 2:
                            # If the next segment can be joined, join the two segments together and try fit again
                            if JoinNeedX[i + m]:
                                # get start and end points of next segment
                                left_prime = VectorUX[0,i+m]
                                right_prime = VectorUX[1,i+m]
                                # Join segments together and try fitting with one piece Bspline
                                k_seq = self.x_f.loc[[left,right_prime]] # Generate knot sequence for joined piece
                                k_vec = self.gen_kv(k_seq) # Generate knot vector from knot sequence
                                ANmat = self.NMat_construct(k_vec) # Construct Bspline matrix from knot vector
                                y_seg = self.y_f.loc[left:right_prime] # Define current segment of data points
                                ctrlp = np.linalg.lstsq(ANmat,y_seg,rcond=None)[0] # Solve for control points by (linear) least squares method
                                curve = ANmat.dot(ctrlp) # Compute values of fitted curve
                                error = abs(y_seg - curve) # Compute errors of fit
                                # If the joined segment passes error test
                                if error.max() < fit_err:
                                    discard.append(left_prime) # Add start point of next segment to discard list
                                    temp.append(error.max()) # Append fit error to Errs list
                                    m += 1 # Increment m (expand to next segment)
                                else:
                                    # If there are elements in temporary list
                                    if len(temp) > 0:
                                        Errs.append(temp[-1]) # append the last element in temp
                                    # Otherwise (could not be joined with another segment)
                                    else:
                                        Errs.append(VectorUX[-1,i]) # append the original fit error for current segment
                                    break # exit while loop
                            # If the next segment cannot be joined
                            else:
                                # If there are elements in temporary list
                                if len(temp) > 0:
                                    Errs.append(temp[-1]) # append the last element in temp
                                # Otherwise (could not be joined with another segment)
                                else:
                                    Errs.append(VectorUX[-1,i]) # append the original fit error for current segment
                                break # exit while loop
                        if i + m > KnotX1.size - 2:
                            # If there are elements in temporary list
                            if len(temp) > 0:
                                Errs.append(temp[-1]) # append the last element in temp
                            # Otherwise (could not be joined with another segment)
                            else:
                                Errs.append(VectorUX[-1,i]) # append the original fit error for current segment
                    # Otherwise if current segment cannot be joined (too short)
                    else:
                        Knots_temp.append(left) # Append left point of current segment to temporary knot list
                        Errs.append(VectorUX[-1,i]) # Append original fit result to storage
            Knots_temp.append(VectorUX[1,-1]) # After iterating through all pieces, append the end point of the last segment
            KnotX1 = np.array(Knots_temp)
            Errs = np.array(Errs)
            VectorUX = np.zeros((3,KnotX1.size-1)) # Initialize numpy array to store bisection results
            VectorUX[0] = KnotX1[:-1] # First row to store indices of starting points of each sub-segment
            VectorUX[1] = np.append(KnotX1[1:-1] + 1, KnotX1[-1]) # Second row to store indices of ending sub-segment
            VectorUX[-1] = Errs # Store fit errors in third row
        
        ShiftNeedX = VectorUX[-1] == -1 # flag short segments in need of shifting
        # If there are short segments
        if ShiftNeedX.any():
            Knots_temp = [] # Initiate temporary list to store end point information
            Errs = [] # Initiate temporary list to store fitted values
            discard = [] # Initiate temporary list to store discard/skipped points
            # Iterate through each segment
            for i in range(KnotX1.size - 1):
                # Get start and end points of current segment
                left = VectorUX[0,i]
                right = VectorUX[1,i]
                # If left point is not in discard list
                if left not in discard:
                    Knots_temp.append(left) # Append to temporary knots list (to be kept)
                    # If not last segment
                    if i < KnotX1.size - 2:
                        # Get the start and end points of the following segment
                        left_prime = VectorUX[0,i+1]
                        right_prime = VectorUX[1,i+1]
                        # If current segment is long
                        if not ShiftNeedX[i]:
                            # If next segment is short
                            if ShiftNeedX[i+1]:
                                # Join segments and perform serial bisection (section from the right)
                                end = right_prime # Define end index as the index of the last point of joined segment
                                # while end point is greater than original endpoint of unjoined segment
                                while end > right:
                                    k_seq = self.x_f.loc[[left,end]] # Generate knot sequence for current piece
                                    k_vec = self.gen_kv(k_seq) # Generate knot vector from knot sequence
                                    ANmat = self.NMat_construct(k_vec) # Construct Bspline matrix from knot vector
                                    y_seg = self.y_f.loc[left:end] # Define current segment of data points
                                    ctrlp = np.linalg.lstsq(ANmat,y_seg,rcond=None)[0] # Solve for control points by (liner) least squares method
                                    curve = ANmat.dot(ctrlp) # Compute values of fitted curve
                                    error = abs(y_seg - curve) # Compute errors of fit
                                    # if fit of joined segment passes error tolerance test
                                    if error.max() < fit_err:
                                        Errs.append(error.max()) # store the fit error of new joined segment
                                        # If end is less than the right-most endpoint of joined segment
                                        if end < right_prime:
                                            VectorUX[1,i] = end # Update the end position of the current segment
                                            VectorUX[-1,i] = error.max() # Update the fit error of the current segment
                                            VectorUX[0,i+1] = end + 1 # Update the start position of the next segment
                                        # Otherwise if entire joined segment can be combined
                                        else:
                                            discard.append(left_prime) # Add starting point of next segment to discard list to be skipped
                                        break # Break out of loop
                                    # Otherwise, if fit error does not pass tolerance test
                                    else:
                                        end = (right + end) // 2 # Bisect segment and retry fit
                                        # If end == right (the right short segment cannot be merged with current long segment)
                                        if end == right:
                                            Errs.append(VectorUX[-1,i]) # Append original fit error to Errs list
                            #Otherwise, if next segment is long
                            else:
                                Errs.append(VectorUX[-1,i]) # Append original fit error to Errs list
                        # Otherwise, if current segment is short    
                        else:
                            Errs.append(VectorUX[-1,i]) # store the fit error (=-1) of original segment
                            # If next segment is long
                            if not ShiftNeedX[i+1]:
                                # Join segments and perform serial bisection (section from the left)
                                start = left # Define start index as the index of the first point of the joined segment
                                # While start point is smaller than the startpoint of the next segment
                                while start < left_prime:
                                    k_seq = self.x_f.loc[[start,right_prime]] # Generate knot sequence for current piece
                                    k_vec = self.gen_kv(k_seq) # Generate knot vector from knot sequence
                                    ANmat = self.NMat_construct(k_vec) # Construct Bspline matrix from knot vector
                                    y_seg = self.y_f.loc[start:right_prime] # Define current segment of data points
                                    ctrlp = np.linalg.lstsq(ANmat,y_seg,rcond=None)[0] # Solve for control points by (liner) least squares method
                                    curve = ANmat.dot(ctrlp) # Compute values of fitted curve
                                    error = abs(y_seg - curve) # Compute errors of fit
                                    # if fit of joined segment passes error tolerance test
                                    if error.max() < fit_err:
                                        # If start is greater than the left-most endpoint of joined segment
                                        if start > left:
                                            VectorUX[1,i] = start - 1 # Update the end position of the current segment
                                            VectorUX[0,i+1] = start # Update the start position of the next segment
                                            VectorUX[-1,i+1] = error.max() # Update the fit error of the next segment
                                        # Otherwise if entire joined segment can be combined
                                        else:
                                            discard.append(left_prime) # Add starting point of next segment to discard list to be skipped
                                        break # Break out of loop
                                    # Otherwise, if fit error does not pass tolerance test
                                    else:
                                        start = (start + left_prime) // 2 + 1 # Bisect segment and retry fit
                            # Otherwise, if the next segment is short, perform relative comparisons to determine whether to keep knot
                            else:
                                m = 1
                                # While i + m is within index range AND next segment is short
                                while (i + m < KnotX1.size - 1) and ShiftNeedX[i + m]:
                                    left_prime = VectorUX[0,i+m]
                                    right_prime = VectorUX[1,i+m]
                                    res = np.zeros(self.spline_order + 1) # Create zero array to store fit error results for comparison
                                    y_seg = self.y_f.loc[left:right_prime] # Define relevant segment of data points
                                    for mul in range(self.spline_order + 1):
                                        k_seq = self.x_f.loc[[left] + [left_prime] * mul + [right_prime]] # Generate knot sequence for joined piece (intervening knot point repeated 0,1,2,...,multipleMAX times)
                                        k_vec = self.gen_kv(k_seq) # Generate knot vector from knot sequence
                                        ANmat = self.NMat_construct(k_vec) # Construct Bspline matrix from knot vector
                                        ctrlp = np.linalg.lstsq(ANmat,y_seg,rcond=None)[0] # Solve for control points by (linear) least squares method
                                        curve = ANmat.dot(ctrlp) # Compute values of fitted curve
                                        error = abs(y_seg - curve) # Compute errors of fit
                                        res[mul] = error.max() # Store max fit error in res
                                    # If fit error of joined segment is smallest (knot can be omitted)
                                    if np.argmin(res) == 0:
                                        VectorUX[1,i] = right_prime # Update end point of current segment
                                        discard.append(left_prime) # Add starting point of next segment to discard list to be skipped
                                        m += 1 # Shift to the next segment and see if it can also be merged
                                    # Otherwise, if knot cannot be dropped
                                    else:
                                        # Perform serial bisection (from right end) to determine new knot point juncture
                                        end = (VectorUX[1,i] + right_prime) // 2 # Start with the first bisection point (because the knot of the full segment cannot be joined)
                                        # While end point is greater than the (updated) right point of current segment
                                        while end > VectorUX[1,i]:
                                            res = np.zeros(self.spline_order + 1) # Create zero array to store fit error results for comparison
                                            y_seg = self.y_f.loc[left:end] # Define relevant data points for fitting
                                            # Fit the curve for multiples 0, 1, 2, ..., Order
                                            for mul in range(self.spline_order + 1):
                                                k_seq = self.x_f.loc[[left] + [VectorUX[1,i] + 1] * mul + [end]] # Generate knot sequence for joined piece (intervening knot point repeated 0,1,2,...,multipleMAX times)
                                                k_vec = self.gen_kv(k_seq) # Generate knot vector from knot sequence
                                                ANmat = self.NMat_construct(k_vec) # Construct Bspline matrix from knot vector
                                                ctrlp = np.linalg.lstsq(ANmat,y_seg,rcond=None)[0] # Solve for control points by (linear) least squares method
                                                curve = ANmat.dot(ctrlp) # Compute values of fitted curve
                                                error = abs(y_seg - curve) # Compute errors of fit
                                                res[mul] = error.max() # Store max fit error in 
                                            # If fit error of joined segment is smallest (knot can be omitted)
                                            if np.argmin(res) == 0:
                                                VectorUX[1,i] = end # Update end point of current segment
                                                VectorUX[0,i+m] = end + 1 # Update start point of next segment
                                            # Otherwise, if knot cannot be dropped
                                            else:
                                                end = (VectorUX[1,i] + end)//2 # Bisect segment and try again
                                        # If knot cannot be dropped        
                                        if end == VectorUX[1,i]:
                                            break # break out of while loop
                    # Otherwise, if i is the last segment
                    else:
                        Errs.append(VectorUX[-1,i])
            # Update VectorUX
            Knots_temp.append(VectorUX[1,-1]) # Append last knot to temporary knot list
            KnotX1 = np.array(Knots_temp)
            Errs = np.array(Errs)
            VectorUX = np.zeros((3,KnotX1.size-1)) # Initialize numpy array to store bisection results
            VectorUX[0] = KnotX1[:-1] # First row to store indices of starting points of each sub-segment
            VectorUX[1] = np.append(KnotX1[1:-1] + 1, KnotX1[-1]) # Second row to store indices of ending sub-segment
            VectorUX[-1] = Errs # Store fit errors in third row
            self.coarse_knotx = VectorUX # Store the index values of the coarse knots in attribute "coarse_knotx"
        # If set to return results
        if out:
            return VectorUX
        
    def TwoPcOptKnot (self,left,DataKnot,right,max_smooth,min_angle,search_range,max_iter,ScanKnot):
        '''
        Helper method to optimize knot position between two adjoining spline segments
        
        Arguments:
            left: int, index of starting point of region under consideration
                  (starting point of first segment)
            DataKnot: int, index of the knot point to be optimized (starting 
                      point of the second segment)
            right: int, index of ending point of region under considerationg
                   (end point of second segment)
            max_smooth: int, minimum continuity required
            min_angle: float, minimum "kink" angle required at multiple knots
            search_range: int, specifies the increments of the points during
                          optimization
            max_iter: int, maximum number of iterations when doing Gauss-Newton
                      optimization
            ScanKnot: bool, whether to scan all knots
        
        Returns:
            (opt_knot, multiple):
                opt_knot: float, the optimal value of the knot point
                multiple: int, the multiplicity of the knot point
        '''
        self.KnotOpt_T = self.T.loc[left:right]
        self.KnotOpt_X = self.x_f.loc[left:right]
        self.KnotOpt_Y = self.y_f.loc[left:right]
        idx01 = DataKnot - 1
        idx10 = DataKnot
        idx00 = left
        idx11 = right
        DP1 = idx01 - idx00 - self.spline_degree
        DP2 = idx11 - idx10 - self.spline_degree
        LeftSearch = np.zeros(self.spline_order + 1,dtype = int)
        RightSearch = np.zeros(self.spline_order + 1, dtype = int)
        for e in range(self.spline_order + 1):
            LeftSearch[e] = min(DP1,3) + self.spline_order - e - 1
            RightSearch[e] = min(DP2,3) + self.spline_order - e - 1
        if DP1 > 0 and DP2 > 0:
            LeftSearch[self.spline_order] += 1
            RightSearch[self.spline_order] += 1
        elif DP1<=0 and DP2>0:
            RightSearch[self.spline_order] += 1
        elif DP1>0 and DP2<=0:
            LeftSearch[self.spline_order] += 1
        OptimalKnotSave = np.zeros(self.spline_order + 1)
        ErrorSave = np.zeros(self.spline_order + 1)
        AngleSave = np.zeros(self.spline_order + 1)
        SearchRangeOut = np.zeros((2,self.spline_order + 1))
        MultiplexMax = self.spline_order
        
        ErrorSave[0], AngleSave[0] = self.TwoPcKnotEval(0,self.KnotOpt_X.loc[DataKnot]) # Evaluate Error and Angle for multiple = 0 (Use the given internal knot value), save results
        
        for ii in reversed(range(1,self.spline_order+1)):
            LeftSearch1 = max(LeftSearch[ii],0)
            RightSearch1 = max(RightSearch[ii],0)
            SearchRange = [self.KnotOpt_X.loc[max((idx01 - LeftSearch1),idx00)],self.KnotOpt_X.loc[min((idx10 + RightSearch1),idx11)]]
            # If SearchRange is not empty (right point is larger than left point)
            if SearchRange[1] > SearchRange[0]:
                # Generate a range of knot points to evaluate
                ExpandRange = max(int(np.ceil((search_range + 1)/(2*(LeftSearch1 + RightSearch1)))),1)
                Multiple = ii
                KnotLocation = np.linspace(SearchRange[0],SearchRange[1],search_range)
                DisErrorSave = np.zeros(KnotLocation.size)
                DisAnglePsave = np.zeros(KnotLocation.size)
                # For each point in range, evaluate error and JoinAngle
                for i in range(KnotLocation.size):
                    StartPoint = KnotLocation[i]
                    DisErrorSave[i],DisAnglePsave[i] = self.TwoPcKnotEval(Multiple,StartPoint)
                # Compute four-fold knot position
                KnotLocation = KnotLocation[~np.isnan(DisAnglePsave)]
                DisErrorSave = DisErrorSave[~np.isnan(DisAnglePsave)]
                DisAnglePsave = DisAnglePsave[~np.isnan(DisAnglePsave)]
                # Select region
                if ii == self.spline_order:
                    CtrlError1 = DisErrorSave.min() + 10**-10
                    disIdx = np.where(DisErrorSave < CtrlError1)[0]
                    MidPosition = int(np.round((min(disIdx) + max(disIdx))/2))
                    SearchRangeOut[:,ii] = np.array([KnotLocation[max((disIdx[0]-1),0)],KnotLocation[min((disIdx[-1]+1),KnotLocation.size-1)]])
                    OptimalKnotSave[ii] = KnotLocation[MidPosition]
                    AngleSave[ii] = DisAnglePsave[MidPosition]
                    ErrorSave[ii] = DisErrorSave[MidPosition]
                    if not ScanKnot:
                        OptimalKnotSave[1:self.spline_order] = OptimalKnotSave[ii]
                        for e in range(1,self.spline_order):
                            SearchRangeOut[:,e] = np.array([KnotLocation[max((disIdx[0]-1),0)],KnotLocation[min((disIdx[-1]+1),KnotLocation.size-1)]])
                        break
                else:
                    MinError,disIdx = DisErrorSave.min(0), DisErrorSave.argmin(0)
                    SearchRangeOut[:,ii] = np.array([KnotLocation[max(disIdx - ExpandRange,0)],KnotLocation[min(disIdx + ExpandRange,KnotLocation.size-1)]])
                    OptimalKnotSave[ii] = KnotLocation[disIdx]
                    AngleSave[ii] = DisAnglePsave[disIdx]
                    ErrorSave[ii] = MinError
            else:
                MultiplexMax -= 1
                ScanKnot = True
        
        for e in range(1,min(MultiplexMax,self.spline_degree)+1):
            StartPoint = OptimalKnotSave[e]
            SearchRange = SearchRangeOut[:,e]
            Multiple = e
            OptimalKnotO,ErrorO,Angle = self.GNKnotSolver(Multiple,SearchRange,StartPoint,max_iter)
            if ScanKnot and MultiplexMax == self.spline_order:
                StartPoint = OptimalKnotSave[self.spline_order]
                SearchRange = SearchRangeOut[:,self.spline_order]
                OptimalKnotOs,ErrorOs,Angles = self.GNKnotSolver(Multiple,SearchRange,StartPoint,max_iter)
                if ErrorOs < ErrorO:
                    OptimalKnotO = OptimalKnotOs
                    ErrorO = ErrorOs
                    Angle = Angles
            OptimalKnotSave[e] = OptimalKnotO
            ErrorSave[e] = ErrorO
            AngleSave[e] = Angle
        
        # Check if error is smaller without internal knot
        if np.argmin(ErrorSave)==0:
            OptimalKnotOut = self.x_f.loc[DataKnot]
            MultipleOut = 0
            Ang = AngleSave[0]
            return (OptimalKnotOut,MultipleOut,Ang)
        
        # decide multiple knot
        SmoothOutput = min(self.spline_degree - max_smooth,MultiplexMax)
        AngleOut = AngleSave[1:SmoothOutput+1]
        idx1 = np.where(AngleOut > min_angle)[0]+1
        
        # If idx1 is not empty
        if idx1.size:
            ErrorOut = ErrorSave[idx1]
            minidx = ErrorOut.argmin(0)
            OptimalIdx = idx1[minidx]
            OptimalKnotOut = OptimalKnotSave[OptimalIdx]
            MultipleOut = OptimalIdx
            Ang = AngleSave[OptimalIdx]
        # Otherwise if idx1 is empty, remove the knot
        else:
            idx = 0
            OptimalKnotOut = OptimalKnotSave[idx]
            MultipleOut = idx
            Ang = AngleSave[idx]
        return (OptimalKnotOut,MultipleOut,Ang)

    
    def TwoPcKnotEval (self,Multiple,Knot):
        '''
        Helper function to calculate fit error of a two-piece Bspline, and also
        calculate the "kink" angle at the internal knot separating the two
        segments.
        
        Arguments:
            Multiple: int, the multiplicity of the internal knot
            Knot: float, value of the internal knot
            
        Returns:
            (Error,JoinAngle):
                Error: float, maximum fitting error
                JoinAngle: float, "kink" angle (in degrees) at the knot point
        '''
        # Construct array in the form of [1, Knot, Knot^2, ..., Knot^degree]
        Tcal = np.ones(self.spline_order)
        for e in range(1,self.spline_order):
            Tcal[e] = Tcal[e-1] * Knot
        # Construct array for evaluating the appropriate derivative
        Tcal = Tcal.dot(self.D_mats[self.spline_order - Multiple]) # For knot with multiplicity M, its (Order - M)th derivative is discontinuous (and therefore has a "kink" angle)
        # Get the index of boundary point at knot
        middle = self.KnotOpt_X[self.KnotOpt_X < Knot].index[-1]
        # Get coefficients of Bspline polynomials (as determined by knot values)
        Knot_seq = [self.KnotOpt_X.iloc[0]] + [Knot] * Multiple + [self.KnotOpt_X.iloc[-1]] # Create knot sequence
        Knot_vec = self.gen_kv(Knot_seq) # Generate knot vector (repeat end points)
        Nmat = self.NewMatrix(Knot_vec) # Calculate coefficients
        # Compute Bspline Bases
        N = pd.DataFrame(np.zeros((self.KnotOpt_Y.size,self.spline_order+Multiple)),index = self.KnotOpt_Y.index) # Initiate empty matrix
        Ncal_left = np.reshape(Nmat[:,self.spline_degree],(self.spline_order,self.spline_order),'F') # Get the coefficients for the left segment
        N.loc[:middle,:self.spline_degree] = np.array(self.KnotOpt_T.loc[:middle,:].dot(Ncal_left)) # Build spline bases for the left segment
        Ncal_right = np.reshape(Nmat[:,self.spline_degree + Multiple],(self.spline_order,self.spline_order),'F') # Get the coefficients for the right segment
        N.loc[middle + 1:,Multiple:Multiple+self.spline_degree] = np.array(self.KnotOpt_T.loc[middle + 1:,:].dot(Ncal_right)) # Build spline bases for the right segment
        Pctrl = np.linalg.lstsq(N,self.KnotOpt_Y,rcond = None)[0] # Compute the contrl points
        Error = abs(self.KnotOpt_Y - N.dot(Pctrl)).max() # Find the max error
        ## Calculate join angle
        Sleft = Tcal.dot(Ncal_left)
        DT_left = Sleft.dot(Pctrl[:self.spline_order]) # Calculate derivative value approaching from the left
        Sright = Tcal.dot(Ncal_right)
        DT_right = Sright.dot(Pctrl[Multiple:Multiple+self.spline_order]) # Calculate derivative value approaching from the right
        JoinAngle = abs(np.arctan2(DT_right,1) - np.arctan2(DT_left,1)) * 180 / np.pi
        return (Error,JoinAngle)

    def GNKnotSolver (self,Multiple,SearchRange,StartPoint,max_iter):
        '''
        Helper function, optimize internal knot position for a two-piece spline
        by minimizing mean squares using Gauss-Newton method
        
        Arguments:
            Multiple: int, multiple of the internal knot
            SearchRange: numpy array, specifies the search range for optimization
            StartPoint: float, the initial knot
            max_iter: int, maximum number of iterations
            
        Returns:
            (OptimalKnotO, ErrorO, Angle):
                OptimalKnotO: float, value of optimal knot
                ErrorO: float, maximum error of fit using optimal knot
                Angle: float, "kink" angle at knot point (in degrees)
        '''
        # Specify a very small step size to perform numerical differentiation
        Stepsize = np.sqrt(np.spacing(1))
        # Get the left and right bounds for knot optimization
        KnotLeft = SearchRange[0]
        KnotRight = SearchRange[1]
        # Initialize search
        OptimalKnot = StartPoint
        
        for iterationstep in range(max_iter):
            # Generate knot vector to specify Bspline bases
            Knot_seq = [self.KnotOpt_X.iloc[0]] + [OptimalKnot] * Multiple + [self.KnotOpt_X.iloc[-1]]
            # Slightly change 
            OptimalKnot1 = OptimalKnot + Stepsize
            Knot1_seq = [self.KnotOpt_X.iloc[0]] + [OptimalKnot1] * Multiple + [self.KnotOpt_X.iloc[-1]]
            
            Knot_vec = self.gen_kv(Knot_seq)
            Knot1_vec = self.gen_kv(Knot1_seq)
            # Find index of knot boundary
            middle = self.KnotOpt_X[self.KnotOpt_X < OptimalKnot].index[-1]
            middle1 = self.KnotOpt_X[self.KnotOpt_X < OptimalKnot1].index[-1]
            # Update Nmat
            Nmat = self.NewMatrix(Knot_vec)
            Nmat1 = self.NewMatrix(Knot1_vec)
            #
            Tcal = np.ones(self.spline_order)
            for e in range(1,self.spline_order):
                Tcal[e] = Tcal[e-1] * OptimalKnot
            Tcal = Tcal.dot(self.D_mats[self.spline_order - Multiple])
            # Compute Bspline bases based on Knot and Knot1
            N = pd.DataFrame(np.zeros((self.KnotOpt_Y.size,self.spline_order+Multiple)),index = self.KnotOpt_Y.index)
            N1 = N.copy()
            # Compute Bspline bases for Knot
            Ncal_left = np.reshape(Nmat[:,self.spline_degree],(self.spline_order,self.spline_order),'F')
            N.loc[:middle,:self.spline_degree] = np.array(self.KnotOpt_T.loc[:middle,:].dot(Ncal_left))
            Ncal_right = np.reshape(Nmat[:,self.spline_degree+Multiple],(self.spline_order,self.spline_order),'F')
            N.loc[middle + 1:,Multiple:Multiple + self.spline_degree] = np.array(self.KnotOpt_T.loc[middle + 1:,:].dot(Ncal_right))
            # Compute Bspline bases for Knot1
            Ncal1_left = np.reshape(Nmat1[:,self.spline_degree],(self.spline_order,self.spline_order),'F')
            N1.loc[:middle1,:self.spline_degree] = np.array(self.KnotOpt_T.loc[:middle1,:].dot(Ncal1_left))
            Ncal1_right = np.reshape(Nmat1[:,self.spline_degree + Multiple],(self.spline_order,self.spline_order),'F')
            N1.loc[middle1 + 1:,Multiple:Multiple + self.spline_degree] = np.array(self.KnotOpt_T.loc[middle1 + 1:,:].dot(Ncal1_right))
            # Compute control points for both cases
            Pctrl = np.linalg.lstsq(N,self.KnotOpt_Y,rcond=None)[0]
            Pctrl1 = np.linalg.lstsq(N1,self.KnotOpt_Y,rcond=None)[0]
            # Compute errors for both cases
            Gmat = abs(self.KnotOpt_Y - N.dot(Pctrl))
            Gmat1 = abs(self.KnotOpt_Y - N1.dot(Pctrl1))
            # Compute Jacobian matrix G'mat (derivation of error in respect to knot point)
            Jmat = (Gmat1 - Gmat) / Stepsize
            deltaX = Jmat.dot(Gmat)/(Jmat.dot(Jmat)) # Compute adjustment step size (deltaX is a scalar!)
            if iterationstep == 0:
                deltaXLOld = deltaX
            elif deltaXLOld*deltaX < 0:
                deltaX /= 2
                deltaXLOld = deltaX
            # Increment OptimalKnot position
            OptimalKnot -= deltaX
            ## Calculate angle
            Sleft = Tcal.dot(Ncal_left)
            DT_left = Sleft.dot(Pctrl[:self.spline_order])
            Sright = Tcal.dot(Ncal_right)
            DT_right = Sright.dot(Pctrl[Multiple:Multiple + self.spline_order])
            JoinAngle = abs(np.arctan2(DT_right,1)-np.arctan2(DT_left,1)) * 180 / np.pi
            # Ensure that OptimalKnot is within bounds
            OptimalKnot = max(KnotLeft, OptimalKnot)
            OptimalKnot = min(KnotRight, OptimalKnot)
            Error = Gmat.max()
            # check for stop
            # If iterationstep = 0 (if this is the first cycle)
            if iterationstep == 0:
                # Initialize parameters
                OptimalKnotLOld = OptimalKnot # Initialize "old position" to OptimalKnot (initial input)
                OptimalKnotO = OptimalKnot # Initialize output results to OptimalKnot
                ErrorO = Error # Initialize output Error to initial Error
                Angle = JoinAngle # Initialize output angle to initial JoinAngle
                checkflag = False # Initialize checkflag to False
            # Otherwise, during iteration:
            else:
                # If Error is smaller than previous Error
                if Error < ErrorO:
                    # Update results
                    OptimalKnotO = OptimalKnot
                    ErrorO = Error
                    Angle = JoinAngle
                # If OptimalKnot increment is small
                if (abs(OptimalKnotLOld - OptimalKnot) < 10**-12):
                    # If previous increment was not small
                    if not checkflag:
                        checkflag = True # set checkflag to True, do NOT update stored (old) knot value (check to see if next increment is also small)
                    # If previous increment was also small
                    else:
                        break # break out of loop
                # Otherwise, if OptimalKnot increment is large
                else:
                    # Update stored (old) knot value
                    OptimalKnotLOld = OptimalKnot
                    checkflag = False # Set checkflag to False
            
        return (OptimalKnotO,ErrorO,Angle)
        
    def optimize_knots (self,max_smooth,min_angle,search_range,max_iter,ScanKnot, out = False):
        '''
        Helper method for performing two-piece knot optimization across entire
        length of data
        
        Arguments:
            max_smooth: int, minimum continuity required
            min_angle: float, minimum "kink" angle required at multiple knots
            search_range: int, specifies the increments of the points during
                          optimization
            max_iter: int, maximum number of iterations when doing Gauss-Newton
                      optimization
            ScanKnot: bool, whether to scan all knots
            out: bool, whether to return results, default set to "True"
        '''
        KnotX1 = np.append(self.coarse_knotx[0],self.coarse_knotx[1,-1]).astype(int) # Infer Knot indices from VectorUX
        opt_knot = [] # Initialize empty list to store optimal knots
        join_angles = [] # Initialize empty list to store join angles
        left = KnotX1[0] # Initialize left point (of two-piece spline) to the first point of data region (start from the left)
        opt_knot.append(self.x_f.loc[left]) # Save this point
        # Iterate across all internal knots (excluding the two knots at the ends)
        for i in range(1,KnotX1.size-1):
            # Check that the length from the left to the interior knot is greater than 2 * (degree + 1) (in case that the optimal knot position was moved too close to the next stored knot point)
            if KnotX1[i] - left + 1 > 2 * self.spline_order:
                # Perform two-piece knot optimization
                knot,mult,ang = self.TwoPcOptKnot(left,KnotX1[i],KnotX1[i+1],max_smooth,min_angle,search_range,max_iter,ScanKnot)
                # Update opt_knot
                opt_knot += [knot] * mult
                # if multiple is not 0
                if mult:
                    join_angles.append(ang)
                    left = self.x_f[self.x_f>=knot].index[0] # Update index of the left endpoint (as determined by the optimal knot position)
        # Finally append end point to complete knot sequence
        opt_knot.append(self.x_f.loc[KnotX1[-1]])
        # Save relevant information in attributes
        self.knot_seq = opt_knot # knot_seq: knot sequence (includes the two end points[not repeated], internal multiple knots are repeated)
        self.knot_vec = self.gen_kv(self.knot_seq) # knot_vec: knot vector (internal multiple knots are repeated, end points are repeated)
        self.knot_int, self.knot_multiple = np.unique(self.knot_seq[1:-1], return_counts = True) # knot_int: unique sequence of internal knots (exclues the end points), knot_multiple: array of int with same length as knot_int, specifies the multiplicity of each knot in knot_int
        self.join_angles = join_angles # Save the join angles at each interior knot
        # If out is set to "True", return results
        if out:
            return (opt_knot,join_angles)
        
    def get_FWHM(self):
        '''
        Method to extract full-width at half-maximum (FWHM) of trace, defined 
        as the duration from the half-point of rise to the half-point of decay
        '''
        # Try to deduce FWHM from curve_fit results
        try:
            bg1 = self.best_fit.best_values['bg1']
            delta = self.best_fit.best_values['delta']
            iota = self.best_fit.best_values['iota']
            self.half_max_l =  bg1 + delta / 2
            self.half_max_l_emp = (bg1 + self.Peak_emp) / 2
            self.half_max_r =  bg1 + (delta + iota) / 2
            self.half_max_r_emp = (bg1 + iota + self.Peak_emp) / 2
            T_max = self.best_fit.best_values['T_max']
            self.Tmax_fit = T_max # Peak time point determined by fit
            self.Span_fit = delta - iota # Maximum span determined by fit
            self.Span_emp = self.Peak_emp - bg1 - iota # Empirical span determined by subtracting fit background from max peak value
            self.Peak_fit = bg1 + delta
            # If rise phase is linear
            if 'A' in self.best_fit.model.name:
                slope = - delta / self.best_fit.best_values['alpha'] # compute slope
                self.l_50 = solver_linear(slope, T_max, bg1 + delta, self.half_max_l)
                self.l_50_emp = solver_linear(slope, T_max, bg1 + delta, self.half_max_l_emp)
            # If rise phase is exponential
            elif 'B' in self.best_fit.model.name:
                self.l_50 = solver_one_phase(delta, T_max, - self.best_fit.best_values['k_rise'], bg1, self.half_max_l)
                self.l_50_emp = solver_one_phase(delta, T_max, - self.best_fit.best_values['k_rise'], bg1, self.half_max_l_emp)
            # If decay is linear
            if '0' in self.best_fit.model.name:
                slope = (iota - delta) / self.best_fit.best_values['eta']
                self.r_50 = solver_linear(slope, T_max, bg1 + delta, self.half_max_r)
                self.r_50_emp = solver_linear(slope, T_max, bg1 + delta, self.half_max_r_emp)
            # If decay is one-phase exponential
            elif '1' in self.best_fit.model.name:
                span = delta - iota
                plateau = bg1 + iota
                self.r_50 = solver_one_phase(span, T_max, self.best_fit.best_values['k_slow'], plateau, self.half_max_r)
                self.r_50_emp = solver_one_phase(span, T_max, self.best_fit.best_values['k_slow'], plateau, self.half_max_r_emp)
            # If decay is two-phase exponential
            elif '2' in self.best_fit.model.name:
                span = delta - iota
                plateau = bg1 + iota
                self.r_50 = solver_two_phase(span, T_max, self.best_fit.best_values['percent_fast'], self.best_fit.best_values['k_fast'], self.best_fit.best_values['k_slow'], plateau, self.half_max_r)
                self.r_50_emp = solver_two_phase(span, T_max, self.best_fit.best_values['percent_fast'], self.best_fit.best_values['k_fast'], self.best_fit.best_values['k_slow'], plateau, self.half_max_r_emp)
        # If there is no best_fit attribute, do nothing
        except AttributeError:
            pass
        # Try to deduce FWHM from spline_fit results
        try:
            # Estimate pre-event and post-event baselines (by performing integration between the endpoints and dividing by length of integration region [equivalent of taking the mean for continuous functions])
            bg1 = self.spline_func.integrate(self.x.loc[self.startidx],self.x.loc[self.pre_endidx]) / (self.x.loc[self.pre_endidx] - self.x.loc[self.startidx])
            bg2 = self.spline_func.integrate(self.x.loc[self.post_startidx],self.x.loc[self.endidx]) / (self.x.loc[self.endidx] - self.x.loc[self.post_startidx])
            # Estimate peak value from global maximum of spline
            deriv1 = self.spline_func.derivative(nu=1) # Get BSpline representation of first derivative of BSpline
            x0 = solver_spline(deriv1,0) # Find roots of first derivative (assume roots will exist)
            x0_rel = x0[(x0 > self.x.loc[self.pre_endidx]) & (x0 < self.x.loc[self.post_startidx])] # Limit to roots that are between the pre- and post-event baselines
            y0_rel = self.spline_func(x0_rel) # Get the spline function values corresponding to the roots of first derivative (between the baselines)
            Peak, T_max = y0_rel.max(), x0_rel[np.argmax(y0_rel)] # Find the global maximum and its corresponding x values
            self.Tmax_fit = T_max # Peak time point determined by fit
            self.Span_fit = Peak - bg2 # Span value determined by fit
            self.Peak_fit = Peak
            self.half_max_l = (bg1 + Peak) / 2
            self.half_max_r = (bg2 + Peak) / 2
            left = solver_spline(self.spline_func,self.half_max_l) # Solve for x values that evaluate to half_max_l
            self.l_50 = left[left < T_max][-1] # Define the left point as the largest x value smaller than T_max
            right = solver_spline(self.spline_func,self.half_max_r) # Solve for x values that evaluate to half_max_r
            self.r_50 = right[right > T_max][0] # Define the right point as the smallest x value greater than T_max
            
        # If there is no best_fit attribute, do nothing
        except AttributeError:
            pass
        # Compute full-width half maximum
        try:
            self.FWHM = self.r_50 - self.l_50 # Calculate full-width at half-maximum
            self.FWHM_emp = self.r_50_emp - self.l_50_emp # Calculate full-width at half-maximum determined empirically
            self.RHM_fit = self.r_50 - self.Tmax_fit # Calculate right width at half-maximum using fit values
            self.RHM_emp = self.r_50_emp - self.Tmax_emp # Calculate right width at half-maximum using raw value
            self.LHM_fit = self.Tmax_fit - self.l_50 # Calculate left width at half-maximum using fit values
            self.LHM_emp = self.Tmax_emp - self.l_50_emp # Calculate left width at half-maximum using raw value
            self.delt_Tmax = self.Tmax_emp - self.Tmax_fit # Calculate the difference between empirical and fitted peak timepoint
            self.peak_err = self.delt_Tmax / self.FWHM # Calculate the fit error as a percentage difference between
        except AttributeError: # If there are no attributes named r_50 & l_50, set FWHM to NaN
            self.FWHM = np.nan
            self.FWHM_emp = np.nan
            self.RHM_fit = np.nan
            self.RHM_emp = np.nan
            self.LHM_fit = np.nan
            self.LHM_emp = np.nan
            self.delt_Tmax = np.nan
            self.peak_err = np.nan
    
    def get_duration(self):
        '''
        Method to extract event duration of trace, defined as the duration from
        10 percent of rise to 90 percent of decay
        '''
        # Try to deduce duration from curve_fit results
        try:
            bg1 = self.best_fit.best_values['bg1']
            delta = self.best_fit.best_values['delta']
            iota = self.best_fit.best_values['iota']
            self.rise =  bg1 + delta / 10
            self.decay =  bg1 + (delta + 9* iota)/ 10
            T_max = self.best_fit.best_values['T_max']
            # If rise phase is linear
            if 'A' in self.best_fit.model.name:
                slope = - delta / self.best_fit.best_values['alpha'] # compute slope
                self.t_10 = solver_linear(slope, T_max, bg1 + delta, self.rise)
            # If rise phase is exponential
            elif 'B' in self.best_fit.model.name:
                self.t_10 = solver_one_phase(delta, T_max, - self.best_fit.best_values['k_rise'], bg1, self.rise)
            # If decay is linear
            if '0' in self.best_fit.model.name:
                slope = (iota - delta) / self.best_fit.best_values['eta']
                self.t_90 = solver_linear(slope, T_max, bg1 + delta, self.decay)
            # If decay is one-phase exponential
            elif '1' in self.best_fit.model.name:
                span = delta - iota
                plateau = bg1 + iota
                self.t_90 = solver_one_phase(span, T_max, self.best_fit.best_values['k_slow'], plateau, self.decay)
            # If decay is two-phase exponential
            elif '2' in self.best_fit.model.name:
                span = delta - iota
                plateau = bg1 + iota
                self.t_90 = solver_two_phase(span, T_max, self.best_fit.best_values['percent_fast'], self.best_fit.best_values['k_fast'], self.best_fit.best_values['k_slow'], plateau, self.decay)
        # If there is no best_fit attribute, do nothing
        except AttributeError:
            pass
        # Try to deduce FWHM from spline_fit results
        try:
            # Estimate pre-event and post-event baselines (by performing integration between the endpoints and dividing by length of integration region [equivalent of taking the mean for continuous functions])
            bg1 = self.spline_func.integrate(self.x.loc[self.startidx],self.x.loc[self.pre_endidx]) / (self.x.loc[self.pre_endidx] - self.x.loc[self.startidx])
            bg2 = self.spline_func.integrate(self.x.loc[self.post_startidx],self.x.loc[self.endidx]) / (self.x.loc[self.endidx] - self.x.loc[self.post_startidx])
            # Estimate peak value from global maximum of spline
            deriv1 = self.spline_func.derivative(nu=1) # Get BSpline representation of first derivative of BSpline
            x0 = solver_spline(deriv1,0) # Find roots of first derivative (assume roots will exist)
            x0_rel = x0[(x0 > self.x.loc[self.pre_endidx]) & (x0 < self.x.loc[self.post_startidx])] # Limit to roots that are between the pre- and post-event baselines
            y0_rel = self.spline_func(x0_rel) # Get the spline function values corresponding to the roots of first derivative (between the baselines)
            Peak, T_max = y0_rel.max(), x0_rel[np.argmax(y0_rel)] # Find the global maximum and its corresponding x values
            self.rise = (9 * bg1 + Peak) / 10
            self.decay = (9 * bg2 + Peak) / 10
            left = solver_spline(self.spline_func,self.rise) # Solve for x values that evaluate to half_max_l
            self.t_10 = left[left < T_max][-1] # Define the left point as the largest x value smaller than T_max
            right = solver_spline(self.spline_func,self.decay) # Solve for x values that evaluate to half_max_r
            self.t_90 = right[right > T_max][0] # Define the right point as the smallest x value greater than T_max
        # If there is no best_fit attribute, do nothing
        except AttributeError:
            pass
        # Compute duration
        try:
            self.duration = self.t_90 - self.t_10
        except AttributeError: # If there are no attributes named t_90 & t_10, set duration to NaN
            self.duration = np.nan
    
    def post_process(self,out = False):
        '''
        Method of post processing, extracts kinetic parameters and vesicle
        information
        
        Argument:
            out: bool, if set to "True", return post processing results
        
        Returns:
            dict of appropriate values
        '''
        ## Extract kinetic parameters
        self.get_FWHM() # Get FWHM
        self.get_duration() # Get duration
        ## Calculate goodness of fit statistics
        # Calculate residual degrees of freedom
        try:
            self.df = self.y_val.size - self.best_fit.nvarys # If curve fit was performed, degrees of freedom is the number of data points minus number of parameters in best fit model
        except AttributeError:
            try:
                self.df = self.y_f.size - (self.knot_int.size + self.spline_order) # If spline fit was performed, degrees of freedom is the number of data points minus the degrees of freedom of the spline
            except AttributeError:
                self.df = np.nan # If no fit was performed, set degrees of freedom to np.nan
        # Calculate root mean squared error (RMSE)
        try:
            self.RMSE = np.sqrt(self.residuals.dot(self.residuals) / self.df)
        except AttributeError:
            self.RMSE = np.nan # If there is no residuals attribute, set RMSE to NaN
        # Locate immediate region around exocytosis event, test normality of errors in immeidate region (using Shapiro-Wilk test)
        try:
            t_l = self.t_10 - 2 * self.FWHM
            t_r = self.t_90 + 2 * self.FWHM
            try:
                idxs = (self.x_f >= t_l) & (self.x_f <= t_r) # In the case of spine fit
            except AttributeError:
                idxs = (self.x_val >= t_l) & (self.x_val <= t_r) # In the case of curve fit
            temp = self.residuals.where(idxs) # Change values outside of immediate region to NaN
            self.event_startidx = temp.first_valid_index()
            self.event_endidx = temp.last_valid_index()
            # If there are fewer than 30 data points within region of concern, extend region to include at least 30 data points
            if idxs.sum() < 30:
                addit = 30 - idxs.sum() # Find number of additonal points needed
                front = addit // 3 # Add 1/3 of additional points to front
                back = addit - front # Add remainder of additional points to back
                self.event_startidx = max(temp.index[0],self.event_startidx - front) # Get new index of left point
                self.event_endidx = min(temp.index[-1],self.event_endidx + back) # Get new index of right point
                idxs.loc[self.event_startidx:self.event_endidx] = True # Change all values within new end points to "True"
            # Calculate p_value from D'Agosto and Pearson's K Squared test for normality
            self.res_pval = stats.normaltest(self.residuals[idxs])[1]
            self.res_skew = stats.skew(self.residuals[idxs]) # Calculate skew
            self.res_kurt = stats.kurtosis(self.residuals[idxs],fisher = False) # Calculate kurtosis (Using Pearson's definition)
        # If the attributes do not exist, trace was not fit (either by curve fitting or spline fitting), all results are NaN
        except AttributeError:
            self.res_pval = np.nan
            self.res_skew = np.nan
            self.res_kurt = np.nan
        # If out is set to "True", return results
        if out:
            res = {}
            res['FWHM'] = self.FWHM
            res['FWHM_emp'] = self.FWHM_emp
            res['Duration'] = self.duration
            res['RHM_emp'] = self.RHM_emp
            res['RHM_fit'] = self.RHM_fit
            res['delt_Tmax'] = self.delt_Tmax
            res['peak_err'] = self.peak_err
            try:
                res['Event Start'] = self.t_10
            except AttributeError: # If there is no attribute called t_10, set 'Event Start' to NaN
                res['Event Start'] = np.nan
            try:
                res['Span_emp'] = self.Span_emp
            except AttributeError: # If there is no attribute called t_10, set 'Event Start' to NaN
                res['Span_emp'] = np.nan
            try:
                res['Span_fit'] = self.Span_fit
            except AttributeError: # If there is no attribute called t_10, set 'Event Start' to NaN
                res['Span_fit'] = np.nan
            res['Root Mean Squared Error'] = self.RMSE
            res['p value of Normality'] = self.res_pval
            res['Residual Skew'] = self.res_skew
            res['Residual Kurtosis'] = self.res_kurt
            res['Vesicle Type'] = self.ves_type
            res['Remarks'] = self.remark
            return res
    
    def plot_fit(self,save_plot = False,show_plot = True,rasterize = False,dpi = 300,out = False,save_path = None):
        '''
        Method to plot and save least squares fitting results
        '''
        # If fit_method attribute is None, return None - do not plot
        if self.fit_method is None:
            return None
        # Create a figure object
        fig, ax1 = plt.subplots(nrows = 1, ncols = 1,num = self.name,figsize = (16,9),dpi = dpi,clear = True)
        # Set up parameters for plotting FWHM & Duration
        try:
            top = max(self.Peak_emp,self.Peak_fit)
            bottom = self.y_val.min()
            span = top - bottom
            fwhm_pos = top + span / 10
            duration_pos = bottom - span / 10
            emp_pos = min(self.half_max_r_emp,self.half_max_l_emp)
        except AttributeError:
            pass
        # Set up axes title and axes labels
        ax1.set_xlabel(xlabel = 'Time (s)') # Set x axis label
        ax1.set_ylabel(ylabel = 'A.F.U.') # Set y axis label
        # Plot original raw data
        ax1.plot(self.x_val,self.y_val,'.', label = 'Raw Data')
        try:
            ax1.plot(self.x_val.loc[self.event_startidx:self.event_endidx],self.y_val.loc[self.event_startidx:self.event_endidx],'b.', label = 'Event')
        except AttributeError:
            pass
        # Try plotting curve fit results
        try:
            x_spike = self.x_val[self.spike_index] # Get the x values in the spike region
            x_dense = np.linspace(x_spike.iloc[0],x_spike.iloc[-1],num = self.N_spike * 20) # Create dense x array in the region of spike
            x_rest = self.x_val[~self.spike_index] # Get the x values outside the spike region
            pos = x_rest.index.get_loc(x_spike.index[-1]+1) # Find insert position
            x = np.insert(x_rest.values,pos,x_dense) # Insert x_dense in the middle of x_rest, where the spike values were - replace the original x values of spike segment with much denser array of x (to capture rapid change of the curve)
            y = self.curve_func(x) # Compute y values according to best fit model
            ax1.plot(x,y,'r-', label = 'Best Fit') # Plot best fit model using constructed x array
            ax1.set_title(label = self.best_fit.model.name[6:-1]) # Set title of axis
            timespan = self.x_val.iloc[-1] - self.x_val.iloc[0] # Find the timespan of event
            if self.FWHM / timespan < 0.05: # If FWHM is less than 5% of event timespan
                ax1_ins = inset_axes(ax1,width = '30%',height = '50%',loc='upper right') # Create inset
                ax1_ins.set_xlim(left = self.t_10 - timespan * 0.01, right = self.t_90 + timespan * 0.01) # Set limits of x values in inset to encompass duration
                # Plot original data and best-fit curve
                ax1_ins.plot(self.x_val,self.y_val,'.', label = 'Raw Data')
                ax1_ins.plot(self.x_val.loc[self.event_startidx:self.event_endidx],self.y_val.loc[self.event_startidx:self.event_endidx],'b.',label = 'Event')
                ax1_ins.plot(x,y,'r-', label = 'Best Fit') # Plot best fit model using constructed x array
                ## Draw full-width half max (projecting upward)
                ax1_ins.plot([self.l_50,self.l_50],[self.half_max_l,fwhm_pos],'k:')
                ax1_ins.plot([self.l_50,self.l_50],[self.half_max_l,fwhm_pos],'c.')
                ax1_ins.plot([self.r_50,self.r_50],[self.half_max_r,fwhm_pos], 'k:')
                ax1_ins.plot([self.r_50,self.r_50],[self.half_max_r,fwhm_pos], 'c.')
                ax1_ins.plot([self.l_50,self.r_50],[fwhm_pos, fwhm_pos],'c--',label = 'FWHM')
                ## Draw duration (projecting downward)
                ax1_ins.plot([self.t_10,self.t_10],[self.rise, duration_pos], 'k:')
                ax1_ins.plot([self.t_10,self.t_10],[self.rise, duration_pos], 'm.')
                ax1_ins.plot([self.t_90,self.t_90],[self.decay, duration_pos], 'k:')
                ax1_ins.plot([self.t_90,self.t_90],[self.decay, duration_pos], 'm.')
                ax1_ins.plot([self.t_10,self.t_90],[duration_pos, duration_pos], 'm--',label = 'Duration')
                ## Plot peak positions
                ax1_ins.plot([self.Tmax_emp,self.Tmax_emp],[duration_pos,self.Peak_emp],'y-.')
                ax1_ins.plot([self.Tmax_fit,self.Tmax_fit],[duration_pos,self.Peak_fit],'k-.')
                ax1_ins.plot(self.Tmax_emp,self.Peak_emp,'yx',self.Tmax_fit,self.Peak_fit,'k+') # Label Peak points
                ## Plot empirical values
                if self.half_max_l_emp > self.half_max_r_emp: # If the half-rise point is higher than half-decay point
                    ax1_ins.plot([self.l_50_emp,self.l_50_emp],[emp_pos,self.half_max_l_emp],'k:')
                    ax1_ins.plot([self.l_50_emp,self.l_50_emp],[emp_pos,self.half_max_l_emp],'y.')
                elif self.half_max_l_emp < self.half_max_r_emp: # If the half-rise point is lower than half-decay point
                    ax1_ins.plot([self.r_50_emp,self.r_50_emp],[emp_pos,self.half_max_r_emp],'k:')
                    ax1_ins.plot([self.r_50_emp,self.r_50_emp],[emp_pos,self.half_max_r_emp],'y.')
                ax1_ins.plot([self.l_50_emp,self.r_50_emp],[emp_pos,emp_pos],'y.')
                ax1_ins.plot([self.l_50_emp,self.r_50_emp],[emp_pos,emp_pos],'y--',label = 'FWHM_emp')
                ## Prepare figure legend
                h,l = ax1_ins.get_legend_handles_labels() # Get handles and labels from ax1
            else:
                ## Draw full-width half max (projecting upward)
                ax1.plot([self.l_50,self.l_50],[self.half_max_l,fwhm_pos],'k:')
                ax1.plot([self.l_50,self.l_50],[self.half_max_l,fwhm_pos],'c.')
                ax1.plot([self.r_50,self.r_50],[self.half_max_r,fwhm_pos], 'k:')
                ax1.plot([self.r_50,self.r_50],[self.half_max_r,fwhm_pos], 'c.')
                ax1.plot([self.l_50,self.r_50],[fwhm_pos, fwhm_pos],'c--', label = 'FWHM')
                ## Draw duration (projecting downward)
                ax1.plot([self.t_10,self.t_10],[self.rise, duration_pos], 'm.')
                ax1.plot([self.t_10,self.t_10],[self.rise, duration_pos], 'k:')
                ax1.plot([self.t_90,self.t_90],[self.decay, duration_pos], 'm.')
                ax1.plot([self.t_90,self.t_90],[self.decay, duration_pos], 'k:')
                ax1.plot([self.t_10,self.t_90],[duration_pos, duration_pos], 'm--', label = 'Duration')
                ## Plot peak positions
                ax1.plot([self.Tmax_emp,self.Tmax_emp],[duration_pos,self.Peak_emp],'y--')
                ax1.plot([self.Tmax_fit,self.Tmax_fit],[duration_pos,self.Peak_fit],'k-.',label = 'Fit Peak')
                ax1.plot(self.Tmax_emp,self.Peak_emp,'kx',self.Tmax_fit,self.Peak_fit,'k+') # Label Peaks
                ## Plot empirical values
                if self.half_max_l_emp > self.half_max_r_emp: # If the half-rise point is higher than half-decay point
                    ax1.plot([self.l_50_emp,self.l_50_emp],[emp_pos,self.half_max_l_emp],'k:')
                    ax1.plot([self.l_50_emp,self.l_50_emp],[emp_pos,self.half_max_l_emp],'y.')
                elif self.half_max_l_emp < self.half_max_r_emp: # If the half-rise point is lower than half-decay point
                    ax1.plot([self.r_50_emp,self.r_50_emp],[emp_pos,self.half_max_r_emp],'k:')
                    ax1.plot([self.r_50_emp,self.r_50_emp],[emp_pos,self.half_max_r_emp],'y.')
                ax1.plot([self.l_50_emp,self.r_50_emp],[emp_pos,emp_pos],'y.')
                ax1.plot([self.l_50_emp,self.r_50_emp],[emp_pos,emp_pos],'y--',label = 'FWHM_emp')
                ## Prepare figure legend
                h,l = ax1.get_legend_handles_labels() # Get handles and labels from ax1
        # If error (no attribute), do nothing
        except AttributeError:
            pass
        # Try plotting spline fit results
        try:
            ax1.plot(self.x_val,self.y_val_sm,'g.',label = 'Smoothed Data')
            ax1.plot(self.x_val,self.spline_interp(self.x_val),'y-',label = 'Interpolated')
            ax1.plot(self.x_val,self.spline_func(self.x_val), 'r-', label = 'Best Fit')
            ax1.set_title(label = 'C: cubic spline')
            ## Draw full-width half max (projecting upward)
            ax1.plot([self.l_50,self.l_50],[self.half_max_l,fwhm_pos],'k:')
            ax1.plot([self.l_50,self.l_50],[self.half_max_l,fwhm_pos],'c.')
            ax1.plot([self.r_50,self.r_50],[self.half_max_r,fwhm_pos], 'k:')
            ax1.plot([self.r_50,self.r_50],[self.half_max_r,fwhm_pos], 'c.')
            ax1.plot([self.l_50,self.r_50],[fwhm_pos, fwhm_pos],'c--', label = 'FWHM')
            ## Draw duration (projecting downward)
            ax1.plot([self.t_10,self.t_10],[self.rise, duration_pos], 'm.')
            ax1.plot([self.t_10,self.t_10],[self.rise, duration_pos], 'k:')
            ax1.plot([self.t_90,self.t_90],[self.decay, duration_pos], 'm.')
            ax1.plot([self.t_90,self.t_90],[self.decay, duration_pos], 'k:')
            ax1.plot([self.t_10,self.t_90],[duration_pos, duration_pos], 'm--', label = 'Duration')
            ## Plot peak positions
            ax1.plot([self.Tmax_emp,self.Tmax_emp],[duration_pos,self.Peak_emp],'y--')
            ax1.plot([self.Tmax_fit,self.Tmax_fit],[duration_pos,self.Peak_fit],'k-.',label = 'Fit Peak')
            ax1.plot(self.Tmax_emp,self.Peak_emp,'kx',self.Tmax_fit,self.Peak_fit,'k+') # Label Peak points
            ## Plot empirical values
            if self.half_max_l_emp > self.half_max_r_emp: # If the half-rise point is higher than half-decay point
                ax1.plot([self.l_50_emp,self.l_50_emp],[emp_pos,self.half_max_l_emp],'k:')
                ax1.plot([self.l_50_emp,self.l_50_emp],[emp_pos,self.half_max_l_emp],'y.')
            elif self.half_max_l_emp < self.half_max_r_emp: # If the half-rise point is lower than half-decay point
                ax1.plot([self.r_50_emp,self.r_50_emp],[emp_pos,self.half_max_r_emp],'k:')
                ax1.plot([self.r_50_emp,self.r_50_emp],[emp_pos,self.half_max_r_emp],'y.')
            ax1.plot([self.l_50_emp,self.r_50_emp],[emp_pos,emp_pos],'y.')
            ax1.plot([self.l_50_emp,self.r_50_emp],[emp_pos,emp_pos],'y--',label = 'FWHM_emp')
            ## Prepare figure legend
            h,l = ax1.get_legend_handles_labels() # Get handles and labels from ax1
        # If error (no attribute), do nothing
        except AttributeError:
            pass
        fig.legend(handles=h,labels=l,loc = 'upper right')
        # If out is set to "True", return figure
        if out:
            return (fig, ax1)
        # If save_plot is "True", save plot to pdf file
        if save_plot:
            # If save_path is not given, save file in current directory
            if save_path is None:
                fig.savefig(f'{self.name}.pdf', dpi = 'figure', format = 'pdf')
            # Otherwise, save file in designated path
            else:
                fig.savefig(f'{save_path}\\{self.name}.pdf', dpi = 'figure', format = 'pdf')
        # Show figure if show_plot is set to "True"
        if show_plot:
            fig.show()
        # If show_plot is set to "False", close figure (release from memory)
        else:
            plt.close(fig)    

###############################################################################
##############  Cell class, deals with all ROIs in a given cell  ##############
###############################################################################

class cell(object):
    def __init__(self,timetrace,measurement,path):
        '''
        Instantiating a cell class requires the following:
            timetrace: str, file name of csv file containing time traces
            measurement: str, file name of csv file containing measurements
            path: str, path name of folder which contains the csv files
        '''
        self.timetrace = timetrace # Save timetrace filename to timetrace attribute
        self.filename = measurement.split('_')[-1] # Infer filename from measurement file (includes file extension)
        self.cellnum = int(self.filename.split('.')[0]) # Infer cell number from filename (assume all file names are in the form of Measurement_##.csv), convert to int datatype
        self.Slices = get_slices(measurement)
        # If not existent, create folder in Analysis named cellnum to store results
        self.path_common = f'{path}\\Analysis\\{self.cellnum}'
        if not os.path.isdir(self.path_common):
            os.mkdir(self.path_common)
        self.path_preprocess = f'{self.path_common}\\pre_process'
        self.path_fit = f'{self.path_common}\\fit'
        self.ROI_list = self.Slices.index # Get list/iterable of all ROIs of this cell
        self.ROI_count = self.Slices.size # Get the number of all ROIs of this cell
        self.ROIs = pd.Series(np.nan,index = self.ROI_list, dtype = object) # Create empty pandas Series
        
    def pre_process (self,show_plot = False,save_plot = True,save_result = True,out = False):
        '''
        Function for pre-processing ROIs for further analysis
        
        Arguments:
            show_plot: bool, if "True", shows plots. Default value is "False".
            save_plot: bool, if "True", saves plots. Default value is "True".
            save_result: bool, if "True", saves result to appropriate subfolder.
                         Default is "False"
            out: bool, if "True", returns result. Default value is "True"
            
        Returns:
            pandas DataFrame object containing preprocessing results for all traces
            stored in timetrace file
        '''
        res_idx = ['startidx','pre_endidx','Slice','post_startidx','endidx','spike','spike_start','spike_end','process','remark','ves_type','fit_method'] # These are the attributes to save
        self.preprocess_res = pd.DataFrame(np.nan,index = res_idx,columns = self.ROI_list) # Create emtpy pandas DataFrame file filled with NaNs to store results
        print(f'\nPre-processing cell #{self.cellnum}')
        Data = column_generator(self.timetrace) # Read Data from timetrace file
        t = next(Data) # Get timestamps
        # If save_plot of save_result is "True", 
        if save_plot or save_result:
            # If savepath folder does not exist, make a new folder (to store graphs)
            if not os.path.isdir(self.path_preprocess):
                os.mkdir(self.path_preprocess)
        # Iterate through each trace in timetrace file
        for raw in Data:
            roi = trace(t,raw,self.Slices.loc[raw.name])
            self.ROIs.loc[roi.name] = roi # Save trace object in ROIs attribute
            print(f'\nPre-processing {roi.name}')
            ans = roi.pre_process(out=True) # Pre-process trace
            roi.plot_features(save_plot=save_plot,show_plot=show_plot,save_path=self.path_preprocess) # Plot and save trace features
            self.preprocess_res[roi.name] = pd.Series(ans) # Store pre-process results in appropriate column of preprocess_res
        # If save_result is set to "True", save result to csv file
        if save_result:
            self.preprocess_res.to_csv(f'{self.path_preprocess}\\{self.filename}')
        # If out is set to "True", return result
        if out:
            return self.preprocess_res
        
    def post_process(self,show_plot = False,read_file = True,save_result = True,out = False):
        '''
        Method to perform least squares regression (using appropriate method)
        based on pre-processing results
        
        Arguments:
            show_plot: bool, if "True", shows plots. Default value is "True".
            read_file: bool, if "True", read previously saved '.csv' file from
                       appropriate read_path. Default value "True"
            save_result: bool, if "True", saves result to appropriate subfolder.
                         Default is "False"
            out: bool, if "True", returns result. Default value is "True"
            
        Returns:
            pandas DataFrame object containing preprocessing results for all traces
            stored in timetrace file
        '''
        # If read_file set to True
        if read_file:
            # If savepath folder does not exist, raise exception
            if not os.path.isdir(self.path_preprocess):
                raise Exception(f'Cell # {self.cellnum} has not yet been pre-processed!')
            # Update preprocess_res attribute
            self.preprocess_res = pd.read_csv(f'{self.path_preprocess}\\{self.filename}',index_col = 0) # Read prior saved result
        res_idx = ['Remarks','Vesicle Type', 'FWHM', 'FWHM_emp', 'RHM_emp', 'RHM_fit', 'Duration', 'Event Start','Root Mean Squared Error','p value of Normality','Residual Skew','Residual Kurtosis', 'peak_err', 'delt_Tmax', 'Span_fit', 'Span_emp', 'Event Start']
        self.fit_res = pd.DataFrame(np.nan,index = res_idx,columns = self.ROI_list)
        print(f'\nProcessing cell #{self.cellnum}')
        Data = column_generator(self.timetrace) # Read Data from timetrace file
        t = next(Data) # Get timestamps
        # Iterate over each trace in ROIs attribute
        if save_result:
            # If save directory does not exist, create folder
            if not os.path.isdir(self.path_fit):
                os.mkdir(self.path_fit)
        for raw in Data:
            roi = trace(t,raw,self.Slices.loc[raw.name])
            self.ROIs.loc[roi.name] = roi # Save trace object in ROIs attribute
            roi.set_values(**self.preprocess_res[roi.name].to_dict()) # Directly set parameters as stored in preprocess_res attribute
            # If fit_method attribute of trace object is 'curve'
            if roi.fit_method == 'curve':
                print(f'Fitting {roi.name} by non-linear curve fit...')
                roi.curve_fit(report = False) # Fit using non-linear least squares curve fit
            # Otherwise if fit_method attribute of trace object is 'spline'
            elif roi.fit_method == 'spline':
                print(f'Fitting {roi.name} by spline regression...')
                roi.spline_fit() # Fit using free-knot spline regression
            # Otherwise skip and do not fit ROI
            else:
                print(f'Skipping {roi.name}')
            # Post process cells
            ans = roi.post_process(out = True)
            self.fit_res[roi.name] = pd.Series(ans) # Store post-process results in appropriate column of fit_res
            roi.plot_fit(save_plot = True,show_plot = show_plot,save_path = self.path_fit) # Plot and save figures
        # If save_result is set to "True", save result to csv file
        if save_result:
            self.fit_res.to_csv(f'{self.path_fit}\\{self.filename}')
        # If out is set to "True", return result
        if out:
            return self.fit_res

###############################################################################
################# Experiment class, deals with entire dataset #################
###############################################################################

class EXP(object):
    def __init__(self,path):
        '''
        Instantiating an EXP class requires the following:
            path: str, directory path of folder which holds all the information
        '''
        self.path = path # Save path to path attribute
        self.name = path.split('\\')[-1] # Save experiment name as the name of the folder
        # If not already existent, create folder in path named "Analysis" to store results
        self.result_path = f'{path}\\Analysis'
        if not os.path.isdir(self.result_path):
            os.mkdir(self.result_path)
        os.chdir(path) # Change directory to specified path
        files = np.char.array(os.listdir(path)) # Generate a numpy chararray of all the file names in folder
        files.sort()
        CSV_idxs = files.find('.csv') > 0 # Detect the indices of all '.csv' files
        CSVs = files[CSV_idxs] # Get list of all CSV files
        # Try to read blind key file
        try:
            self.blind_key = pd.read_csv('Blind key.csv',header = None, names = ['Blind Key'],index_col = 1,squeeze = True) # Read blind key and save contents to blind_key attribute
        # If file not found
        except FileNotFoundError:
            raise FileNotFoundError('Could not find blind key!')
        MNT_idx = CSVs.find('Measurement_') == 0 # Find the indices of measurement files
        TT_idx = CSVs.find('Time Trace(s)_') == 0 # Find the indices of timetrace files
        # If the number of measurement files and timetrace files do not match, raise exception
        if MNT_idx.sum() != TT_idx.sum():
            raise Exception('Unequal number of Time Trace and Measurement files!')
        self.measurements = CSVs[MNT_idx] # Get numpy chararray containing filenames of measurement files
        self.timetraces = CSVs[TT_idx] # Get numpy chararray containing filenames of timetrace files
        self.cell_list = np.array(self.measurements.lstrip('Measurement_').rstrip('.csv'),dtype = int) # Generate the list of cells (inferred from names of measurement files, should be an array of integer numbers)
        self.cell_count = self.cell_list.size # Get total number of cells
        self.Cells = pd.Series(np.nan,index = self.cell_list) # Initialize empty pandas Series object to store cell objects (store as program progresses)
        
    def pre_process(self,start_at = None):
        '''
        Method to pre-process all ROIs in all cells.
        
        Argument:
            start_at: int, cell number to start iteration. Cell number is the
                      number following the '_' character in the timetrace / 
                      measurement file names
        '''
        # Determine start point of iteration
        if start_at is None:
            n = 0
        else:
            # Try converting start_at to int type
            try:
                start_at = int(start_at)
            # If error, raise exception
            except ValueError:
                raise TypeError('Cell number must be int')
            # Find the index of start_at value in cell_list
            try:
                n = int(np.where(self.cell_list == start_at)[0])
            # If n cannot be converted to integer, cell number was not found
            except TypeError:
                raise ValueError('Cell number not found')
        # Iterate over Cell objects
        for i in range(n,self.cell_count):
            self.current_cell = self.cell_list[i]
            CELL = cell(self.timetraces[i],self.measurements[i],self.path) # Create cell instance
            self.Cells.loc[CELL.cellnum] = CELL # Store cell instance in Cells attribute
            CELL.pre_process() # Preprocess ROIs in current cell
            key = input('Enter "exit" to exit, otherwise program will continue\n')
            if key == 'exit':
                print(f'Cell #{self.current_cell} was just processed')
                if i == self.cell_count-1:
                    print(f'Cell #{self.current_cell} was the last cell!')
                else:
                    print(f'Next cell is #{self.cell_list[i+1]}')
                break
            else:
                continue
    
    def post_process(self, read_file = True):
        '''
        Method to fit data and extract kinetic parameters
        
        Argument:
            read_file: if "True", read pre-processing results from saved .csv
                       file. Default set to "True".
        '''
        # Iterate over all Cell objects
        for i in range(self.cell_count):
            self.current_cell = self.cell_list[i]
            CELL = cell(self.timetraces[i],self.measurements[i],self.path) # Create cell instance
            self.Cells.loc[CELL.cellnum] = CELL # Store cell instance in Cells attribute
            CELL.post_process(read_file = read_file) # Read previously saved pre-process results and fit curve
            print(f'Finished processing cell #{CELL.cellnum}')
    
    def summarize(self):
        '''
        Method to summarize data (after post-processing)
        
        Saves summary information in an excel file
        '''
        self.unblind = self.blind_key.str.split('_Cell').str[0] # Split each element in blind_key Series by splitting at '_Cell', and only keep the part before '_Cell'. (Assumes original movie files were named "condition_Cell###.nd2")
        self.conditions, counts = np.unique(self.unblind,return_counts = True) # Find and count number of unique elements in unblind
        shared_index = ['Docked','New Arrival','Not Determined','Total Events','Events Analyzed','Events Accepted','FWHM Mean','FWHM Std Dev','Duration Mean','Duration Std Dev']
        # Create DataFrame full of zeros to store grand summary results
        grand_sum_idx = shared_index.copy()
        grand_sum_idx.extend(['Number of Cells','Number of Cells Analyzed'])
        grand_summary = pd.DataFrame(0,index = grand_sum_idx,columns = self.conditions)
        grand_summary.loc['Number of Cells'] = pd.Series(counts,index = self.conditions) # Fill in "Number of Cells" row
        # Create empty dataframe to store cell by cell results
        cell_by_cell_cols = ['Cell Name','Condition','ROIs Not Fitted','ROIs Rejected','ROIs Accepted','ROIs with large peak errors']
        cell_by_cell_cols.extend(shared_index)
        cell_by_cell = pd.DataFrame(np.nan,index = self.cell_list,columns = cell_by_cell_cols)
        # Create temporary dictionary to store lists
        temp = {}
        # Iterate across all conditions
        for cond in self.conditions:
            temp[cond] = {'FWHM':[],'FWHM_emp':[],'Duration':[],'Ves_Type':[],'RHM_fit':[],'RHM_emp':[],'Span_emp':[],'Span_fit':[]} # For each condition, create a dictionary of empty list to store results
        # Iterate across all cells
        for cell in self.cell_list:
            cond = self.unblind[cell] # Find the condition of current cell (from unblinded key)
            cell_by_cell.loc[cell,'Cell Name'] = self.blind_key[cell] # Get cell name (before blinding)
            cell_by_cell.loc[cell,'Condition'] = self.unblind[cell] # Get cell condition
            # Try to read fit results from file
            try:
                fit_res = pd.read_csv(f'{self.result_path}\\{cell}\\fit\\{cell}.csv',index_col=0)
                _,event_num = fit_res.shape # Infer the number of events from shape of fit_res (= number of columns)
                p_vals = fit_res.loc['p value of Normality'].astype(np.float_) # get p values, convert datatype from "str" to "np.float64"
                peak_errs = fit_res.loc['peak_err'].astype(np.float_) # get fit error values, convert datatype from "str" to "np.float64"
                idx_not_fitted = p_vals.isnull() # Find the entries in p_vals that are NaN, these correspond to ROIs in this cell that was not fitted
                idx_reject = p_vals < 0.05 # Find the entries in p_vals that are less than 0.05, these are ROIs whose fit are rejected (and subject to manual inspection and processing)
                idx_accept = p_vals >=0.05 # Find the entries whose p value is >= 0.05
                idx_large_errs = peak_errs >= 0.05 # Find the entries whose peak erros are greater than 5 %
                # Use the boolean arrays generated above to find the indices, which are the ROIs
                rois_not_fitted = p_vals[idx_not_fitted].index
                rois_rejected = p_vals[idx_reject].index
                rois_accepted = p_vals[idx_accept].index
                rois_large_errs = peak_errs[idx_large_errs].index
                fwhm = fit_res.loc['FWHM'].astype(np.float_)[rois_accepted] # get FWHMs, convert datatype from "str" to "np.float64"
                fwhm_emp = fit_res.loc['FWHM_emp'].astype(np.float_)[rois_accepted] # get FWHM_emp, convert datatype from "str" to "np.float64"
                dur = fit_res.loc['Duration'].astype(np.float_)[rois_accepted] # get durations, convert datatype from "str" to "np.float64"
                rhm_fit = fit_res.loc['RHM_fit'].astype(np.float_)[rois_accepted] # get RHM fit values, convert datatype from "str" to "np.float64"
                rhm_emp = fit_res.loc['RHM_emp'].astype(np.float_)[rois_accepted] # get RHM empirical values, convert datatype from "str" to "np.float64"
                span_emp = fit_res.loc['Span_emp'].astype(np.float_)[rois_accepted] # get Sapn_emp values, convert datatype from "str" to "np.float64"
                span_fit = fit_res.loc['Span_fit'].astype(np.float_)[rois_accepted] # get Span_fit values, convert datatype from "str" to "np.float64"
                acc_events = idx_accept.sum() # Count the number of "True" values in idx_accept as number of events accepted
                anal_events = p_vals.count() # Count the number of non-null events in p_vals as number of events analyzed
                ves_types = fit_res.loc['Vesicle Type'] # get vesicle types
                ves_type, ves_count = np.unique(ves_types,return_counts = True) # Get counts of vesicle types
                ves_counts = pd.Series(ves_count,index = ves_type)
                ## Update values in cell-by-cell summary
                # Concatenate string elements into a single string and store in appropriate position
                try:
                    cell_by_cell.loc[cell,'ROIs Not Fitted'] = rois_not_fitted.str.cat(sep=',')
                except AttributeError:
                    pass
                try:
                    cell_by_cell.loc[cell,'ROIs Rejected'] = rois_rejected.str.cat(sep=',')
                except AttributeError:
                    pass
                try:
                    cell_by_cell.loc[cell,'ROIs Accepted'] = rois_accepted.str.cat(sep=',')
                except AttributeError:
                    pass
                try:
                    cell_by_cell.loc[cell,'ROIs with large peak errors'] = rois_large_errs.str.cat(sep=',')
                except AttributeError:
                    pass
                cell_by_cell.loc[cell,'Total Events'] = event_num # Update total event number in cell by cell event count
                cell_by_cell.loc[cell,'Events Analyzed'] = anal_events # Count the number of non-NaNs and save as number of events analyzed in cell-by-cell summary
                cell_by_cell.loc[cell,'Events Accepted'] = acc_events # Count the number of "True" values in idx_accept as number of events accepted
                cell_by_cell.loc[cell,ves_type] = ves_counts # Update the counts in cell by cell summary
                cell_by_cell.loc[cell,'FWHM Mean'] = fwhm.mean() # Calculate and store cell mean of FWHM
                cell_by_cell.loc[cell,'FWHM Std Dev'] = fwhm.std() # Calculate and store cell std dev of FWHM
                cell_by_cell.loc[cell,'Duration Mean'] = dur.mean() # Calculate and store cell mean of Duration
                cell_by_cell.loc[cell,'Duration Std Dev'] = dur.std() # Calculate and store cell std dev of Duration
                temp[cond]['FWHM'].extend(fwhm.dropna()) # Drop NaNs and add FWHM values to list (of appropriate condition)
                temp[cond]['FWHM_emp'].extend(fwhm_emp.dropna()) # Drop NaNs and add FWHM_emp values to list (of appropriate condition)
                temp[cond]['Duration'].extend(dur.dropna()) # Drop NaNs and add FWHM values to list (of appropriate condition)
                temp[cond]['Ves_Type'].extend(ves_types[rois_accepted]) # Add values in ves_types corresponding to non null values in FWHM to list (of appropriate condition)
                temp[cond]['RHM_fit'].extend(rhm_fit.dropna()) # Drop NaNs and add RHM_fit values to list
                temp[cond]['RHM_emp'].extend(rhm_emp.dropna()) # Drop NaNs and add RHM_emp values to list
                temp[cond]['Span_emp'].extend(span_emp.dropna()) # Drop NaNs and add Span_emp values to list
                temp[cond]['Span_fit'].extend(span_fit.dropna()) # Drop NaNs and add Span_fit values to list
                ## Update values in grand summary
                grand_summary.loc['Number of Cells Analyzed',cond] += 1 # Increment the count of "Number of Cells Analyzed" for the appropriate condition
                grand_summary.loc[ves_type,cond] += ves_counts
                grand_summary.loc['Total Events',cond] += event_num
                grand_summary.loc['Events Analyzed',cond] += anal_events
                grand_summary.loc['Events Accepted',cond] += acc_events
            # If encounter FileNotFoundError, the cell was not analyzed
            except FileNotFoundError:
                pass
        ## Write results to Excel file
        # Create ExcelWriter object to write results
        writer = pd.ExcelWriter(f'{self.result_path}\\Summary.xlsx',engine = 'xlsxwriter')
        # Write cell by cell summary results to a sheet named "Cell By Cell Summary"
        cell_by_cell.to_excel(writer,sheet_name='Cell By Cell Summary')
        # Iterate through each condition
        for cond in self.conditions:
            sheet = pd.DataFrame.from_dict(temp[cond]) # Convert dictionary associated with condition to pandas DataFrame
            # Update grand summary
            grand_summary.loc['FWHM Mean',cond] = sheet['FWHM'].mean()
            grand_summary.loc['FWHM Std Dev',cond] = sheet['FWHM'].std()
            grand_summary.loc['Duration Mean',cond] = sheet['Duration'].mean()
            grand_summary.loc['Duration Std Dev',cond] = sheet['Duration'].std()
            # Save as a sheet in excel file
            sheet.to_excel(writer,sheet_name=cond)
        # Save grand summary results to a sheet named "Grand Summary"
        grand_summary.to_excel(writer,sheet_name='Grand Summary')
        # Save ExcelWriter object
        writer.save()
        # Close ExcelWriter object
        writer.close()
        
    def save(self):
        '''
        Method to save object instance to file
        '''
        f = open(f'{self.result_path}\\{self.name}',mode='w+b')
        pickle.dump(self, f, -1)
        f.close()
    
    def get_cell(self,cell_name):
        '''
        Method to retrieve a particular cell object from experiment
        
        Argument:
            cell_name: int, the number of the cell
            
        Returns:
            cell object
        '''
        try:
            cell_name = int(cell_name)
            res = self.Cells.loc[cell_name]
            try:
                if np.isnan(res):
                    raise AttributeError(f'Cell #{cell_name} has not been processed yet!')
            except TypeError:
                return res
        except ValueError:
            raise TypeError('Must provide integer!')
    
    def get_ROI(self,cell_name,roi_name):
        '''
        Method to retrieve a particular trace object from given cell
        
        Arguments:
            cell_name: int, the number of the cell
            roi_name: str or int, the name/number of the ROI
        '''
        CELL = self.get_cell(cell_name)
        try:
            roi_name = int(roi_name)
            # If roi_name has three digits
            if roi_name >= 100:
                idx = f'ROI{roi_name}'
            # Otherwise if roi_name has two digits
            elif 10 <= roi_name < 100:
                idx = f'ROI0{roi_name}'
            # Otherwise if roi_name has only one digit
            elif 0 < roi_name < 10:
                idx = f'ROI00{roi_name}'
            # If not any of these situations, raise ValueError
            else:
                raise IndexError('roi_name MUST be integer between 1 and 999')
        except ValueError:
            idx = roi_name.upper()
        res = CELL.ROIs.loc[idx]
        try:
            if np.isnan(res):
                raise AttributeError(f'{idx} in Cell #{cell_name} has not been processed yet!')
        except TypeError:
            return res
    
    @classmethod    
    def load(cls,path,name):
        '''
        Method to load object instance from file
        '''
        f = open(f'{path}\\Analysis\\{name}',mode = 'r+b')
        return pickle.load(f)
    
###############################################################################
######################## Functions for salvaging data #########################
###############################################################################
        
def minimax(roi,interp,show_plot=True):
    '''
    Find the minimum and maximum points of spline-fit curve of given roi
    
    Arguments:
        roi: trace object, contains data info
        interp: bool.
                True = find minimax values for interpolation spline
                False = find minimax values for least-squares regression B-spline
        show_plot: bool, if "True", displays plot of minimax points
        
    Returns:
        (maxs, mays):
            maxs: np.array object, contains x values of local maxima and minima
            mays: np.array object, conatins y values of local maxima and minima
    '''
    if interp:
        DS = roi.spline_interp.derivative() # Get the spline object of (first order) derivative of spline interpolation curve
        maxs = DS.roots(extrapolate = False) # Find the roots of derivative
        mays = roi.spline_interp(maxs) # Evaluate orignal spline at roots of derivative
    else:
        DS = roi.spline_func.derivative() # Get spline of (first order) derivative of spline fit curve
        maxs = solver_spline(DS,0) # Find roots of derivative spline
        mays = roi.spline_func(maxs) # Evaluate original spline at roots of derivative
    # If show_plot is "True"
    if show_plot:
        plt.plot(roi.x_f,roi.y_f,'b.',label='Raw Data')
        if interp:
            plt.plot(roi.x_val,roi.spline_interp(roi.x_val),'y-',label='Interpolation')
            plt.plot(maxs,mays,'kx',label='Minimax')
        else:
            plt.plot(roi.x_f,roi.spline_curve,'r-',label='Best Fit')
            plt.plot(maxs,mays,'kx',label='Minimax')
        plt.legend(loc='best')
        plt.show()
    return (maxs,mays)

def adjust_peak(roi,T_max,Peak,interp,show_plot = True,update_roi = False):
    '''
    Function to re-evaluate FWHM and duration after adjusting peak value
    
    Arguments:
        roi: trace object, contains original data and fitted information
        T_max: float, x value corresponding to Peak
        Peak: float, the peak value of curve
        interp: bool, whether to adjust peak for interpolation spline or least-
                squares regression B-Spline
                True: adjust peak for interpolation spline
                False: adjust peak for least-squares regression B-Spline
        show_plot: bool, if "True", plots the re-adjusted FWHM and duration
                   information. Default value is "True"
        update_roi: bool, if "True", updates the parameters of trace object
                    with adjusted values. Default value is "False"
    
    Returns: None
    '''
    if interp:
        # Estimate baseline
        bg1 = roi.pre.mean()
        bg2 = roi.post.mean()
        # Estimate half decay
        half_max_r_emp = (bg2 + Peak)/2
        half_max_l_emp = (bg1 + Peak)/2
        right_emp = roi.spline_interp.solve(half_max_r_emp) # Solve for the timepoint of half-decay (empirical)
        r_50_emp = right_emp[right_emp>T_max][0] # Empirical half-decay timepoint is defined as the first timepoint greater than T_max where the smoothed curve reaches half decay
        left_emp = roi.spline_interp.solve(half_max_l_emp) # Solve for the timepoint of half-rise (empirical)
        l_50_emp = left_emp[left_emp<T_max][-1] # Empirical half-rise timepoint is defined as the last timepoint smaller than T_max where the smoothed curve reaches half rise
        RHM_emp = r_50_emp - T_max
        LHM_emp = T_max - l_50_emp
        FWHM_emp = r_50_emp - l_50_emp
        delt_Tmax = T_max - roi.Tmax_fit
        peak_err = delt_Tmax / roi.FWHM
    else:
        # Estimate baseline
        bg1 = roi.spline_func.integrate(roi.x.loc[roi.startidx],roi.x.loc[roi.pre_endidx]) / (roi.x.loc[roi.pre_endidx] - roi.x.loc[roi.startidx])
        bg2 = bg2 = roi.spline_func.integrate(roi.x.loc[roi.post_startidx],roi.x.loc[roi.endidx]) / (roi.x.loc[roi.endidx] - roi.x.loc[roi.post_startidx])
        # Estimate FWHM
        half_max_l = (bg1 + Peak)/2
        half_max_r = (bg2 + Peak)/2
        left = solver_spline(roi.spline_func,half_max_l)
        right = solver_spline(roi.spline_func,half_max_r)
        l_50 = left[left<T_max][-1]
        r_50 = right[right>T_max][0]
        FWHM = r_50 - l_50
        # Estimate duration
        rise = (9*bg1 + Peak)/10
        decay = (9*bg2 + Peak)/10
        left1 = solver_spline(roi.spline_func,rise)
        right1 = solver_spline(roi.spline_func,decay)
        t_10 = left1[left1<T_max][-1]
        t_90 = right1[right1>T_max][0]
        duration = t_90 - t_10
        RHM_fit = r_50 - T_max
        Span_fit = Peak - bg2
        LHM_fit = T_max - l_50
        delt_Tmax = roi.Tmax_emp - T_max
        peak_err = delt_Tmax / FWHM
    # If update_roi is set to "True", do not show plot, update roi attributes
    if update_roi:
        show_plot = False
        if interp:
            roi.Peak_emp = Peak
            roi.Span_emp = Peak - bg2
            roi.Tmax_emp = T_max
            roi.half_max_r_emp = half_max_r_emp
            roi.r_50_emp = r_50_emp
            roi.RHM_emp = RHM_emp
            roi.half_max_l_emp = half_max_l_emp
            roi.l_50_emp = l_50_emp
            roi.LHM_emp = LHM_emp
            roi.FWHM_emp = FWHM_emp
        else:
            roi.half_max_l = half_max_l
            roi.half_max_r = half_max_r
            roi.l_50 = l_50
            roi.r_50 = r_50
            roi.rise = rise
            roi.decay = decay
            roi.t_10 = t_10
            roi.t_90 = t_90
            roi.FWHM = FWHM
            roi.duration = duration
            roi.Tmax_fit = T_max
            roi.Peak_fit = Peak
            roi.RHM_fit = RHM_fit
            roi.Span_fit = Span_fit
            roi.LHM_fit = LHM_fit
        roi.delt_Tmax = delt_Tmax
        roi.peak_err = peak_err
    # If show_plot is set to "True", plot the information
    if show_plot:
        plt.plot(roi.x_f,roi.y_f,'b.',label = 'Raw Data')
        top = roi.y_val.max()
        bottom = roi.y_val.min()
        span = top - bottom
        fwhm_pos = top + span / 10
        duration_pos = bottom - span / 10
        if interp:
            plt.plot(roi.x_val,roi.y_val_sm,'g.', label = 'Smoothed')
            plt.plot(roi.x_val,roi.spline_interp(roi.x_val),'y-',label = 'Interpolation')
            plt.plot([T_max,T_max],[duration_pos,Peak],'y--',label='_nolegend_')
            emp_pos = min(half_max_l_emp,half_max_r_emp)
            plt.plot([l_50_emp,r_50_emp],[half_max_l_emp,half_max_r_emp],'yo',label='_nolegend_')
            if half_max_l_emp > half_max_r_emp: # If half rise is greater than half decay
                plt.plot(l_50_emp,emp_pos,'yo',label='_nolegend_')
                plt.plot([l_50_emp,l_50_emp],[emp_pos,half_max_l_emp],'k:',label='_nolegend_')
            elif half_max_r_emp > half_max_l_emp: # If half decay is greater than half rise
                plt.plot(r_50_emp,emp_pos,'yo',label='_nolegend_')
                plt.plot([r_50_emp,r_50_emp],[emp_pos,half_max_r_emp],'k:',label='-nolegend_')
            plt.plot([l_50_emp,r_50_emp],[emp_pos,emp_pos],'y--',label='FWHM_emp')
        else:
            plt.plot(roi.x_f,roi.spline_curve,'r-',label = 'Best Fit')
            plt.plot([l_50,l_50],[half_max_l,fwhm_pos],'k:',label='_nolegend_')
            plt.plot([r_50,r_50],[half_max_r,fwhm_pos],'k:',label='_nolegend_')
            plt.plot([l_50,r_50],[half_max_l,half_max_r],'co',label='_nolegend_')
            plt.plot([l_50,r_50],[fwhm_pos,fwhm_pos],'c--',label='FWHM')
            plt.plot([t_10,t_10],[rise,duration_pos],'k:',label='_nolegend_')
            plt.plot([t_90,t_90],[decay,duration_pos],'k:',label='_nolegend_')
            plt.plot([t_10,t_90],[rise,decay],'mo',label='_nolegend_')
            plt.plot([t_10,t_90],[duration_pos,duration_pos],'m--',label='Duration')
        plt.plot(T_max,Peak,'kx',label='_nolegend_')
        plt.legend(loc='best')
        plt.show()
    return None

def copy(Cell,roi):
    '''
    Function to copy given roi of given Cell to clipboard
    
    Arguments:
        Cell: cell object which contains the roi
        roi: trace object whose characteristics are to be salvaged
        
    Returns:
        None.
    '''
    r = Tk()
    r.withdraw()
    r.clipboard_clear()
    r.clipboard_append(f'{Cell.cellnum}\t{roi.name}\t{roi.ves_type}\t{roi.half_max_l}\t{roi.half_max_r}\t{roi.rise}\t{roi.decay}\t{roi.l_50}\t{roi.r_50}\t{roi.t_10}\t{roi.t_90}\t{roi.FWHM}\t{roi.duration}\t{roi.FWHM_emp}\t{roi.RHM_emp}\t{roi.Span_emp}\t{roi.LHM_emp}\t{roi.l_50_emp}\t{roi.r_50_emp}\t{roi.half_max_r_emp}\t{roi.RHM_fit}\t{roi.Span_fit}\t{roi.LHM_fit}\t{roi.delt_Tmax}\t{roi.peak_err}')
    r.update()
    r.destroy()
    print('Cell#\tROI name\tVes type\tHalf-rise\tHalf-decay\t10% rise\t90% decay\tl_50\tr_50\tt_10\tt_90\tFWHM\tduration\tFWHM_emp\tRHM_emp\tSpan_emp\tLHM_emp\tl_50_emp\tr_50_emp\tEmpirical half-decay\tRHM_fit\tSpan_fit\tLHM_fit\tdelt_Tmax\tpeak_err')

def baseline_change (trace):
    '''
    Helper function to calculate the baseline difference (in arbitrary fluorescence units)
    
    Argument:
        trace: trace object
    
    Returns:
        diff: float, the difference of the average fluorescence baseline
    '''
    try:
        s1 = trace.startidx
        e1 = trace.pre_endidx
        s2 = trace.post_startidx
        e2 = trace.endidx
    except:
        return np.nan
    pre_bg = trace.y.loc[s1:e1].mean()
    post_bg = trace.y.loc[s2:e2].mean()
    return pre_bg - post_bg

###############################################################################
############################### Execution Code ################################
###############################################################################

if __name__ == '__main__':
    __spec__ = None
    # Get data file path
    path = 'F:\\D drive backup\\Edwards Lab Project\\Data\\RAW\\EXP183_alt'
#    path = './'
    
    # results = EXP(path)
    results = EXP.load(path,'EXP183_alt')

    # results.pre_process()
    # results.post_process()
    # results.summarize()
    # results.save()
