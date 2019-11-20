import h5py
import numpy as np
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import ipywidgets as widgets
from lmfit.models import GaussianModel, LinearModel, VoigtModel, PolynomialModel

def ImportData(FolderPath, FileName) :
    
    # Check file
    if FileName == 'None' :
        raise HaltException('Please select file')
    f = h5py.File(FolderPath + '/' + FileName, 'r')
    if not 'BinnedData/E_bin_centers' in f :
        raise HaltException('Energy data missing')
    if not 'BinnedData/delays_fs' in f :
        raise HaltException('Delay data missing')
    if not 'BinnedData/XAS_2dmatrix' in f :
        raise HaltException('Intensity data missing')
    if not 'BinnedData/XAS_2dmatrix_err' in f :
        raise HaltException('Error bar data missing')
    
    # Load data
    Energy = f['BinnedData/E_bin_centers'][...]
    Delay = f['BinnedData/delays_fs'][...]
    Signal = f['/BinnedData/XAS_2dmatrix'][...]
    ErrorBars = f['/BinnedData/XAS_2dmatrix_err'][...]
    
    f.close()
    return Energy, Signal, Delay, ErrorBars

def TrimData(Energy, Signal, Delay, ErrorBars, Min, Max) :
    
    # Trim data
    Min_Index = (np.abs(Energy - Min)).argmin()
    Max_Index = (np.abs(Energy - Max)).argmin()
    Energy = Energy[Min_Index:Max_Index]
    Signal = np.transpose(Signal)
    Signal = Signal[Min_Index:Max_Index]
    Signal = np.transpose(Signal)
    ErrorBars = np.transpose(ErrorBars)
    ErrorBars = ErrorBars[Min_Index:Max_Index]
    ErrorBars = np.transpose(ErrorBars)
    
    # Remove empty data sets
    remove = np.array([],dtype=int)
    i = 0
    while i < len(Signal) :
        if np.count_nonzero(np.isnan(Signal[i])) == len(Signal[i]) :
            remove = np.append(remove, i)
        i += 1
    Signal = np.delete(Signal,remove,axis=0)
    ErrorBars = np.delete(ErrorBars,remove,axis=0)
    Delay = np.delete(Delay,remove)
    
    return Energy, Signal, Delay, ErrorBars
    
def SubtractBackground(Energy, Signal, Background_Energy, Background, Min, Max) :
    Min_Index = (np.abs(Background_Energy - Min)).argmin()
    Max_Index = (np.abs(Background_Energy - Max)).argmin()
    Background = np.transpose(Background)
    Background = Background[Min_Index:Max_Index]
    Background = np.transpose(Background)
    Background_Energy = Background_Energy[Min_Index:Max_Index]
    Background_Do = True
    if not np.array_equal(Background_Energy, Energy) :
        print('Warning: Energy axes do not match')
        print('Warning: Background subtraction cancelled')
        Background_Do = False
    if Background_Do :
        Signal = Signal - Background
        print('Background Successfully subtracted from data')
    
    return Signal

def FitModel(NumberPeaks) :
    Model = LinearModel(prefix='B_')
    i = 0
    while i < NumberPeaks :
        Model = Model + GaussianModel(prefix='G'+str(i+1)+'_')
        i+=1
    return Model

def Convert2Dataframe(FitParameters) :
    NumberPeaks = (FitParameters.shape[1]-3)/3
    Header = list()
    Header.append(('Delay','B_Intercept','B_Slope'))
    i = 0
    while i < NumberPeaks :
        Header.append(('amp'+str(i+1),'ω'+str(i+1),'σ'+str(i+1)))
        i+=1
    Header = [y for x in Header for y in x]
    FitParameters = df(data=FitParameters,columns=Header)
    
    return FitParameters

def Fits(x,y,err,Delay,Fit) :
    fit_x_delta = 0.01
    fit_len = int((max(x)-min(x))/fit_x_delta + 1)
    fit_x = np.zeros((fit_len))
    i = 0
    while i < fit_len :
        fit_x[i] = min(x) + i * fit_x_delta
        i += 1
    fit_comps = Fit.eval_components(Fit.params, x=fit_x)
    fit_y = Fit.eval(x=fit_x)
    NumberPeaks = len(fit_comps) - 1
    
    # Plot fits
    plt.figure(figsize = [6,4])
    plt.plot(x, y,'r.', label='data')
    plt.plot(fit_x, fit_y, 'k-', label='fit')
    plt.plot(fit_x, fit_comps['B_'], 'k--', label='Baseline')
    i = 0
    while i < NumberPeaks :
        plt.fill(fit_x, fit_comps['G'+str(i+1)+'_'], '--', label='Peak '+str(i+1), alpha=0.5)
        i+=1
    plt.errorbar(x, y, yerr=err, fmt='o')
    plt.plot((287.3,287.3),(0,max(y)),'k:',label='CO Gas Phase')
    plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
    plt.xlabel('Photon Energy (eV)'), plt.ylabel('Intensity (au)')
    plt.title('Delay: ' + str(Delay) + ' fs')
    
    # Save parameters
    FitParameters = np.zeros((int(3+3*NumberPeaks)))
    FitParameters[0] = Delay
    string = '<p>Baseline |&nbsp; Intercept: '+ str(round(Fit.params['B_intercept'].value,4)) + ',&nbsp; Slope: ' + str(round(Fit.params['B_slope'].value,4))
    FitParameters[1] = Fit.params['B_intercept'].value
    FitParameters[2] = Fit.params['B_slope'].value
    i = 0
    while i < NumberPeaks :
        string = string + '<p>Peak '+str(i+1)+' |&nbsp; Amplitude: ' + str(round(Fit.params['G'+str(i+1)+'_amplitude'].value,4)) + ',&nbsp; Energy: ' + str(round(Fit.params['G'+str(i+1)+'_center'].value,2)) + ' eV,&nbsp; Width: ' + str(round(Fit.params['G'+str(i+1)+'_sigma'].value,3))
        FitParameters[3+3*i] = Fit.params['G'+str(i+1)+'_amplitude'].value
        FitParameters[4+3*i] = Fit.params['G'+str(i+1)+'_center'].value
        FitParameters[5+3*i] = Fit.params['G'+str(i+1)+'_sigma'].value
        i+=1
    
    ShowPlot = widgets.Output()
    with ShowPlot :
        plt.show()
    
    ShowText = widgets.HTML(
        value=string,
        placeholder='',
        description='',
    )
    
    display(widgets.Box([ShowPlot,ShowText]))
    
    return FitParameters, fit_x, fit_y

def PlotAnalysis(Name, FitParameters) :
    NumberPeaks = (FitParameters.shape[1]-3)/3
    cols = [col for col in FitParameters.columns if Name in col]
    i = 0
    while i < len(cols) :
        plt.plot(FitParameters['Delay'], FitParameters[cols[i]],'.:', label='Peak '+str(i+1))
        i+=1
    plt.legend(), plt.xlabel('Delay (fs)'), plt.ylabel(Name)
    plt.title(Name+' vs Delay')
    plt.show()

def PlotDataAndFits(Energy, Signal, Fit_Energy, Fit_Signal, Delay) :
    fig = plt.figure(figsize=(12,4))

    ax = fig.add_subplot(1, 2, 1)
    x, y, z = Energy, Delay, Signal
    plt.xlabel('Energy (eV)')
    plt.ylabel('Delay (fs)')
    plt.tick_params(axis = 'both', which = 'major')
    plt.title('Data')
    pcm = ax.pcolor(x, y, z, cmap='jet')

    ax = fig.add_subplot(1, 2, 2)
    x, y, z = Fit_Energy, Delay, Fit_Signal
    plt.xlabel('Energy (eV)')
    plt.ylabel('Delay (fs)')
    plt.tick_params(axis = 'both', which = 'major')
    plt.title('Fits')
    pcm = ax.pcolor(x, y, z, cmap='jet')

    plt.show()