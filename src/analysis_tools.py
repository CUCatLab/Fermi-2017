import h5py
import numpy as np
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import ipywidgets as widgets
from lmfit.models import GaussianModel, LinearModel, VoigtModel, PolynomialModel
import re
from os import listdir
from os.path import isfile, join, dirname

def FileList(FolderPath, Filter) :
    FileList = [f for f in listdir(FolderPath) if isfile(join(FolderPath, f))]
    FileList = [k for k in FileList if Filter in k]
    return FileList

def ImportData(FolderPath, FileName) :
    
    # Check file
    if FileName == 'None' :
        raise HaltException('Please select file')
    f = h5py.File(FolderPath + '/' + FileName, 'r')
    if not 'BinnedData/xas_bins' in f :
        raise HaltException('Energy data missing')
    if not 'BinnedData/delay_bins' in f :
        raise HaltException('Delay data missing')
    if not 'BinnedData/XAS_2dmatrix' in f :
        raise HaltException('Intensity data missing')
    if not 'BinnedData/XAS_2dmatrix_err' in f :
        raise HaltException('Error bar data missing')
    
    # Load data
    Energy = f['BinnedData/xas_bins'][...]
    Delay = f['BinnedData/delay_bins'][...]
    Signal = f['/BinnedData/XAS_2dmatrix'][...]
    ErrorBars = f['/BinnedData/XAS_2dmatrix_err'][...]
    
    f.close()
    return Energy, Signal, Delay, ErrorBars

def Normalize(x, y, Min, Max) :
    Min_Index = (np.abs(x - Min)).argmin()
    Max_Index = (np.abs(x - Max)).argmin()
    if len(y.shape) == 1 :
        Norm = np.nanmean(y[Min_Index:Max_Index])
    if len(y.shape) == 2 :
        Norm = np.nanmean(np.transpose(y)[Min_Index:Max_Index])
    y = y / Norm - 1
    return x, y
    
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

class Fits :
    
    def __init__(self,ModelString) :
        
        self.NumModels = {}
        self.NumModels['Total'] = len(ModelString)
        self.NumModels['Linear'] = len(re.findall('L', ModelString))
        self.NumModels['Gaussian'] = len(re.findall('G', ModelString))
        self.NumModels['Voigt'] = len(re.findall('V', ModelString))
        if self.NumModels['Total'] != self.NumModels['Linear'] + self.NumModels['Gaussian'] + self.NumModels['Voigt'] :
            print('Warning: Number of total functions does not equal number of summed functions')
        ModelCounter= 0
        i = 0
        while i < self.NumModels['Linear'] :
            if ModelCounter == 0 :
                self.Model = LinearModel(prefix='L'+str(i+1)+'_')
            else :
                self.Model = self.Model + LinearModel(prefix='L'+str(i+1)+'_')
            ModelCounter = ModelCounter + 1
            i += 1
        i = 0
        while i < self.NumModels['Gaussian'] :
            if ModelCounter == 0 :
                self.Model = GaussianModel(prefix='G'+str(i+1)+'_')
            else :
                self.Model = self.Model + GaussianModel(prefix='G'+str(i+1)+'_')
            ModelCounter = ModelCounter + 1
            i += 1
        i = 0
        while i < self.NumModels['Voigt'] :
            if ModelCounter == 0 :
                self.Model = VoigtModel(prefix='V'+str(i+1)+'_')
            else :
                self.Model = self.Model + VoigtModel(prefix='V'+str(i+1)+'_')
            ModelCounter = ModelCounter + 1
            i += 1
    
    def Fit(self,x,y,err,Delay,Params) :
        
        self.x = x
        self.y = y
        self.err = err
        self.Delay = Delay
        Fit = self.Model.fit(y, Params, x=x, fit_kws={'maxfev': 10000})
        
        fit_x_delta = 0.01
        fit_len = int((max(x)-min(x))/fit_x_delta + 1)
        self.fit_x = np.zeros((fit_len))
        i = 0
        while i < fit_len :
            self.fit_x[i] = min(x) + i * fit_x_delta
            i += 1
        self.fit_comps = Fit.eval_components(Fit.params, x=self.fit_x)
        self.fit_y = Fit.eval(x=self.fit_x)
        
        # Save parameters
        Parameters = np.zeros((1+2*self.NumModels['Linear']+3*self.NumModels['Gaussian']+3*self.NumModels['Voigt']))
        ParameterNames = list(('Delay',))
        Parameters[0] = Delay
        self.ParameterString = ''
        Counter = 1
        i = 0
        while i < self.NumModels['Linear'] :
            ParameterNames.extend(('L'+str(i+1)+'_intercept','L'+str(i+1)+'_slope'))
            self.ParameterString = self.ParameterString + '<p>Linear '+str(i+1)+' |&nbsp; Intercept: '+ str(round(Fit.params['L'+str(i+1)+'_intercept'].value,4)) + ',&nbsp; Slope: ' + str(round(Fit.params['L'+str(i+1)+'_slope'].value,4))
            Parameters[0+Counter] = Fit.params['L'+str(i+1)+'_intercept'].value
            Parameters[1+Counter] = Fit.params['L'+str(i+1)+'_slope'].value
            Counter = Counter + 2
            i += 1
        i = 0
        while i < self.NumModels['Gaussian'] :
            ParameterNames.extend(('G'+str(i+1)+'_amp','G'+str(i+1)+'_ω','G'+str(i+1)+'_σ'))
            self.ParameterString = self.ParameterString + '<p>Gaussian '+str(i+1)+' |&nbsp; Amplitude: ' + str(round(Fit.params['G'+str(i+1)+'_amplitude'].value,4)) + ',&nbsp; Energy: ' + str(round(Fit.params['G'+str(i+1)+'_center'].value,2)) + ' eV,&nbsp; Width: ' + str(round(Fit.params['G'+str(i+1)+'_sigma'].value,3))
            Parameters[0+Counter] = Fit.params['G'+str(i+1)+'_amplitude'].value
            Parameters[1+Counter] = Fit.params['G'+str(i+1)+'_center'].value
            Parameters[2+Counter] = Fit.params['G'+str(i+1)+'_sigma'].value
            Counter = Counter + 3
            i += 1
        i = 0
        while i < self.NumModels['Voigt'] :
            ParameterNames.extend(('V'+str(i+1)+'_amp','V'+str(i+1)+'_ω','V'+str(i+1)+'_σ'))
            self.ParameterString = self.ParameterString + '<p>Voigt '+str(i+1)+' |&nbsp; Amplitude: ' + str(round(Fit.params['V'+str(i+1)+'_amplitude'].value,4)) + ',&nbsp; Energy: ' + str(round(Fit.params['V'+str(i+1)+'_center'].value,2)) + ' eV,&nbsp; Width: ' + str(round(Fit.params['V'+str(i+1)+'_sigma'].value,3))
            Parameters[0+Counter] = Fit.params['V'+str(i+1)+'_amplitude'].value
            Parameters[1+Counter] = Fit.params['V'+str(i+1)+'_center'].value
            Parameters[2+Counter] = Fit.params['V'+str(i+1)+'_sigma'].value
            Counter = Counter + 3
            i += 1
        
        self.Parameters = Parameters
        self.ParameterNames = ParameterNames
        self.FitReport = Fit.fit_report()
    
    def Plot(self) :

        # Plot fits
        plt.figure(figsize = [6,4])
        plt.plot(self.x, self.y,'r.', label='data')
        plt.plot(self.fit_x, self.fit_y, 'k-', label='fit')
        i = 0
        while i < self.NumModels['Linear'] :
            plt.plot(self.fit_x, self.fit_comps['L'+str(i+1)+'_'], 'k--', label='Linear '+str(i+1))
            i += 1
        i = 0
        while i < self.NumModels['Gaussian'] :
            plt.fill(self.fit_x, self.fit_comps['G'+str(i+1)+'_'], '--', label='Gaussian '+str(i+1), alpha=0.5)
            i+=1
        i = 0
        while i < self.NumModels['Voigt'] :
            plt.fill(self.fit_x, self.fit_comps['V'+str(i+1)+'_'], '--', label='Voigt '+str(i+1), alpha=0.5)
            i+=1
        plt.errorbar(self.x, self.y, yerr=self.err, fmt='o')
#         plt.plot((287.3,287.3),(0,max(self.y)),'k:',label='CO Gas Phase')
        plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
        plt.xlabel('Photon Energy (eV)'), plt.ylabel('Intensity (au)')
        plt.title('Delay: ' + str(self.Delay) + ' fs')
        
        ShowPlot = widgets.Output()
        with ShowPlot :
            plt.show()

        ShowText = widgets.HTML(
            value=self.ParameterString,
            placeholder='',
            description='',
        )

        display(widgets.Box([ShowPlot,ShowText]))

def PlotAnalysis(Name, FitParameters, ParameterNames) :
    cols = [col for col in FitParameters.columns if Name in col]
    i = 0
    while i < len(cols) :
        plt.plot(FitParameters['Delay'], FitParameters[cols[i]],'.:', label=cols[i])
        i+=1
    plt.legend(), plt.xlabel('Delay (fs)'), plt.ylabel(Name)
    plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
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