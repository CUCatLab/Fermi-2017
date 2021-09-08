import os
from os import listdir
from os.path import isfile, join
import sys
import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import cmath
import igor.igorpy as igor
import re
import yaml
import struct
from lmfit import model, Model
from lmfit.models import GaussianModel, SkewedGaussianModel, VoigtModel, ConstantModel, LinearModel, QuadraticModel, PolynomialModel

##### Data Tools #####

class DataTools :
    
    def __init__(self) :
        
        pass
    
    def FileList(self,FolderPath,Filter) :
        
        FileList = [f for f in listdir(FolderPath) if isfile(join(FolderPath, f))]
        for i in range(len(Filter)):
            FileList = [k for k in FileList if Filter[i] in k]
        for i in range(len(FileList)):
            FileList[i] = FileList[i].replace('.yaml','')
        
        return FileList
    
    def LoadBinnedData(self,DataFile,par) :
        
        # Check file
        f = h5py.File(par['FolderPath'] + '/' + DataFile, 'r')
        if not 'BinnedData/xas_bins' in f :
            raise Exception('Energy data missing')
        if not 'BinnedData/delay_bins' in f :
            raise Exception('Delay data missing')
        if not 'BinnedData/XAS_2dmatrix' in f :
            raise Exception('Intensity data missing')
        if not 'BinnedData/XAS_2dmatrix_err' in f :
            raise Exception('Error bar data missing')

        # Load data
        Energy = f['BinnedData/xas_bins'][...]
        Delay = f['BinnedData/delay_bins'][...]
        Signal = f['/BinnedData/XAS_2dmatrix'][...]
        ErrorBars = f['/BinnedData/XAS_2dmatrix_err'][...]

        f.close()
        
        # Energy offset
        Energy = Energy + par['xOffset']
        
        # Create data frame
        Data =  df(data=np.transpose(Signal),columns=Delay)
        Data.index = Energy
        ErrorBars =  df(data=np.transpose(ErrorBars),columns=Delay,index=Energy)
        
        # Remove empty data sets
        Data = Data.dropna(axis=1, how='all')
        ErrorBars = ErrorBars.dropna(axis=1, how='all')
        
        return Data, ErrorBars
    
    def LoadHDF(self,Folder,File) :
        
        Store = pd.HDFStore(Folder + '/' + File)
        Data = {'File': File}
        for key in Store.keys() :
            Data[str.replace(key,'/','')] = Store.get(key)
        Store.close()

        return Data
    
    def TrimData(self,Data,xRange) :
        
        # Trim data
        Mask = np.all([Data.index.values>min(xRange),Data.index.values<max(xRange)],axis=0)
        Data = Data[Mask]
        
        return Data

    def SubtractBackground(self,Data,ErrorBars,Background,Background_ErrorBars,par,ShowPlot=True) :
        
        Min = max(min(Data.index),min(Background.index))
        Max = min(max(Data.index),max(Background.index))

        Data = self.TrimData(Data,[Min,Max])
        ErrorBars = self.TrimData(ErrorBars,[Min,Max])
        Background = self.TrimData(Background,[Min,Max])
        Background_ErrorBars = self.TrimData(Background_ErrorBars,[Min,Max])
        
        if np.array_equal(Data.index,Background.index) :
            Data = Data.subtract(Background.values)
            print('Background Successfully subtracted from data')
            ErrorBars = ErrorBars**2
            ErrorBars = ErrorBars.add((Background_ErrorBars**2).values)
            ErrorBars = ErrorBars**0.5
            
        else :
            print('Warning: Energy axes do not match')
            print('Warning: Background subtraction cancelled')
        
        if ShowPlot :
            
            plt.errorbar(Background.index, Background.values, yerr=Background_ErrorBars.iloc[:,0], fmt='o', color='black')
            plt.xlabel('Energy (eV)'), plt.ylabel('Intensity (au)')
            plt.title('Background: '+par['Background'])
            plt.show()

        return Data, ErrorBars
        

##### Fit Tools #####

class FitTools :
    
    def __init__(self,Data,ErrorBars,par,Name='') :
        
        self.Data = Data
        self.ErrorBars = ErrorBars
        self.par = par
        self.Name = Name
        
    def SetModel(self) :
        
        par = self.par
        Data = self.Data
        
        ModelString = list()
        for Peak in par['Models'] :
            ModelString.append((Peak,par['Models'][Peak]['model']))
        
        for Model in ModelString :
            try :
                FitModel
            except :
                if Model[1] == 'Constant' :
                    FitModel = ConstantModel(prefix=Model[0]+'_')
                if Model[1] == 'Linear' :
                    FitModel = LinearModel(prefix=Model[0]+'_')
                if Model[1] == 'Gaussian' :
                    FitModel = GaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'SkewedGaussian' :
                    FitModel = SkewedGaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'Voigt' :
                    FitModel = VoigtModel(prefix=Model[0]+'_')
            else :
                if Model[1] == 'Constant' :
                    FitModel = FitModel + ConstantModel(prefix=Model[0]+'_')
                if Model[1] == 'Linear' :
                    FitModel = FitModel + LinearModel(prefix=Model[0]+'_')
                if Model[1] == 'Gaussian' :
                    FitModel = FitModel + GaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'SkewedGaussian' :
                    FitModel = FitModel + SkewedGaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'Voigt' :
                    FitModel = FitModel + VoigtModel(prefix=Model[0]+'_')
        
        ModelParameters = FitModel.make_params()
        FitsParameters = df(index=ModelParameters.keys(),columns=Data.columns.values)
        
        self.FitModel = FitModel
        self.ModelParameters = ModelParameters
        self.FitsParameters = FitsParameters
    
    def SetParameters(self, Value=None) :
        
        par = self.par
        ModelParameters = self.ModelParameters
        FitsParameters = self.FitsParameters
        
        ParameterList = ['intercept','offset','amplitude','center','sigma']
        Parameters = {'Standard': par['Models']}

        if 'Cases' in par and Value != None:
            for Case in par['Cases'] :
                if Value >= min(par['Cases'][Case]['zRange']) and Value <= max(par['Cases'][Case]['zRange']) :
                    Parameters[Case] = par['Cases'][Case]

        for Dictionary in Parameters :
            for Peak in Parameters[Dictionary] :
                for Parameter in Parameters[Dictionary][Peak] :
                    if Parameter in ParameterList :
                        for Key in Parameters[Dictionary][Peak][Parameter] :
                            if Key != 'set' :
                                exec('ModelParameters["'+Peak+'_'+Parameter+'"].'+Key+'='+str(Parameters[Dictionary][Peak][Parameter][Key]))
                            else :
                                exec('ModelParameters["'+Peak+'_'+Parameter+'"].'+Key+str(Parameters[Dictionary][Peak][Parameter][Key]))
                                    
        self.ModelParameters = ModelParameters
    
    def Fit(self,**kwargs) :
        
        for kwarg in kwargs :
            if kwarg == 'fit_x':
                fit_x = kwargs[kwarg]
            if kwarg == 'NumberPoints':
                NumberPoints = kwargs[kwarg]
        
        dt = DataTools()
        self.SetModel()
        
        Data = self.Data
        Name = self.Name
        par = self.par
        ModelParameters = self.ModelParameters
        FitModel = self.FitModel
        FitsParameters = self.FitsParameters
        
        if 'xRange' in par :
            Data = dt.TrimData(Data,[par['xRange'][0],par['xRange'][1]])
        x = Data.index.values
        try:
            fit_x
        except :
            try :
                NumberPoints
            except :
                fit_x = x
            else :
                fit_x = np.zeros((NumberPoints))
                for i in range(NumberPoints) :
                    fit_x[i] = min(x) + i * (max(x) - min(x)) / (NumberPoints - 1)
        
        Fits = df(index=fit_x,columns=Data.columns.values)
        
        FitsResults = list()
        FitsComponents = list()
        
        for idx,Column in enumerate(Data) :
            
            self.SetParameters(Value=Column)
            
            y = Data[Column].values
            FitResults = FitModel.fit(y, ModelParameters, x=x, nan_policy='omit')
            fit_comps = FitResults.eval_components(FitResults.params, x=fit_x)
            fit_y = FitResults.eval(x=fit_x)
            ParameterNames = [i for i in FitResults.params.keys()]
            for Parameter in (ParameterNames) :
                FitsParameters[Column][Parameter] = FitResults.params[Parameter].value
            Fits[Column] = fit_y
            FitsResults.append(FitResults)
            FitsComponents.append(fit_comps)
            
            sys.stdout.write(("\rFitting %i out of "+str(Data.shape[1])) % (idx+1))
            sys.stdout.flush()
        
        
        
        self.Fits = Fits
        self.FitsParameters = FitsParameters
        self.FitsResults = FitsResults
        self.FitsComponents = FitsComponents
    
    def ShowFits(self,xLabel='Energy (eV)',yLabel='Intensity (au)') :
        
        Data = self.Data
        ErrorBars = self.ErrorBars
        Fits = self.Fits
        par = self.par
        
        FitsParameters = self.FitsParameters
        FitsComponents = self.FitsComponents
        
        for idx,Column in enumerate(Data) :
            
            plt.figure(figsize = [6,4])
            plt.errorbar(Data.index, Data[Column], yerr=ErrorBars[Column],label='Data', fmt='o', color='black')
            plt.plot(Fits.index, Fits[Column], 'r-', label='Fit')
            for Component in FitsComponents[idx] :
                if not isinstance(FitsComponents[idx][Component],float) :
                    plt.fill(Fits.index, FitsComponents[idx][Component], '--', label=Component[:-1], alpha=0.5)
            plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
            plt.xlabel(xLabel), plt.ylabel(yLabel)
            plt.title(str(Column))
            plt.show()
            
            Peaks = list()
            for Parameter in FitsParameters.index :
                Name = Parameter.split('_')[0]
                if Name not in Peaks :
                    Peaks.append(Name)

            string = ''
            for Peak in Peaks :
                string = string + Peak + ' | '
                for Parameter in FitsParameters.index :
                    if Peak == Parameter.split('_')[0] : 
                        string = string + Parameter.split('_')[1] + ': ' + str(round(FitsParameters[Column][Parameter],3))
                        string = string + ', '
                string = string[:-2] + '\n'
            print(string)
            print(75*'_')