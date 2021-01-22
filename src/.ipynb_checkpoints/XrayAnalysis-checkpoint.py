import numpy as np
import pandas as pd
from pandas import DataFrame as df
import yaml
import matplotlib.pyplot as plt
import ipywidgets as widgets
from lmfit.models import GaussianModel, LinearModel, VoigtModel, PolynomialModel
import re
import os
from os import listdir
from os.path import isfile, join, dirname
import sys

# sys.path.append(os.getcwd() + '/../src/')
import AnalysisTools

##### Plotly settings #####

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook+plotly_mimetype'
pio.templates.default = 'simple_white'
pio.templates[pio.templates.default].layout.update(dict(
    title_y = 0.95,
    title_x = 0.5,
    title_xanchor = 'center',
    title_yanchor = 'top',
    legend_x = 0,
    legend_y = 1,
    legend_traceorder = "normal",
    legend_bgcolor='rgba(0,0,0,0)',
    margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=50, #top margin
        )
))

class XrayTools :
    
    def __init__(self,ParameterFile,DataFile,xRange=[-float('inf'),float('inf')]) :
        
        with open(ParameterFile[0]+'/'+ParameterFile[1]+'.yaml', 'r') as stream:
            par = yaml.safe_load(stream)
        
        dt = AnalysisTools.DataTools()
        Data, ErrorBars = dt.LoadData(DataFile,par)
        
        ##### Scale Data #####
        
        if 'Scaling' in par :
            
            if par['Scaling']['Type'] == 'Data' :
                Data = Data * par['Scaling']['Factor']
                ErrorBars = ErrorBars * par['Scaling']['Factor']
                print('Scaling data by ' + str(par['Scaling']['Factor']))
        
        ##### Background Subtraction #####
        
        if 'Background' in par :
            
            BackgroundFile = par['Background']
            
            string = DataFile.replace('.h5','')
            string = string.split('_')[1:]
            for segment in string :
                if segment[0] == 'E' :
                    BackgroundFile += '_'+segment+'.h5'
            
            if os.path.isfile(par['FolderPath']+'/'+BackgroundFile) :
                Background, Background_ErrorBars = dt.LoadData(BackgroundFile,par)
                Background = dt.TrimData(Background,xRange)
                Background_ErrorBars = dt.TrimData(Background_ErrorBars,xRange)
                if 'Scaling' in par and par['Scaling']['Type'] == 'Background' :
                    Background = Background * par['Scaling']['Factor']
                    Background_ErrorBars = Background_ErrorBars * par['Scaling']['Factor']
                    print('Scaling background by ' + str(par['Scaling']['Factor']))
                Data, ErrorBars = dt.SubtractBackground(Data,ErrorBars,Background,Background_ErrorBars,par)
            else:
                print('Background file not found. Background subtraction canceled.')
            
        self.ParameterFile = ParameterFile
        self.DataFile = DataFile
        self.par = par
        self.Data = Data
        self.ErrorBars = ErrorBars
    
    def FitData(self,Region) :
        
        Data = self.Data
        ErrorBars = self.ErrorBars
        par = self.par
        
        dt = AnalysisTools.DataTools()
        
        print(par['Description'])
        
        Data = dt.TrimData(Data, par['Spectra'][Region]['xRange'])
        ErrorBars = dt.TrimData(ErrorBars,par['Spectra'][Region]['xRange'])
        
        ##### Peak Assignments #####
        
        PeakList = list()
        AssignmentList = list()
        for Peak in par['Spectra'][Region]['Models'] :
            PeakList.append(Peak)
            if 'assignment' in par['Spectra'][Region]['Models'][Peak] :
                AssignmentList.append(par['Spectra'][Region]['Models'][Peak]['assignment'])
            else :
                AssignmentList.append(Peak)
        FitsAssignments = df(AssignmentList,index=PeakList,columns=['Assignment'])
        
        ##### Fit Data #####
        
        print('\nFitting Data...')
        
        fit = AnalysisTools.FitTools(Data,ErrorBars,par['Spectra'][Region])
        fit.Fit(NumberPoints=501)
        fit.ShowFits()
        
        Fits = fit.Fits
        FitsParameters = fit.FitsParameters
        
        ##### Convert FitsComponents to DataFrame #####
        
        FitsComponents = pd.DataFrame(fit.FitsComponents)
        FitsComponents.index = Data.columns
        for key in FitsComponents :
            FitsComponents = FitsComponents.rename(columns={key:str.replace(key,'_','')})
        
        print('\nDone fitting data')
        
        ##### Plot 2D Data & Fits #####
        
        plt.figure(figsize = [16,6])
        
        plt.subplot(1, 2, 1)
        x = Data.index.values
        y = Data.columns.values
        z = np.transpose(Data.values)
        plt.ylabel('Delay (fs)', fontsize=16)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
        plt.title('Data', fontsize=16)
        pcm = plt.pcolor(x, y, z, cmap='jet', shading='auto')
        
        plt.subplot(1, 2, 2)
        x = Fits.index.values
        y = Fits.columns.values
        z = np.transpose(Fits.values)
        plt.xlabel('Wavenumber (cm$^-$$^1$)', fontsize=16)
        plt.ylabel('Delay (fs)', fontsize=16)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
        plt.title('Fits', fontsize=16)
        pcm = plt.pcolor(x, y, z, cmap='jet', shading='auto')
        
        plt.show()
        
        ##### Plot Trends #####
        
        FitsParameters = FitsParameters.T
        UniqueParameters = ('amplitude','center','sigma')
        for uniqueParameter in UniqueParameters :
            fig = go.Figure()
            for parameter in FitsParameters :
                if uniqueParameter in parameter :
                    Name = parameter.split('_')[0]
                    if 'assignment' in par['Spectra'][Region]['Models'][Name] :
                        Name = par['Spectra'][Region]['Models'][Name]['assignment']
                    fig.add_trace(go.Scatter(x=FitsParameters.index,y=FitsParameters[parameter],name=Name,mode='lines+markers'))
            fig.update_layout(xaxis_title='Delay (fs)',yaxis_title='Fit Value',title=uniqueParameter,legend_title='',width=800,height=400)
            fig.show()
        
        self.Data = Data
        self.ErrorBars = ErrorBars
        self.Fits = Fits
        self.FitsParameters = FitsParameters
        self.FitsComponents = FitsComponents
        self.FitsAssignments = FitsAssignments