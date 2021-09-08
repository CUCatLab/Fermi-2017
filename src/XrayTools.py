import numpy as np
import scipy
import pandas as pd
from pandas import DataFrame as df
import yaml
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
import ipywidgets as widgets
from ipywidgets import Button, Layout
from lmfit import model, Model
import re
import os
from os import listdir
from os.path import isfile, join, dirname
import sys
from IPython.display import clear_output
import AnalysisTools
from pylab import rc

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

Folders = {}
Folders['Parameters'] = os.getcwd()+'/../Parameters'
Folders['Fits'] = os.getcwd()+'/../Fits'
Folders['Figures'] = os.getcwd()+'/../Figures'
Folders['Binned'] = os.getcwd()+'/../Binned/BT2'

class Fit :
    
    def __init__(self) :
        
        self.Folders = Folders
        
    def LoadData(self,ParametersFile,DataFile,xRange=[-float('inf'),float('inf')]) :

        self.DataFile = DataFile

        with open(ParametersFile[0]+'/'+ParametersFile[1]+'.yaml', 'r') as stream:
            par = yaml.safe_load(stream)

        dt = AnalysisTools.DataTools()
        Data, ErrorBars = dt.LoadBinnedData(DataFile,par)

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
                Background, Background_ErrorBars = dt.LoadBinnedData(BackgroundFile,par)
                Background = dt.TrimData(Background,xRange)
                Background_ErrorBars = dt.TrimData(Background_ErrorBars,xRange)
                if 'Scaling' in par and par['Scaling']['Type'] == 'Background' :
                    Background = Background * par['Scaling']['Factor']
                    Background_ErrorBars = Background_ErrorBars * par['Scaling']['Factor']
                    print('Scaling background by ' + str(par['Scaling']['Factor']))
                Data, ErrorBars = dt.SubtractBackground(Data,ErrorBars,Background,Background_ErrorBars,par)
            else:
                print('Background file not found. Background subtraction canceled.')

        self.ParametersFile = ParametersFile
        self.DataFile = DataFile
        self.par = par
        self.Data = Data
        self.ErrorBars = ErrorBars
    
    def Fit(self,Region) :
        
        Data = self.Data
        ErrorBars = self.ErrorBars
        par = self.par
        DataFile = self.DataFile
        
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
        plt.title(str.split(DataFile,".")[0]+' - Data', fontsize=16)
        pcm = plt.pcolor(x, y, z, cmap='jet', shading='auto')
        
        plt.subplot(1, 2, 2)
        x = Fits.index.values
        y = Fits.columns.values
        z = np.transpose(Fits.values)
        plt.xlabel('Wavenumber (cm$^-$$^1$)', fontsize=16)
        plt.ylabel('Delay (fs)', fontsize=16)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
        plt.title(str.split(DataFile,".")[0]+' - Fits', fontsize=16)
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
            fig.update_layout(xaxis_title='Delay (fs)',yaxis_title=uniqueParameter,title=str.split(DataFile,".")[0],legend_title='',width=800,height=400)
            fig.show()
        
        ##### Store Fits ####
        
        self.Fits = Fits
        self.FitsParameters = FitsParameters
        self.FitsComponents = FitsComponents
        self.FitsAssignments = FitsAssignments
        self.FitsData = Data
        self.FitsErrorBars = ErrorBars
        
        ##### GUI #####

        def CopyData_Clicked(b) :
            Data.to_clipboard()
        CopyData = widgets.Button(description="Copy Data")
        CopyData.on_click(CopyData_Clicked)

        def CopyFits_Clicked(b) :
            Fits.to_clipboard()
        CopyFits = widgets.Button(description="Copy Fits")
        CopyFits.on_click(CopyFits_Clicked)

        def CopyParameters_Clicked(b) :
            FitsParameters.to_clipboard()
        CopyParameters = widgets.Button(description="Copy Parameters")
        CopyParameters.on_click(CopyParameters_Clicked)

        def Save2File_Clicked(b) :
            os.makedirs(Folders['Fits'], exist_ok=True)
            FitsFile = Folders['Fits'] +'/' + self.ParametersFile[1] + ' - ' + str.replace(self.DataFile,'.h5','') + ' - ' + self.Region.value + '.hdf'
            Data.to_hdf(FitsFile,'Data',mode='w')
            ErrorBars.to_hdf(FitsFile,'ErrorBars',mode='a')
            Fits.to_hdf(FitsFile,'Fits',mode='a')
            FitsParameters.to_hdf(FitsFile,'Fits_Parameters',mode='a')
            FitsComponents.to_hdf(FitsFile,'Fits_Components',mode='a')
            FitsAssignments.to_hdf(FitsFile,'Fits_Assignments',mode='a')
        Save2File = widgets.Button(description="Save to File")
        Save2File.on_click(Save2File_Clicked)

        display(widgets.Box([CopyData,CopyFits,CopyParameters,Save2File]))
    
    def UI(self) :
        
        dt = AnalysisTools.DataTools()
        
        out = widgets.Output()
        
        ##### Button Functions #####

        def UpdateFiles_Clicked(b):
            with open(Folders['Parameters']+'/'+self.ParametersFiles.value+'.yaml', 'r') as stream:
                par = yaml.safe_load(stream)
            self.DataFiles.options = dt.FileList(par['FolderPath'],[par['Runs']])
        UpdateFiles = widgets.Button(description="Update",layout = Layout(width='10%'))
        UpdateFiles.on_click(UpdateFiles_Clicked)
        
        def FitData_Clicked(b):
            with out :
                clear_output(True)
                self.LoadData([Folders['Parameters'],self.ParametersFiles.value],self.DataFiles.value)
                self.Fit(self.Region.value)
        FitData = widgets.Button(description="Fit",layout = Layout(width='10%'))
        FitData.on_click(FitData_Clicked)

        ##### Widgets #####

        self.ParametersFiles = widgets.Dropdown(
            options=dt.FileList(Folders['Parameters'],['.yaml']),
            description='Parameter File',
            layout=Layout(width='70%'),
            style = {'description_width': '150px'},
            disabled=False,
        )

        self.Region = widgets.Dropdown(
            options = ['Pi Star','Middle','Shape Resonance'],
            description = 'Select Region',
            layout = Layout(width='40%'),
            style = {'description_width': '150px'},
            disabled = False,
        )

        with open(Folders['Parameters']+'/'+self.ParametersFiles.value+'.yaml', 'r') as stream:
            par = yaml.safe_load(stream)

        self.DataFiles = widgets.Dropdown(
            options=dt.FileList(par['FolderPath'],[par['Runs']]),
            description='Data File',
            layout=Layout(width='50%'),
            style = {'description_width': '150px'},
            disabled=False,
        )
        
        display(widgets.Box([self.ParametersFiles,UpdateFiles]))
        display(self.DataFiles)
        display(self.Region)
        display(FitData)
        
        display(out)

class Trends :
    
    def __init__(self) :
        
        self.Folders = Folders
    
    def LoadData(self,ParametersFile,DataFile) :
        
        dt = AnalysisTools.DataTools()
        
        # Load Data
        Data = dt.LoadHDF(DataFile[0],DataFile[1])
        Data = Data['Fits_Parameters']
        Data['Precursor 1'] = Data['Precursor10_amplitude'] + Data['Precursor11_amplitude']
        Data['Precursor 2'] = Data['Precursor20_amplitude'] + Data['Precursor21_amplitude']
        Data = Data.rename({'Precursor2_amplitude': 'Precursor 2', 'Adsorbed_amplitude': 'Adsorbed', 'Unknown_amplitude': 'Unknown'}, axis=1)
        Data = Data.reindex(columns=['Adsorbed','Precursor 1','Precursor 2','Unknown'])
        
        # Load Parameters
        with open(ParametersFile[0]+'/'+ParametersFile[1]+'.yaml', 'r') as stream:
            par = yaml.safe_load(stream)
        
        self.Data = Data
        self.DataFile = DataFile
        self.par = par
    
    def Fit(self) :
        
        Data = self.Data
        DataFile = self.DataFile
        
        linewidth=2
        fontsize = 20
    
        def func(t,a,b,t0,sigma):
            return a*scipy.special.erf((t-t0)/sigma)+a+b

        FitsResults = list()

        FitModel = Model(func)
        FitParameters = FitModel.make_params()

        FitParameters['b'].min = 0
        FitParameters['t0'].min = 0
        FitParameters['sigma'].min = 0

        TempData = Data[Data.index<=max(self.par['Trends']['xRange'])]
        TempData = TempData[TempData.index>=min(self.par['Trends']['xRange'])]
        Parameters = self.par['Trends']['Fits']
        Fits_TrendFits = df(index=np.linspace(min(TempData.index),max(TempData.index),101))
        Fits_TrendParameters = df(index=['a','b','t0','sigma'])
        
        sns.set_theme(style="ticks")
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot()
        
        Fit = True
        Normalize = False
        for idx,column in enumerate(TempData.columns) :
            if Normalize :
                Min = min(TempData[column])
                TempData[column] = TempData[column]-Min
                Max = max(TempData[column])
                TempData[column] = TempData[column]/Max
            if column in Parameters :
                if 'a' in Parameters[column] :
                    FitParameters['a'].value = Parameters[column]['a']
                if 'b' in Parameters[column] :
                    FitParameters['b'].value = Parameters[column]['b']
                if 't0' in Parameters[column] :
                    FitParameters['t0'].value = Parameters[column]['t0']
                if 'sigma' in Parameters[column] :
                    FitParameters['sigma'].value = Parameters[column]['sigma']
            y = TempData[column].values.astype('float64')
            t = TempData.index.values
            FitResults = FitModel.fit(y, FitParameters, t=t)
            FitsResults.append(FitResults)
            fit_y = FitResults.eval(t=Fits_TrendFits.index)
            Fits_TrendFits[column] = fit_y
            Fits_TrendParameters[column] = np.array((FitResults.params['a'].value,FitResults.params['b'].value,FitResults.params['t0'].value,FitResults.params['sigma'].value))
            plt.scatter(t,y,label=column)
            plt.plot(Fits_TrendFits.index,fit_y)
        
        ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
        plt.xlabel('Time (fs)',fontsize=fontsize)
        plt.ylabel('Amplitude *au()',fontsize=fontsize)
        plt.legend(frameon=False, fontsize=fontsize, bbox_to_anchor=(1,0.8), loc="upper left")
        plt.show()
        display(Fits_TrendParameters)

        def FitsAppend2File_Clicked(b) :
            Data.to_hdf(DataFile[0] + '/' + DataFile[1],'Fits_TrendData',mode='a')
            Fits_TrendFits.to_hdf(DataFile[0] + '/' + DataFile[1],'Fits_TrendFits',mode='a')
            Fits_TrendParameters.to_hdf(DataFile[0] + '/' + DataFile[1],'Fits_TrendParameters',mode='a')
        FitsAppend2File = widgets.Button(description="Append to File")
        FitsAppend2File.on_click(FitsAppend2File_Clicked)

        display(FitsAppend2File)
    
    def UI(self) :
        
        dt = AnalysisTools.DataTools()
        
        out = widgets.Output()
        
        ##### Button Functions #####

        def UpdateFiles_Clicked(b):
            with open(Folders['Parameters']+'/'+self.ParametersFiles.value+'.yaml', 'r') as stream:
                par = yaml.safe_load(stream)
            self.DataFiles.options = dt.FileList(Folders['Fits'],[par['Runs']])
        UpdateFiles = widgets.Button(description="Update",layout = Layout(width='10%'))
        UpdateFiles.on_click(UpdateFiles_Clicked)
        
        def FitData_Clicked(b):
            with out :
                clear_output(True)
                self.LoadData([Folders['Parameters'],self.ParametersFiles.value],[Folders['Fits'],self.DataFiles.value])
                self.Fit()
        FitData = widgets.Button(description="Fit",layout = Layout(width='10%'))
        FitData.on_click(FitData_Clicked)

        ##### Widgets #####

        self.ParametersFiles = widgets.Dropdown(
            options=dt.FileList(Folders['Parameters'],['.yaml']),
            description='Parameter File',
            layout=Layout(width='70%'),
            style = {'description_width': '150px'},
            disabled=False,
        )

        with open(Folders['Parameters']+'/'+self.ParametersFiles.value+'.yaml', 'r') as stream:
            par = yaml.safe_load(stream)

        self.DataFiles = widgets.Dropdown(
            options=dt.FileList(Folders['Fits'],[par['Runs']]),
            description='Data File',
            layout=Layout(width='50%'),
            style = {'description_width': '150px'},
            disabled=False,
        )
        
        display(widgets.Box([self.ParametersFiles,UpdateFiles]))
        display(self.DataFiles)
        display(FitData)
        
        display(out)


class Angles :
    
    def __init__(self) :
        
        Folders['Parameters'] = os.getcwd()+'/../Parameters'
        Folders['Fits'] = os.getcwd()+'/../Fits'
        Folders['Figures'] = os.getcwd()+'/../Figures'
        Folders['Binned'] = os.getcwd()+'/../Binned/BT2'
    
    def LoadData(self,HorFile,VerFile) :
        
        dt = AnalysisTools.DataTools()
        
        # Load Data
        HorData = dt.LoadHDF(HorFile[0],HorFile[1])
        VerData = dt.LoadHDF(VerFile[0],VerFile[1])
        
        self.HorData = HorData
        self.VerData = VerData
    
    def Calculate(self) :
        
        HorData = self.HorData
        VerData = self.VerData
        
        # Plot formatting
        symbols = 5*['.','+','x','^','*','_','o']
        linestyle = 5*['solid','dotted','dashed','dashdot',(0,(3,3,1,3,1,3)),(0,(1,5)),(0,(5,10))]
        colors = 5*['black','blue','red','green','orange','purple','gray']
        markersize = 150
        linewidth=2
        fontsize = 16
        
        # Amplitudes
        HorFitsPars = HorData['Fits_Parameters']
        HorAmpCols = [col for col in HorFitsPars.columns if 'amp' in col]
        VerFitsPars = VerData['Fits_Parameters']
        VerAmpCols = [col for col in VerFitsPars.columns if 'amp' in col]
        CommonCols = [x for x in VerAmpCols if x in HorAmpCols]
        HorAmp = HorFitsPars[CommonCols]
        VerAmp = VerFitsPars[CommonCols]
        CommonDelays = [x for x in VerAmp.index if x in HorAmp.index]
        HorAmp = HorAmp.filter(items=CommonDelays,axis=0)
        VerAmp = VerAmp.filter(items=CommonDelays,axis=0)
        
        # Plot horizontal amplitudes
        plt.figure(figsize=(10, 5))
        ax = plt.subplot()
        for column in HorAmp :
            plt.plot(HorAmp.index,HorAmp[column],marker='o')
        plt.xlabel('Time (fs)',fontsize=fontsize)
        plt.ylabel('Intensity (au)',fontsize=fontsize)
        ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
        plt.title('Horizontal',fontsize=fontsize+4)
        plt.show()
        
        # Plot vertical amplitudes
        plt.figure(figsize=(10, 5))
        ax = plt.subplot()
        for column in VerAmp :
            plt.plot(VerAmp.index,VerAmp[column],marker='o')
        plt.xlabel('Time (fs)',fontsize=fontsize)
        plt.ylabel('Intensity (au)',fontsize=fontsize)
        ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
        plt.title('Vertical',fontsize=fontsize+4)
        plt.show()
        
        # Define functions
        
        pi = np.pi

        def cos(theta) :
            return np.cos(theta/180*pi)

        def sin(theta) :
            return np.sin(theta/180*pi)

        def OutOfPlane(theta,phi,gamma) :
            I = 1 - cos(theta)**2 * cos(gamma)**2 - sin(theta)**2 * sin(gamma)**2 * cos(phi)**2 - 2*sin(gamma)*cos(gamma)*sin(theta)*cos(theta)*cos(phi)
            return I

        def InPlane(theta,phi,gamma) :
            I = 1 - sin(gamma)**2 * sin(phi)**2
            return I

        def I(theta, gamma) :
            I_OP = OutOfPlane(theta,45,gamma)
            I_IP = InPlane(theta,45,gamma)
            return gamma, I_OP, I_IP

        # Simplified angle function: same as new analysis when x-ray angle (theta) set to zero
        def AngleFunction(theta) :
            return (np.sin((theta)/180*np.pi))**2 / (0.5+((np.cos((theta)/180*np.pi))**2/2))
        
        # Calculate and plot ratio versus angles
        Gamma = np.linspace(0,90,9001)
        x, OP, IP = I(3,Gamma)
        RatiosRef = OP/IP
        fontsize = 20
        sns.set_theme(style="ticks")
        fig, ax = plt.subplots(figsize=(10,5))
        sns.lineplot(x=x,y=OP, color='black', label='$I_{OP}$')
        sns.lineplot(x=x,y=IP, color='red', label='$I_{IP}$')
        sns.lineplot(x=x,y=RatiosRef, color='blue', label='$\\frac{I_{OP}}{I_{IP}}$')
        ax.lines[0].set_linestyle("dashed")
        ax.lines[1].set_linestyle("dotted")
        ax.set_xlabel('CO-Surface Angle (Degrees)',fontsize=fontsize)
        ax.set_ylabel('Intensity (norm)',fontsize=fontsize)
        ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
        plt.legend(frameon=False, fontsize=fontsize)
        plt.show()
        
        # Calculate ratios        
        Ratios = VerAmp/HorAmp.replace({ 0 : np.nan })
        Ratios = VerAmp/HorAmp.replace({ 0 : np.nan })
        Ratios['Precursor 1'] = Ratios['Precursor10_amplitude'] + Ratios['Precursor11_amplitude']
        Ratios['Precursor 2'] = Ratios['Precursor20_amplitude'] + Ratios['Precursor21_amplitude']
        Ratios = Ratios.rename({'Precursor2_amplitude': 'Precursor 2', 'Adsorbed_amplitude': 'Adsorbed'}, axis=1)
        Ratios = Ratios.reindex(columns=['Adsorbed','Precursor 1','Precursor 2'])
        
        # Calculate angles from surface normal
        Angles = pd.DataFrame(index=Ratios.index,columns=Ratios.columns)
        for Energy in Ratios :
            for Delay in Ratios[Energy].index.values :
                index = (np.abs(RatiosRef - Ratios[Energy].loc[Delay])).argmin()
                Angles[Energy].loc[Delay] = Gamma[index]

        # Plot function
        def Data2Plot(Data) :
            for idx,column in enumerate(Data) :
                label = column
                x = Data.index
                y = Data[column].values.astype('float64')
                sns.scatterplot(x=x,y=y,label=label,marker=symbols[idx],color=cm.Set1(idx),s=markersize)
                sns.lineplot(x=x,y=y,label='_',marker=symbols[idx],color=cm.Set1(idx),linewidth=linewidth)
                idx += 1

        # Ratio plot
        sns.set_theme(style="ticks")
        fig = plt.figure(figsize=(10, 5))
        rc('axes', linewidth=linewidth)
        ax = fig.add_subplot(1, 1, 1)
        Data2Plot(Ratios)
        ax.tick_params(axis='both',which='both',labelsize = fontsize)
        plt.ylabel('$I_{OP}/I_{IP}$',fontsize=fontsize)
        plt.xlabel('Delay (fs)',fontsize=fontsize)
        plt.title('Ratios',fontsize=fontsize+4)
        plt.legend(frameon=False, fontsize=fontsize, bbox_to_anchor=(1,0.65), loc="upper left")
        plt.show()
        
        # Angles plot
        sns.set_theme(style="ticks")
        fig = plt.figure(figsize=(10, 5))
        rc('axes', linewidth=linewidth)
        ax = fig.add_subplot(1, 1, 1)
        Data2Plot(Angles)
        ax.tick_params(axis='both',which='both',labelsize = fontsize)
        plt.ylabel('Angle (Degrees)',fontsize=fontsize)
        plt.xlabel('Delay (fs)',fontsize=fontsize)
        plt.title('Angles',fontsize=fontsize+4)
        plt.legend(frameon=False, fontsize=fontsize, bbox_to_anchor=(1,0.65), loc="upper left")
        plt.show()
        
        # Interactive angles plot
        fig = px.scatter(Angles)
        fig.update_traces(mode='lines+markers')
        fig.update_layout(xaxis_title='Delay (fs)',yaxis_title='Angle (Degrees)',legend_title='',height=500)
        fig.show()
        
        self.HorAmp = HorAmp
        self.VerAmp = VerAmp
        self.Ratios = Ratios
        self.Angles = Angles
        
        Filename = widgets.Text(
            value='Angles',
            placeholder='Filename',
            description='Filename:',
            disabled=False
        )
        
        def Angles2File_Clicked(b) :
            os.makedirs(Folders['Fits'], exist_ok=True)
            Angles.to_hdf(Folders['Fits']+'/'+Filename.value+'.hdf','Angles')
        Angles2File = widgets.Button(description="Save to File")
        Angles2File.on_click(Angles2File_Clicked)
        
        display(widgets.Box([Angles2File,Filename]))
    
    def UI(self) :
        
        dt = AnalysisTools.DataTools()
        
        out = widgets.Output()
        
        ##### Button Functions #####

        def UpdateFiles_Clicked(b):
            self.HorFiles.options = dt.FileList(Folders['Fits'],['Hor'])
            self.VerFiles.options = dt.FileList(Folders['Fits'],['Ver'])
        UpdateFiles = widgets.Button(description="Update",layout = Layout(width='10%'))
        UpdateFiles.on_click(UpdateFiles_Clicked)
        
        def Calculate_Clicked(b):
            with out :
                clear_output(True)
                self.LoadData([Folders['Fits'],self.HorFiles.value],[Folders['Fits'],self.VerFiles.value])
                self.Calculate()
        Calculate = widgets.Button(description="Calculate",layout = Layout(width='10%'))
        Calculate.on_click(Calculate_Clicked)

        ##### Widgets #####

        self.ParametersFiles = widgets.Dropdown(
            options=dt.FileList(Folders['Parameters'],['.yaml']),
            description='Parameter File',
            layout=Layout(width='70%'),
            style = {'description_width': '150px'},
            disabled=False,
        )

        with open(Folders['Parameters']+'/'+self.ParametersFiles.value+'.yaml', 'r') as stream:
            par = yaml.safe_load(stream)

        self.HorFiles = widgets.Dropdown(
            options=dt.FileList(Folders['Fits'],['Hor']),
            description='Horizontal',
            layout=Layout(width='50%'),
            style = {'description_width': '150px'},
            disabled=False,
        )
        
        self.VerFiles = widgets.Dropdown(
            options=dt.FileList(Folders['Fits'],['Ver']),
            description='Vertical',
            layout=Layout(width='50%'),
            style = {'description_width': '150px'},
            disabled=False,
        )
        
        display(widgets.Box([self.HorFiles,UpdateFiles]))
        display(self.VerFiles)
        display(Calculate)
        
        display(out)