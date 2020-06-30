import numpy as np
from pandas import DataFrame as df
import lmfit
from lmfit import model
from lmfit.models import GaussianModel, LinearModel, VoigtModel, PolynomialModel
import analysis_tools

def Parameters(Region) :

    dic = {
        'FolderPath': '../../Binned/BT2/',
        'Runs': 'XAS_018_020',
        'BackgroundRun': 'XAS_099_099_bin.h5',
#         'BackgroundRun': 'None',
        'ROI': (286, 290),
        'NormROI': (285, 286.4),
        'NumPeaks': 8,
        'NumRefPeaks': 2,
        'xOffset': -0.35,
        'Normalize': True,
        'ScalingFactor': 1,
    }

    if Region == 'Middle' :
        dic['ROI'] = (289,291.8)
        dic['NumPeaks'] = 1        
        dic['NumRefPeaks'] = 0

    if Region == 'Shape Resonance' :
        dic['ROI'] = (290,295)
        dic['NumPeaks'] = 1        
        dic['NumRefPeaks'] = 0
    
    return dic

def FitData(Energy,Delay,Signal,ErrorBars,Region,NumberPeaks,NumRefPeaks) :

    ##### Build fit model #####

    ModelString = 'L'
    i = 0
    while i < NumberPeaks :
        ModelString = ModelString + 'G'
        i+=1
    Fits = analysis_tools.Fits(ModelString)
    Params = Fits.Model.make_params()

    ##### Fit data #####

    i = 0
    while i < len(Delay) :

        ##### Remove NaNs #####

        remove = np.argwhere(np.isnan(Signal[i]))
        x = np.delete(Energy,remove)
        y = np.delete(Signal[i],remove)
        err = np.delete(ErrorBars[i],remove)

        ##### Define parameters #####

        if Region == 'Pi Star' :

            Params['L1_slope'].value = 0
            Params['L1_slope'].vary = True
            Params['L1_intercept'].value = 0
            Params['L1_intercept'].vary = True
            Params['G1_amplitude'].value = 0.03
            Params['G1_amplitude'].min = 0
            Params['G1_center'].value = 288.05
            Params['G1_center'].vary = True
            Params['G1_sigma'].value = 0.17
            Params['G1_sigma'].vary = True
            if NumberPeaks >= 2 :
                Params['G2_amplitude'].value = 0.02
                Params['G2_amplitude'].vary = True
                Params['G2_amplitude'].min = 0
                Params['G2_center'].value = 287.42
                Params['G2_center'].vary = False
                Params['G2_sigma'].value = 0.1
                Params['G2_sigma'].vary = False
            if NumberPeaks >= 3 :
                Params['G3_amplitude'].set(expr='G2_amplitude*1.18/4.1')
                Params['G3_amplitude'].min = 0
                Params['G3_amplitude'].vary = True
                Params['G3_center'].set(expr='G2_center+0.256')
                Params['G3_center'].vary = False
                Params['G3_sigma'].value = 0.1
                Params['G3_sigma'].vary = False
            if NumberPeaks >= 4 :
                Params['G4_amplitude'].set(expr='G2_amplitude*0.21/4.1')
                Params['G4_amplitude'].vary = True
                Params['G4_amplitude'].min = 0
                Params['G4_center'].set(expr='G2_center+2*0.256')
                Params['G4_center'].vary = False
                Params['G4_sigma'].value = 0.1
                Params['G4_sigma'].vary = False
            if NumberPeaks >= 5 :
                Params['G5_amplitude'].set(expr='G2_amplitude*0.3')
                Params['G5_amplitude'].min = 0
                Params['G5_amplitude'].vary = True
                Params['G5_center'].set(expr='G2_center-0.256')
                Params['G5_center'].vary = False
                Params['G5_sigma'].value = 0.1
                Params['G5_sigma'].vary = False
            if NumberPeaks >= 6 :
                Params['G6_amplitude'].set(expr='G2_amplitude*0.087658654')
                Params['G6_amplitude'].min = 0
                Params['G6_amplitude'].vary = True
                Params['G6_center'].set(expr='G2_center-2*0.256')
                Params['G6_center'].vary = False
                Params['G6_sigma'].value = 0.1
                Params['G6_sigma'].vary = False
            if NumberPeaks >= 7 :
                Params['G7_amplitude'].value = 0.01
                Params['G7_amplitude'].min = 0
                Params['G7_amplitude'].vary = True
#                 Params['G7_center'].value = 287.63
                Params['G7_center'].value = 287.69
                Params['G7_center'].vary = False
                Params['G7_sigma'].value = 0.08
                Params['G7_sigma'].vary = False
            if NumberPeaks >= 8 :
                Params['G8_amplitude'].value = 1
                Params['G8_amplitude'].min = 0
                Params['G8_amplitude'].vary = True
                Params['G8_center'].value = 288.09
                Params['G8_center'].vary = True
                Params['G8_sigma'].value = 0.586
                Params['G8_sigma'].vary = True

            # Reference delay
            if NumRefPeaks >= 1 :
                if i == 0 :
                    if NumberPeaks >= 2 :
                        Params['G2_amplitude'].value = 0
                        Params['G2_amplitude'].vary = False
                    if NumberPeaks >= 3 :
                        Params['G3_amplitude'].value = 0
                        Params['G3_amplitude'].vary = False
                    if NumberPeaks >= 4 :
                        Params['G4_amplitude'].value = 0
                        Params['G4_amplitude'].vary = False
                    if NumberPeaks >= 5 :
                        Params['G5_amplitude'].value = 0
                        Params['G5_amplitude'].vary = False
                    if NumberPeaks >= 6 :
                        Params['G6_amplitude'].value = 0
                        Params['G6_amplitude'].vary = False
                    if NumberPeaks >= 7 :
                        Params['G7_amplitude'].value = 0
                        Params['G7_amplitude'].vary = False
                else :
                    Params['G1_amplitude'].value = FitParameters[0,3]
                    Params['G1_center'].value = FitParameters[0,4]
                    Params['G1_center'].vary = False
                    Params['G1_sigma'].value = FitParameters[0,5]
                    Params['G1_sigma'].vary = False
#             if Delay[i] > 175 :
#                 Params['G1_center'].value = 287.92
#                 Params['G1_center'].vary = False
            if NumRefPeaks >= 2 :
                if i != 0 :
                    Params['G8_amplitude'].set(expr='G1_amplitude * '+str(FitParameters[0,24]/FitParameters[0,3]))
                    Params['G8_center'].value = FitParameters[0,25]
                    Params['G8_center'].vary = False
                    Params['G8_sigma'].value = FitParameters[0,26]
                    Params['G8_sigma'].vary = False

        if Region == 'Middle' :

            Params['L1_slope'].value = -0.003129523
            Params['L1_slope'].vary = False
            Params['L1_intercept'].value = 0.916163878
            Params['L1_intercept'].vary = False
            Params['G1_amplitude'].value = 0.03
            Params['G1_amplitude'].min = 0
            Params['G1_center'].value = 290.9
            Params['G1_center'].vary = False
            Params['G1_sigma'].value = 0.35

        if Region == 'Shape Resonance' :

            Params['L1_slope'].value = -0.003402839
            Params['L1_slope'].vary = False
            Params['L1_intercept'].value = 0.997181538
            Params['L1_intercept'].vary = False
            Params['G1_amplitude'].value = 0.03
            Params['G1_amplitude'].min = 0
            Params['G1_center'].value = 293.76
            Params['G1_center'].vary = False
            Params['G1_sigma'].value = 0.34
            Params['G1_sigma'].vary = False
            
        ##### Fit the data #####

        FitResults = Fits.Fit(x,y,err,Delay[i],Params)
        Fits.Plot()
        if i == 0 :
            FitParameters = np.zeros((len(Delay),len(Fits.Parameters)))
            Fit_Signal = np.zeros((len(Delay),len(Fits.fit_y)))
        FitParameters[i] = Fits.Parameters
        Fit_Energy = Fits.fit_x
        Fit_Signal[i] = Fits.fit_y

        i += 1

        print('_'*110)
    
    FitParameters = df(data=FitParameters,columns=Fits.ParameterNames)
    return Fits, FitParameters, Fit_Energy, Fit_Signal