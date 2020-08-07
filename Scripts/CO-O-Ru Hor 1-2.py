import numpy as np
from pandas import DataFrame as df
import lmfit
from lmfit import model
from lmfit.models import GaussianModel, LinearModel, VoigtModel, PolynomialModel
import AnalysisTools

class Run :

    def __init__(self) :

        pass

    def Parameters(self,Region) :

        dic = {
            'FolderPath': '../../Binned/BT2/',
            'Runs': 'XAS_018_020',
            'BackgroundRun': 'XAS_099_099_E100.h5',
    #         'BackgroundRun': 'None',
            'NormROI': (285, 286.4),
            'xOffset': -0.31,
            'Normalize': False,
            'ScalingFactor': 1,
        }

        if Region == 'Pi Star' :
            dic['ROI'] = (286, 290)
            dic['NumPeaks'] = 8
            dic['NumRefPeaks'] = 2
            dic['ZonesDelayRange'] = (-2000,10000)
            dic['Zones'] = {
                            'Low': {'Range': (286.6,287.0), 'a': 0.002, 'b': 0.003, 't0': 0, 'sigma': 237},
                            'Vibrational1': {'Range': (287.2,287.5), 'a': 0.011, 'b': 0.0004, 't0': 260, 'sigma': 990},
                            'Vibrational2': {'Range': (287.6,287.9), 'a': 0.011, 'b': 0.0004, 't0': 260, 'sigma': 990},
                            'Negative': {'Range': (288.2,289), 'a': -0.004, 'b': 0.0076, 't0': 315, 'sigma': 415},
                            # 'Negative': {'Range': (288.5,289), 'a': -0.004, 'b': 0.01, 't0': 0, 'sigma': 910},
                            # 'Vibrational': {'Range': (287.1,287.4), 'a': 0.0033, 'b': 0.0027, 't0': 245, 'sigma': 265},
                            # 'Delta v < 0': {'Range': (286.7,287.1), 'a': 0.0018, 'b': 0.001, 't0': 0, 'sigma': 1200},
                            # 'Transition state': {'Range': (287.5,287.8), 'a': 0.02, 'b': 0.0094, 't0': 700, 'sigma': 880}
                           }

        if Region == 'Middle' :
            dic['ROI'] = (289,291.8)
            dic['NumPeaks'] = 1
            dic['NumRefPeaks'] = 0

        if Region == 'Shape Resonance' :
            dic['ROI'] = (292,295)
            dic['NumPeaks'] = 1
            dic['NumRefPeaks'] = 0

        return dic

    def FitData(self,Energy,Delay,Signal,ErrorBars,Region,NumberPeaks,NumRefPeaks) :

        ##### Build fit model #####

        ModelString = 'L'
        i = 0
        while i < NumberPeaks :
            ModelString = ModelString + 'G'
            i+=1
        fit = AnalysisTools.FitTools(ModelString)
        Params = fit.Model.make_params()

        ##### Fit data #####

        i = 0
        while i < len(Delay) :

            ##### Remove NaNs #####

            remove = np.argwhere(np.isnan(Signal[i]))
            x = np.delete(Energy,remove)
            y = np.delete(Signal[i],remove)
            err = np.delete(ErrorBars[i],remove)

            ##### Define parameters #####

            """
            CO desorption on Ru(0001) values:
                CO (adsorbed):  288.01 eV  Unpumped
                                287.92 eV  Pumped
                CO (gas phase): 287.42 eV  0-0 transition
                                0.256  eV  vibrational spacing
            """

            Vibrational_Sigma = 0.12

            if Region == 'Pi Star' :

                Params['L1_intercept'].value = 0.27450087
                Params['L1_intercept'].vary = True
                Params['L1_slope'].value = -0.000955913
                Params['L1_slope'].vary = True
                Params['G1_amplitude'].value = 0.03
                Params['G1_amplitude'].min = 0
                Params['G1_center'].value = 288.01
                Params['G1_center'].min = 287.92
                Params['G1_center'].vary = True
                Params['G1_sigma'].value = 0.166
                Params['G1_sigma'].vary = False
                if NumberPeaks >= 2 :
                    Params['G2_amplitude'].value = 0.01
                    Params['G2_amplitude'].vary = True
                    Params['G2_amplitude'].min = 0
                    Params['G2_center'].value = 287.42
                    Params['G2_center'].vary = False
                    Params['G2_sigma'].value = Vibrational_Sigma
                    Params['G2_sigma'].min = 0.1
                    Params['G2_sigma'].vary = False
                if NumberPeaks >= 3 :
                    Params['G3_amplitude'].set(expr='G2_amplitude*1.18/4.1')
                    Params['G3_amplitude'].min = 0
                    Params['G3_amplitude'].vary = True
                    Params['G3_center'].set(expr='G2_center+0.256')
                    Params['G3_center'].vary = False
                    Params['G3_sigma'].value = Vibrational_Sigma
                    Params['G3_sigma'].min = 0.1
                    Params['G3_sigma'].vary = False
                if NumberPeaks >= 4 :
    #                 Params['G4_amplitude'].set(expr='G2_amplitude*0.21/4.1')
                    Params['G4_amplitude'].min = 0
                    Params['G4_amplitude'].vary = True
                    Params['G4_center'].set(expr='G2_center+2*0.256')
                    Params['G4_center'].vary = False
                    Params['G4_sigma'].value = Vibrational_Sigma
                    Params['G4_sigma'].min = 0.1
                    Params['G4_sigma'].vary = False
                if NumberPeaks >= 5 :
                    Params['G6_amplitude'].set(expr='G5_amplitude/1.8')
                    Params['G5_amplitude'].min = 0
    #                 Params['G5_amplitude'].vary = True
                    Params['G5_center'].set(expr='G2_center-0.256')
                    Params['G5_center'].vary = False
                    Params['G5_sigma'].value = Vibrational_Sigma
                    Params['G5_sigma'].vary = False
                if NumberPeaks >= 6 :
    #                 Params['G6_amplitude'].set(expr='G2_amplitude*0.142496041')
                    Params['G6_amplitude'].min = 0
                    Params['G6_amplitude'].vary = True
                    Params['G6_center'].set(expr='G2_center-2*0.256')
                    Params['G6_center'].vary = False
                    Params['G6_sigma'].value = Vibrational_Sigma
                    Params['G6_sigma'].vary = False
                if NumberPeaks >= 7 :
                    Params['G7_amplitude'].value= 0.01
                    Params['G7_amplitude'].min = 0
                    Params['G7_amplitude'].vary = True
                    Params['G7_center'].value = 287.65
                    Params['G7_center'].vary = False
                    Params['G7_sigma'].value = 0.1
                    Params['G7_sigma'].vary = False
                if NumberPeaks >= 8 :
                    Params['G8_amplitude'].value = 1
                    Params['G8_amplitude'].min = 0
                    Params['G8_amplitude'].vary = True
                    Params['G8_center'].value = 288.13
                    Params['G8_center'].min = 288.05
                    Params['G8_center'].vary = True
                    Params['G8_sigma'].value = 0.576
                    Params['G8_sigma'].value = 0.5
                    Params['G8_sigma'].vary = False

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
                        Params['L1_intercept'].value = FitParameters[0,1]
                        Params['L1_intercept'].vary = False
                        Params['L1_slope'].value = FitParameters[0,2]
                        Params['L1_slope'].vary = False
                        Params['G1_amplitude'].value = FitParameters[0,3]
                        Params['G1_center'].value = FitParameters[0,4]
                        Params['G1_sigma'].value = FitParameters[0,5]
                        Shift = 0
                        if Delay[i] > 100 :
                            Shift = -0.045
                        if Delay[i] > 400 :
                            Shift = -0.09
    #                     if Delay[i] > 600 :
    #                         Shift = -0.17
    #                     if Delay[i] > 900 :
    #                         Shift = -0.21
                        Params['G1_center'].value = FitParameters[0,4] + Shift
                        if NumRefPeaks >= 2 :
                            Params['G8_amplitude'].set(expr='G1_amplitude * '+str(FitParameters[0,24]/FitParameters[0,3]))
                            Params['G8_center'].set(expr='G1_center + '+str(FitParameters[0,25]-FitParameters[0,4]))
                            Params['G8_sigma'].value = FitParameters[0,26]

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

            FitResults = fit.Fit(x,y,err,Delay[i],Params)
            fit.Plot()
            if i == 0 :
                FitParameters = np.zeros((len(Delay),len(fit.Parameters)))
                Fit_Signal = np.zeros((len(Delay),len(fit.fit_y)))
            FitParameters[i] = fit.Parameters
            Fit_Energy = fit.fit_x
            Fit_Signal[i] = fit.fit_y

            i += 1

            print('_'*110)

        FitParameters = df(data=FitParameters,columns=fit.ParameterNames)

        fit.PlotAnalysis('amp',FitParameters,fit.ParameterNames)
        fit.PlotAnalysis('ω',FitParameters,fit.ParameterNames)
        fit.PlotAnalysis('σ',FitParameters,fit.ParameterNames)
        fit.PlotDataAndFits(Energy, Signal, Fit_Energy, Fit_Signal, Delay)

        self.fit = fit
        self.FitParameters = FitParameters
        self.Fit_Energy = Fit_Energy
        self.Fit_Signal = Fit_Signal
