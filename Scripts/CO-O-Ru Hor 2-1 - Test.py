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
            'Runs': 'XAS_014_014',
            'BackgroundRun': 'XAS_099_099_bin.h5',
    #         'BackgroundRun': 'None',
            'NormROI': (285, 286.4),
            'xOffset': -0.31,
            'Normalize': False,
            'ScalingFactor': 0.62,
        }

        if Region == 'Pi Star' :
            dic['ROI'] = (286, 290)
            dic['NumPeaks'] = 6
            dic['NumRefPeaks'] = 1
            dic['ZonesDelayRange'] = (-2000,10000)
            dic['Zones'] = {
                            'Low': {'Range': (286.6,287.0), 'a': 0.002, 'b': 0.003, 't0': 0, 'sigma': 237},
                            'Vibrational1': {'Range': (287.4,287.5), 'a': 0.011, 'b': 0.0004, 't0': 260, 'sigma': 990},
                            'Vibrational2': {'Range': (287.6,287.7), 'a': 0.011, 'b': 0.0004, 't0': 260, 'sigma': 990},
                            'Negative': {'Range': (288.2,289), 'a': -0.004, 'b': 0.0076, 't0': 315, 'sigma': 415},
                            # 'Low1': {'Range': (286.6,286.8), 'a': 0.002, 'b': 0.003, 't0': 0, 'sigma': 237},
                            # 'Low2': {'Range': (286.8,287.0), 'a': 0.002, 'b': 0.003, 't0': 0, 'sigma': 237},
                            # 'Vibrational1': {'Range': (287.0,287.2), 'a': 0.011, 'b': 0.0004, 't0': 260, 'sigma': 990},
                            # 'Vibrational2': {'Range': (287.2,287.4), 'a': 0.011, 'b': 0.0004, 't0': 260, 'sigma': 990},
                            # 'Unknown2': {'Range': (287.4,287.6), 'a': 0.023, 'b': 0.015, 't0': 500, 'sigma': 750},
                            # 'Unknown3': {'Range': (287.6,287.8), 'a': 0.023, 'b': 0.015, 't0': 500, 'sigma': 750},
                            # 'Unknown4': {'Range': (287.8,288.0), 'a': 0.023, 'b': 0.015, 't0': 500, 'sigma': 750},
                            # 'Unknown5': {'Range': (288.0,288.2), 'a': 0.023, 'b': 0.015, 't0': 500, 'sigma': 750},
                            # 'Unknown6': {'Range': (288.2,288.4), 'a': 0.023, 'b': 0.015, 't0': 500, 'sigma': 750},
                            # 'Negative': {'Range': (288.4,289), 'a': -0.004, 'b': 0.0076, 't0': 315, 'sigma': 415},
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

            ##### Define constraints #####

            """
            CO desorption on Ru(0001) values:
                CO (adsorbed):  288.01 eV  Unpumped
                                287.92 eV  Pumped
                CO (gas phase): 287.42 eV  0-0 transition
                                0.256  eV  vibrational spacing
            """

            Vibrational_Sigma = 0.1

            if Region == 'Pi Star' :

                Params['L1_intercept'].value = 0.159991829
                Params['L1_intercept'].vary = True
                Params['L1_slope'].value = -0.000555426
                Params['L1_slope'].vary = True
                Params['G1_amplitude'].value = 0.02
                Params['G1_amplitude'].min = 0
                Params['G1_center'].value = 288.01
                Params['G1_center'].min = 287.92
                Params['G1_center'].vary = True
                Params['G1_sigma'].value = 0.415
                Params['G1_sigma'].vary = False
                if NumberPeaks >= 2 :
                    Name = 'G2'
                    Params[Name+'_amplitude'].value = 0.01
                    Params[Name+'_amplitude'].vary = True
                    Params[Name+'_amplitude'].min = 0
                    Params[Name+'_center'].value = 287.42
                    Params[Name+'_center'].vary = True
                    Params[Name+'_sigma'].value = Vibrational_Sigma
                    Params[Name+'_sigma'].min = 0.1
                    Params[Name+'_sigma'].vary = True
                if NumberPeaks >= 3 :
                    Name = 'G3'
                    Params[Name+'_amplitude'].min = 0
                    Params[Name+'_amplitude'].vary = True
                    Params[Name+'_center'].value = 287
                    Params[Name+'_center'].max = 287.4
                    Params[Name+'_center'].vary = True
                    Params[Name+'_sigma'].value = .18
                    Params[Name+'_sigma'].min = .1
                    Params[Name+'_sigma'].max = .24
                    Params[Name+'_sigma'].vary = True
                if NumberPeaks >= 4 :
                    Name = 'G4'
                    Params[Name+'_amplitude'].value= 0.01
                    Params[Name+'_amplitude'].min = 0
                    Params[Name+'_amplitude'].vary = True
                    Params[Name+'_center'].value = 287.65
                    Params[Name+'_center'].vary = True
                    Params[Name+'_sigma'].value = 0.1
                    Params[Name+'_sigma'].vary = True
                # if NumberPeaks >= 3 :
                #     Params['G3_amplitude'].set(expr='G2_amplitude*1.18/4.1')
                #     Params['G3_amplitude'].min = 0
                #     Params['G3_amplitude'].vary = True
                #     Params['G3_center'].set(expr='G2_center+0.256')
                #     Params['G3_center'].vary = False
                #     Params['G3_sigma'].value = Vibrational_Sigma
                #     Params['G3_sigma'].min = 0.1
                #     Params['G3_sigma'].vary = False
                # if NumberPeaks >= 4 :
                #     Params['G4_amplitude'].set(expr='G2_amplitude*0.21/4.1')
                #     Params['G4_amplitude'].vary = True
                #     Params['G4_amplitude'].min = 0
                #     Params['G4_center'].set(expr='G2_center+2*0.256')
                #     Params['G4_center'].vary = False
                #     Params['G4_sigma'].value = Vibrational_Sigma
                #     Params['G4_sigma'].min = 0.1
                #     Params['G4_sigma'].vary = False
                if NumberPeaks >= 5 :
    #                 Params['G5_amplitude'].set(expr='G2_amplitude*0.381932549')
                    Params['G5_amplitude'].min = 0
                    Params['G5_amplitude'].vary = True
                    Params['G5_center'].value = 287
                    Params['G5_center'].max = 287.4
                    Params['G5_center'].vary = True
                    Params['G5_sigma'].value = .18
                    Params['G5_sigma'].min = .1
                    Params['G5_sigma'].max = .24
                    Params['G5_sigma'].vary = False
                if NumberPeaks >= 6 :
                    Params['G6_amplitude'].value= 0.01
                    Params['G6_amplitude'].min = 0
                    Params['G6_amplitude'].vary = True
                    Params['G6_center'].value = 287.65
                    Params['G6_center'].vary = False
                    Params['G6_sigma'].value = 0.1
                    Params['G6_sigma'].vary = False

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
