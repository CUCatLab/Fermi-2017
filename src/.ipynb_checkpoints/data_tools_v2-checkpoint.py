import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import h5py
from os import walk
from importlib import reload

#import Blobfinder

import fit_tools
reload(fit_tools)

from fit_tools import *


def discover_files(path):
    '''
    Looks in the given directory and returns the filenames,
    '''
    for (dirpath, dirnames, filenames) in walk(path):
        break

    if len(filenames) != 0 :
        mask = np.ones(len(filenames), dtype=bool)
        for i in np.arange(len(filenames)) :
            if filenames[i][0] == '.' :
                mask[i] = 0
        filenames = np.array(filenames)
        filenames = filenames[mask]
        
    if len(dirnames) != 0 :
        mask = np.ones(len(dirnames), dtype=bool)
        for i in np.arange(len(dirnames)) :
            if dirnames[i][0] == '.' :
                mask[i] = 0
        dirnames = np.array(dirnames)
        dirnames = dirnames[mask]

    return dirnames,filenames


def do_runpath(run, run_type, bt, data_path) :
    '''
    Create path to run folder baser on beamtime, run number and base path
    '''    
    
    if bt == 1:
        run_path = data_path + 'XAS%03d/' % run
    elif bt == 2:
        if run_type == 'XAS':
            if run == 15 :
                raise NameError('For Run 15 in BT 2 choose XAS015 = Run 0 or XAS015_NoPump = Run 99')
            elif run == 0 :
                run_path = data_path + 'XAS015/'
            elif run == 99 :
                run_path = data_path + 'XAS015_NoPump/'
            elif run in [4,5,6,7,8,9,10,16,17,21,22,23,24,25,26,31] :
                run_path = data_path + 'XAS%03d_NoPump/' % run
            else :
                run_path = data_path + 'XAS%03d/' % run
        elif run_type == 'XES':
            if run in [1,2,3,4,5,6,7,8,9,10,11,12]:
                run_path = data_path + 'XES%03d_NoPump/' % run
            else :
                run_path = data_path + 'XES%03d/' % run
        else :
            raise NameError('run_type must be XAS or XES')
    else :
        raise NameError('BT must be 1 or 2')
        
    return run_path

def getHarm(run, run_type, bt) :
    if bt == 1 :
        Harm = 60
    elif bt == 2 :
        if run_type == 'XAS' :
            if run in np.concatenate([np.arange(0, 23+1), np.arange(30, 35+1), [99]]) :
                Harm = 60
            else :
                Harm = 55
        elif run_type == 'XES' :
            Harm = 55
        else :
           raise NameError('run_type must be XAS or XES')    
    else :
        raise NameError('BT must be 1 or 2')
        
    return Harm


def get_XAS_intensity(h5file,thr = 0):
    '''
    Load the XAS data, applies a threshold (thr) and returns the integrated intensity
    '''
    XAS_image = h5file['/Laser/BaslerImage2'].value
    XAS_image[XAS_image<thr]=0
    XAS_int = np.sum(np.sum(XAS_image,axis=1),axis=1)
    return XAS_int
    
    
def get_Basler_intensity(h5file, Basler, thr = 20):
    '''
    Load the integrated intensity of a Basler camera, applies a threshold (thr, default thr = 20) before integrating
    '''
    Image = h5file['/Laser/'+Basler].value
    Image = Image[:, 20:-20, 20:-20]
    Image[Image < thr]=0
    Int = np.sum(np.sum(Image,axis=1),axis=1)
    return Int
    

def get_Basler_projection(h5file, Basler, curv_file, thr = 20) :
    '''
    Load all Basler images projected onto the dispersive direction.
    '''
    
    # Load the curvature
    h5_curve = h5py.File(curv_file, 'r')
    p_curv = h5_curve['popt'].value
    h5_curve.close()
    
    # Load the images
    xes_images = h5file['/Laser/' + Basler].value
    
    xes_images = xes_images[:, 20:-20, 20:-20]
    xes_images[xes_images < thr] = 0
    
    # Curvature correction for each immage and projecting corrected image to get spectrum
    xes_spec = []
    for k in np.arange(np.shape(xes_images)[0]) :
        xes_images[k, :, :] = CurvCorr_XES_image(xes_images[k, :, :], p_curv)
        xes_spec.extend(np.sum(xes_images[k, :, :], axis = 0))
        
    return xes_spec


def get_Basler_blobs(h5file, Basler, clustersize = 5, threshold = 8):
    '''
    Does blob fining on the Basler images and returns coordinates of the blobs and number of blobs per shot
    '''
    
    # Blobfinder paramters
    bitdepth        = 8
    clustersize     = clustersize
    threshold       = threshold
    blobfind_params = {'bitdepth': bitdepth, 'clustersize': clustersize, 'threshold': threshold}
    
    # Load  images
    Images = h5file['/Laser/'+Basler].value
    
    n_shots = np.shape(Images)[0]
    
    blobs_x = []
    blobs_y = []
    n_blobs = np.zeros(n_shots)


    for k in np.arange(n_shots):
        #one_Image = Images[k, 20:-20, 20:-20]
        one_Image = np.array(Images[k, 20:-20, 20:-20], order='C')
        
        [x_pos,y_pos,num,integ]= Blobfinder.getBlobs(blobfind_params, one_Image)
        
        blobs_x.extend(x_pos)
        blobs_y.extend(y_pos)
        
        n_blobs[k] = len(x_pos)
        
    return blobs_x, blobs_y, n_blobs

    
def get_ms(h5file) :
    '''
    Reads in the mass spec trace and returns the ToF values and number of counts for each shot
    '''
    
    ms_data = h5file['/Lecroy/Wave1'].value
    
    n_shots = np.shape(ms_data)[0]
    
    n_counts = np.zeros(n_shots)
    tof_all  = []
    
    for k in np.arange(n_shots):
        # Clear zeros
        tof = ms_data[k, ms_data[k, :]!=0]
        # Multiply by sample width (5 ns)
        tof = tof * 5e-9
        
        tof_all.extend(tof)
        
        n_counts[k] = len(tof)
        
    return tof_all, n_counts 
    

def get_FEL_Spectrum_Calib(file_names, load_path, harm, debug, num_delays):
    '''
    Loads and returns the FEL spectrum and calibrated energy
    '''
    # constants
    h   = 4.135667662*10**(-15) #ev s
    c   = 299792458             #m/s    
    
    # Arrays to collect data in
    Intensity_all       = []
    WavelengthSpan_all  = []
    SeedLambda_all      = []
    FEL2_Wavelength_all = []
    Pixel2micron_all    = []
    
    # Loop over files supplied in file_names
    for j in range(len(file_names)):
        
        # If debugging / testing: do only first num_delays delay files
        if debug == True and j > num_delays : break
        
        try :
            h5file = h5py.File(load_path+file_names[j], 'r')
        except IOError :
            print('File ' + file_names[j] + ' could not be read. Skipping File!')
            continue
            
        # Load data for each file
        Intensity       = h5file['/photon_diagnostics/Spectrometer/hor_spectrum'].value
        WavelengthSpan  = h5file['/photon_diagnostics/Spectrometer/WavelengthSpan'].value
        SeedLambda      = h5file['/photon_source/SeedLaser/WavelengthFEL2'].value
        FEL2_Wavelength = h5file['/photon_diagnostics/Spectrometer/Wavelength'].value
        if 'photon_diagnostics/Spectrometer/Pixel2micron' in h5file:
            Pixel2micron   = h5file['photon_diagnostics/Spectrometer/Pixel2micron'].value
        else:
            print('Pixel2micron value is missing... Use Pixel2micron = 15.4639')
            Pixel2micron = 15.4639

        Intensity_all.extend(Intensity)
        WavelengthSpan_all.append(WavelengthSpan)
        SeedLambda_all.append(SeedLambda)
        FEL2_Wavelength_all.append(FEL2_Wavelength)
        Pixel2micron_all.append(Pixel2micron)
        
    Intensity_all       = np.array(Intensity_all)
    WavelengthSpan_all  = np.array(WavelengthSpan_all)
    SeedLambda_all      = np.array(SeedLambda_all)
    FEL2_Wavelength_all = np.array(FEL2_Wavelength_all)
    
    WavelengthSpan  = np.mean(WavelengthSpan_all)
    SeedLambda      = np.mean(SeedLambda_all)
    FEL2_Wavelength = np.mean(FEL2_Wavelength_all)
    Pixel2micron    = np.mean(Pixel2micron_all)
    
    #Harmonics = np.around(SeedLambda/FEL2_Wavelength)
    Harmonics = harm
    
    # Calculate central wavelength based on seed wavelength and harmonics
    Central_Wavelegth = SeedLambda / Harmonics
    
    # Get FEL spectrometer dispersion in nm per pixel
    Dispersion = Pixel2micron * WavelengthSpan / 1000

    # Get index of peak maximum in average FEL spectrum
    Avg_Spectrum = np.mean(Intensity,axis=0)
    Max_Ind_Avg  = np.mean(np.argmax(Avg_Spectrum))
    Avg_Spectrum = correct_FEL_spectrum(Avg_Spectrum, Max_Ind_Avg)
    
    Avg_Spectrum_ind = np.arange(len(Avg_Spectrum))

    p_int = [np.nanmax(Avg_Spectrum), Max_Ind_Avg, 1]
    popt,perr = fit_ponly(gaussian0, Avg_Spectrum_ind, Avg_Spectrum, p0=p_int, sigma=None, bounds=(0, np.inf))

    Peak_Ind = np.around(popt[1])
    
    if np.abs(Peak_Ind-Max_Ind_Avg) > 50:
        Peak_Ind = Max_Ind_Avg
        print('WARNING! Fit failed for energy calibration! Using index of max value.')
        plt.figure()
        plt.plot(Avg_Spectrum_ind, Avg_Spectrum, label = 'Avg FEL spectrum')
        plt.plot(Avg_Spectrum_ind, gaussian0(Avg_Spectrum_ind, popt[0], popt[1], popt[2]), label = 'fit')
        plt.legend(loc=0)
        plt.show()

    # Build wavelength axis
    
    Energy_ind = np.arange(-Peak_Ind, -Peak_Ind+len(Avg_Spectrum))
    Energy_nm = Energy_ind * Dispersion + Central_Wavelegth
    Energy_eV = h * c /(Energy_nm * 1e-9)

    return Intensity_all, Energy_eV


def correct_FEL_spectrum(Int_FEL_spectrum, Max_Ind_Avg, Peak_Width = 50, Bcg_Width = 10):
    '''
    Determines the background in the FEL spectrum in the range defined by Max_Ind_Avg, Peak_Width and Bcg_Width and subtracts it.
    Default is Peak_Width = 50, Bcg_Width = 10 given in pixels. Max_Ind_Avg is the index of the peak maximum in the average FEL spectrum.
    Max_Ind_Avg must be provided.
    Retruns the corrected, i.e. bcg subtracted Int_FEL_spectrum.
    '''
    
    X_FEL_spectrum = np.arange(len(Int_FEL_spectrum))
    
    offset_mask_lo = np.all([X_FEL_spectrum < Max_Ind_Avg - Peak_Width + Bcg_Width, X_FEL_spectrum > Max_Ind_Avg - Peak_Width - Bcg_Width], axis=0)
    offset_mask_hi = np.all([X_FEL_spectrum > Max_Ind_Avg + Peak_Width - Bcg_Width, X_FEL_spectrum < Max_Ind_Avg + Peak_Width + Bcg_Width], axis=0)
    
    offset_mask = np.any([offset_mask_lo, offset_mask_hi], axis=0)
    
    offset = np.average(Int_FEL_spectrum[offset_mask])
    
    Int_FEL_spectrum_corr = Int_FEL_spectrum - float(offset)
    
    return Int_FEL_spectrum_corr


def get_i0(file_names, load_path, harm, Peak_Width = 50, Bcg_Width = 10, get_FELstats = False, debug = False, num_delays = 2):
    '''
    Loads and corrects the FEL spectrum.
    Correction includes the subtraction of a backrgound estimated over the 
    given "offset_range", which is given in pixels! Not eV!.
    Returns the i0 of each shot and the FEL spectrum averaged over shots
    '''
    
    FEL_intensity, FEL_Energy = get_FEL_Spectrum_Calib(file_names, load_path, harm, debug, num_delays)
    
    FEL_intensity = np.array(FEL_intensity,dtype=float) # change type to float
    n_shots = FEL_intensity.shape[0]
    
    # average FEL spectrum (over all shots)
    Spectrum = np.average(FEL_intensity,axis=0)
    X_FEL_spectrum = np.arange(len(Spectrum)) # pixel axis of FEL_Spectrum
    
    # Get index of maximum in average FEL spectrum
    Max_Ind_Avg = int(np.mean(np.argmax(Spectrum)))
    
    ### For checking spectrometer bcg subtraction
    #offset_mask_lo = np.all([X_FEL_spectrum < Max_Ind_Avg - Peak_Width + Bcg_Width, X_FEL_spectrum > Max_Ind_Avg - Peak_Width - Bcg_Width], axis=0)
    #offset_mask_hi = np.all([X_FEL_spectrum > Max_Ind_Avg + Peak_Width - Bcg_Width, X_FEL_spectrum < Max_Ind_Avg + Peak_Width + Bcg_Width], axis=0)
    #offset_mask = np.any([offset_mask_lo, offset_mask_hi], axis=0)
    #
    #plt.figure
    #plt.plot(FEL_Energy, Spectrum, 'r')
    #plt.plot(FEL_Energy[offset_mask], Spectrum[offset_mask], 'bo')
    #plt.show()
    ### ### ### ###
    
    if get_FELstats :
        fitfail_counter = 0
        
        # Arrays to store fit results in
        amps = np.zeros(n_shots)
        centers = np.zeros(n_shots)
        widths = np.zeros(n_shots)
        
        # Check FEL fits
        CheckFELstats = False
        if CheckFELstats :
            plt.figure(figsize = [20, 20])
            plt_num = 100
            plt_counter = 0
            plt.rcParams.update({'font.size': 6})
            #plt.suptitle(h5file)
    
    # Loop over shots
    for j in range(int(n_shots)):
    # Correct FEL spectrum
        FEL_intensity[j,:] = correct_FEL_spectrum(FEL_intensity[j,:], Max_Ind_Avg, Peak_Width=Peak_Width, Bcg_Width = Bcg_Width)
        
        if get_FELstats :
            # Get initial fit paraters
            a_int = np.max(FEL_intensity[j,:]) # Amplitude
            c_int = FEL_Energy[FEL_intensity[j,:]==a_int] # Center
            c_int = np.mean(c_int) # In case more than one data points have the max value
            w_int = 1 # Width
        
            p_int = [a_int, c_int, w_int]
        
            try :
                popt,perr = fit_ponly(gaussian0, FEL_Energy, FEL_intensity[j,:], p0=p_int, sigma=None, bounds=(-np.inf, np.inf))
            except RuntimeError :
                popt = np.zeros(len(p_int))
                popt[:] = np.nan
                fitfail_counter = fitfail_counter + 1
        
            amps[j] = popt[0]
            centers[j] = popt[1]
            widths[j] = np.abs(popt[2]) # abs value is taken to prevent negative width. Could also set bounds in the fit, but this makes the fit VERY slow
            
            if CheckFELstats :
                
                plt_incr = np.floor(int(n_shots) / plt_num)
                if plt_incr == 0:
                    plt_incr = 1
                #if np.remainder(j, plt_incr) == 0 and plt_counter <= 99:
                if widths[j]>0.1 and widths[j]<0.2 and plt_counter <= 99:
                    plt_counter = plt_counter + 1
                    plt.subplot(np.ceil(np.sqrt(plt_num)), np.ceil(np.sqrt(plt_num)), plt_counter)
                    plt.plot(FEL_Energy[Max_Ind_Avg-100:Max_Ind_Avg+100], FEL_intensity[j,Max_Ind_Avg-100:Max_Ind_Avg+100], 'bo', ms = 1)
                    plt.plot(FEL_Energy[Max_Ind_Avg-100:Max_Ind_Avg+100], gaussian0(FEL_Energy[Max_Ind_Avg-100:Max_Ind_Avg+100],amps[j],centers[j],widths[j]), 'r-')
                    plt.title(j)
                        
    if CheckFELstats :
        plt.tight_layout()
        #plt.savefig('/Volumes/FERMI_2017/RESULTS/FELstats/')
        #plt.show()
        
    
    # calculate i0 (integrate FEL spectrum around peak maximum of average spectrum)
    i0_mask = np.all([X_FEL_spectrum > Max_Ind_Avg - Peak_Width, X_FEL_spectrum < Max_Ind_Avg + Peak_Width], axis=0)
    i0 = np.sum(FEL_intensity[:,i0_mask],axis=1)
    
    if get_FELstats :
        return i0, FEL_Energy, Spectrum, amps, centers, widths, fitfail_counter
    else :
        return i0, FEL_Energy, Spectrum

def CurvCorr_XES_image(image_raw, p_curv):
    """
    Corrects an XES image with a given curavture.
    p_curv = [p1, p2, p3] with curv = p1*x^2 + p2*x + p3
    """

    image_cor = np.zeros(np.shape(image_raw))
    
    p1 = p_curv[0]
    p2 = p_curv[1]
    p3 = p_curv[2]

    for i in np.arange(np.shape(image_raw)[0]) :
        shift = poly2(i, p1, p2, p3) - p3
        shift = -int(np.round(shift))
        
        image_cor[i] = np.roll(image_raw[i], shift)
    
    return image_cor

def pix2eV(pix, p_disp) :

    return line(pix, p_disp[0], p_disp[1])
    
    

def mm2fs(mm) :
    c = 299792458000 * 1e-15 # mm per fs
    return mm*2/(c)

def fs2mm(fs) :
    c = 299792458000 * 1e-15 # mm per fs
    return fs*c/2

