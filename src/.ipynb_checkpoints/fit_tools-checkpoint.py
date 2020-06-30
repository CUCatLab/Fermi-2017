import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate


def line(x,a,b):
    return a*x+b

#def gaussian(x,a,b):
#    return np.exp(-x**2./(b**2))

def gaussian(x,a,b,c,d):
    return np.abs(a)*np.exp(-4*np.log(2)*(x-b)**2./(c**2))+d
    #return a*np.exp(-4*np.log(2)*(x-b)**2./(c**2))


def gaussian0(x,a,b,c):
    # return np.abs(a)*np.exp(-4*np.log(2)*(x-b)**2./(c**2))
    return a*np.exp(-4*np.log(2)*(x-b)**2./(c**2))

def exponential(x,a,b,c):
    return np.abs(a)*np.exp(-x/(np.abs(b)))+c
    
def poly2(x, a, b, c):
    return a*x**2 + b*x + c

def fit(function,x,y,p0=None,sigma=None,bounds=(-np.inf, np.inf)):
    '''
    fits a function and return the fit resulting parameters and curve
    '''
    popt,pcov = curve_fit(function,x,y,p0=p0,sigma=sigma, bounds=bounds)
    #x = np.arange(0,1e4)
    curve = function(x,*popt)
    perr = np.sqrt(np.diag(pcov))
    return popt,x,curve,perr

def fit_ponly(function,x,y,p0=None,sigma=None,bounds=None):
    '''
    fits a function and return the fit resulting parameters and curve
    '''
    popt,pcov = curve_fit(function,x,y,p0=p0,sigma=sigma, bounds=bounds)
    #x = np.arange(0,1e4)
    #curve = function(x,*popt)
    perr = np.sqrt(np.diag(pcov))
    return popt,perr

def gaussstep(x, sh, s, off, gw):
    
    gc = 0
    ga = 1
    
    # off = 0.085
    # sh = -0.01
    
    x_span   = np.max(x) - np.min(x)
    x_points = 100
    x_dens   = x_span/x_points
    
    x_temp   = np.arange(np.min(x) - 0.5*x_span, np.max(x) + 0.5*x_span, x_dens)
    x_gauss  = np.arange(-3*gw, 3*gw, x_dens)
    
    #step     = off + sh * np.heaviside(x_temp - s, 0)
    step     = sh * (np.sign(x_temp-s)) + off
    gauss    = gaussian0(x_gauss, ga, gc, gw)
    
    #print np.shape(step)
    #print np.shape(gauss)
    
    y = np.convolve(step, gauss, 'same');
    
    #print np.shape(x)
    #print np.shape(x_temp)
    #print np.shape(y)
    
    y = np.interp(x, x_temp, y)
    
    #if sh >= 0 :
    #    y = y / np.max(y) * (off+sh)
    #else :
    #    y = y / np.max(y) * (off)
    
    return y

def step(x, a, s, o) :
    
    step_temp = np.zeros(np.shape(x))
    
    for i in np.arange(len(x)) :
        if x[i] > s :
            step_temp[i] = 1
        elif x[i] < s :
            step_temp[i] = 0
        elif x[i] == s :
            step_temp[i] = 0.5
    
    
    y = a * step_temp + o
    
    return y

def g_heav(x, a, b, c):
    return a * (np.sign(x-b)) + c


def gauss2(x, t0, sig) :
    integrand = np.exp(-(x-t0)**2 / sig**2)
    return integrand

def gauss_int(x, a, t0, sig, o) :   
    result = np.zeros(len(x))
    for i in np.arange(len(x)) :
        result[i] = scipy.integrate.quad(lambda x: gauss2(x, t0, sig), -np.inf, x[i])[0]
    return o + result / np.mean(np.max(result)) * a


def gauss_int_0(x, a, t0, sig) :   
    result = np.zeros(len(x))
    for i in np.arange(len(x)) :
        result[i] = scipy.integrate.quad(lambda x: gauss2(x, t0, sig), -np.inf, x[i])[0]
    return result / np.mean(np.max(result)) * a

def gauss_int_0_2(x, a1, t01, sig1, a2, t02, sig2) :   
    result1 = np.zeros(len(x))
    result2 = np.zeros(len(x))
    for i in np.arange(len(x)) :
        result1[i] = scipy.integrate.quad(lambda x: gauss2(x, t01, sig1), -np.inf, x[i])[0]
    for i in np.arange(len(x)) :
        result2[i] = scipy.integrate.quad(lambda x: gauss2(x, t02, sig2), -np.inf, x[i])[0]
    
    return result1 / np.mean(np.max(result1)) * a1 + result2 / np.mean(np.max(result2)) * a2