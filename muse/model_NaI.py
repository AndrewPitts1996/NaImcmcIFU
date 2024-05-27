import numpy as np
import scipy.special as sp
import math
from linetools.spectra.xspectrum1d import XSpectrum1D
from linetools.spectra import convolve as lsc
from astropy import units as u
from astropy.units import Quantity
import pdb

# Set up constants for NaI
# From Cashman+17
def transitions():

    lamblu0 = 5891.5833
    lamred0 = 5897.5581

    #lamfblu0 = 3718.17822063
    #lamfred0 = 1875.4234758
    fblu0 = 6.50e-01
    fred0 = 3.24e-01
    
    lamfblu0 = lamblu0 * fblu0
    lamfred0 = lamred0 * fred0
    
    return {'lamblu0':lamblu0, 'lamred0':lamred0, 'lamfblu0':lamfblu0, 'lamfred0':lamfred0}


# Set up model line profile
# theta contains lamred, logN, bD, Cf (in that order)
def model_NaI(theta,velres,newwv):

    # First, get info on transitions
    sol = 2.998e5    # km/s
    transinfo = transitions()
    velratio = 1.0 + (transinfo['lamblu0'] - transinfo['lamred0'])/transinfo['lamred0']
    dmwv = 0.1   # in Angstroms
    
    lamred, logN, bD, Cf = theta

    N = 10.0**logN
    lamblu = lamred * velratio
    taured0 = N * 1.497e-15 * transinfo['lamfred0'] / bD
    taublu0 = N * 1.497e-15 * transinfo['lamfblu0'] / bD

    wv_unit = u.AA
    modwave = np.arange(5870.0,5920.0,dmwv)
    modwave_u = u.Quantity(modwave,unit=wv_unit)

    
    exp_red = -1.0 * (modwave - lamred)**2 / (lamred * bD / sol)**2
    exp_blu = -1.0 * (modwave - lamblu)**2 / (lamblu * bD / sol)**2

    taured = taured0 * np.exp(exp_red)
    taublu = taublu0 * np.exp(exp_blu)

    ## Unsmoothed model profile
    model_NaI = 1.0 - Cf + (Cf * np.exp(-1.0*(taublu + taured)))
    xspec = XSpectrum1D.from_tuple((modwave,model_NaI))
    
    ## Now smooth with a Gaussian resolution element
    ## Can try XSpectrum1D.gauss_smooth (couldn't get this to work)

    # FWHM resolution in pix
    midwv = (transinfo['lamblu0'] + transinfo['lamred0']) / 2.0
    wvres = midwv * velres / sol
    pxres = wvres / dmwv

   
    smxspec = xspec.gauss_smooth(pxres)
    
    
    ## Now rebin to match pixel size of observations
    ## Can try XSpectrum1D.rebin, need to input observed wavelength array
    #wv_unit = u.AA
    uwave = u.Quantity(newwv,unit=wv_unit)
    #uwave = np.array(newwv)
    # Rebinned spectrum
    rbsmxspec = smxspec.rebin(uwave)
    
    modwv = rbsmxspec.wavelength.value
    modflx = rbsmxspec.flux.value
    
    return {'modwv':modwv, 'modflx':modflx}
