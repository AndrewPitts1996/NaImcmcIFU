# Define likelihood and priors
import numpy as np
import scipy.special as sp
import math
import MgII_modeling

# Define the probability function as likelihood * prior.
def lnprior(theta):

    #print('lnprior theta: ',theta)
    lamred, logN, bD, Cf = theta

    sol = 2.998e5    # km/s
    transinfo = MgII_modeling.transitions()
    #vlim = 400.0     # km/s
    vlim = 400.0
    lamlim1 = -1.0 * (vlim * transinfo['lamred0'] / sol) + transinfo['lamred0']
    lamlim2 = (vlim * transinfo['lamred0'] / sol) + transinfo['lamred0']

    #logNlim1 = 10.0
    #logNlim2 = 15.0
    #logNlim1 = 14.0
    #logNlim1 = 13.5
    #logNlim2 = 16.5
    #logNlim1 = 14.1
    logNlim1 = 10.0
    logNlim2 = 18.0
    
    bDlim1 = 10.0
    bDlim2 = 200.0
    #bDlim2 = 500.0
    
    Cflim1 = 0.0
    Cflim2 = 1.0
    
    #if -5.0 < m < 5.0 and -10.0 < b < 10.0 and -10.0 < lnf < 10.0:
    if lamlim1 < lamred < lamlim2 and logNlim1 < logN < logNlim2 and bDlim1 < bD < bDlim2 and Cflim1 < Cf < Cflim2:
        return 0.0
    return -np.inf


def lnlike(theta, wave, flux, err, fwhm):
    #lamred, logN, bD, Cf = theta
    _,flx_model = MgII_modeling.model_MgII(theta,fwhm,wave)
    #inv_sigma2 = 1.0/(err**2)
    
    ## correct for model droppoff in value at boundaries
    inds = np.where(flx_model>=0.5)[0]
    w = slice(inds.min(),inds.max())
    
    return -0.5 * np.sum(((flux[w] - flx_model[w])/err[w])**2 - np.log(2 * np.pi / (err[w])**2))



def lnprob(theta, wave, flux, err, velres):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, wave, flux, err, velres)