import numpy as np
import scipy.special as sp
import math
from linetools.spectra.xspectrum1d import XSpectrum1D
from linetools.spectra import convolve as lsc
from astropy import units as u
from astropy.units import Quantity
import numpy.ma as ma
import matplotlib.pyplot as plt

def transitions():
    # Set up constants for MgII
    # From Cashman 2017
    lamblu0 = 2796.352
    lamred0 = 2803.53

    fblu0 = 0.613
    fred0 = 0.306
    
    lamfblu0 = lamblu0 * fblu0
    lamfred0 = lamred0 * fred0
    
    return {'lamblu0':lamblu0, 'lamred0':lamred0, 'lamfblu0':lamfblu0, 'lamfred0':lamfred0}



# Set up model line profile
# theta contains lamred, logN, bD, Cf (in that order)
def model_MgII(theta,fwhm,newwv):

    ## First, get info on transitions
    sol = 2.998e5    # speed of light km/s
    transinfo = transitions() # transition wavelength and quantum oscillator values
    velratio = 1.0 + (transinfo['lamblu0'] - transinfo['lamred0'])/transinfo['lamred0']
    dmwv = 0.1   # in Angstroms
    
    ## feature wavelength, column density, doppler param/velocity width, covering fraction
    lamred, logN, bD, Cf = theta 
    
    N = 10.0**logN
    lamblu = lamred * velratio
    taured0 = N * 1.497e-15 * transinfo['lamfred0'] / bD
    taublu0 = N * 1.497e-15 * transinfo['lamfblu0'] / bD

    wv_unit = u.AA
    modwave = np.arange(int(newwv.min()),int(newwv.max()),dmwv)
    modwave_u = u.Quantity(modwave,unit=wv_unit)

    
    exp_red = -1.0 * (modwave - lamred)**2 / (lamred * bD / sol)**2
    exp_blu = -1.0 * (modwave - lamblu)**2 / (lamblu * bD / sol)**2

    taured = taured0 * np.exp(exp_red)
    taublu = taublu0 * np.exp(exp_blu)

    ## Unsmoothed model profile
    model_MgII = 1.0 - Cf + (Cf * np.exp(-1.0*(taublu + taured)))
    xspec = XSpectrum1D.from_tuple((modwave,model_MgII))
    
    ## Now smooth with a Gaussian resolution element
    ## FWHM resolution in pix
    # XSpectrum1D.gauss_smooth
    smxspec = xspec.gauss_smooth(fwhm)
    
    
    ## Now rebin to match pixel size of observations
    ## Can try XSpectrum1D.rebin, need to input observed wavelength array
    #wv_unit = u.AA
    uwave = u.Quantity(newwv,unit=wv_unit)
    #uwave = np.array(newwv)
    # Rebinned spectrum
    rbsmxspec = smxspec.rebin(uwave)
    
    modwv = rbsmxspec.wavelength.value
    modflx = rbsmxspec.flux.value
    
    #return {'modwv':modwv, 'modflx':modflx}
    return modwv, modflx


## normalize to the continuum around doublet
## returns the region of the spectrum
def continuum_normalize(flux, waves, error):    
    
    #### define the continuum region for normalization
    regionb = 2765, 2780
    regionr = 2810, 2825
    indb = np.where((waves>regionb[0]) & (waves<regionb[1]) & (flux!=0))
    indr = np.where((waves>regionr[0]) & (waves<regionr[1]) & (flux!=0))
    fitregion = np.concatenate((indb[0],indr[0]))
    

    ## making data regio slightly larger than fitregion
    ## conditional to avoid failing where data is masked or does not contain region wavelengths
    region_w = slice(np.argmin(abs(2765-waves)),np.argmin(abs(2825-waves)))
    region = flux[region_w] != 0
    
    
    wav = waves[region_w][region]
    flx = flux[region_w][region]
    err = error[region_w][region]
    
    
    ## linear fit continuum and normalize flux
    fitflx = flux[fitregion]
    fitwav = waves[fitregion]
    med = np.median(fitflx)
    sig = np.std(fitflx)
    w = abs(fitflx - med) > 1 * sig
    
    
    p = np.polyfit(fitwav[~w], fitflx[~w], deg=1)
    continuum = np.polyval(p,wav)
    
    normflx = flx/continuum
    normerr = err/continuum
    
    return wav,normflx,normerr



##get the fwhm in pixels for gaussian smoothing
def get_fwhm(data,specres):
    
    lamred = 2803 * (1+data['z'])
    ind = np.argmin(abs(data['wavelength']-lamred))
    region = data['wavelength'][ind-20:ind+20]
    
    wavperpix = np.median(np.diff(region)) # resolution in wavelength / pix
    i = np.argmin(abs(lamred-specres['wave']))
    res = specres['fwhm'][i] # resolution in wavelength
    fwhm = res/wavperpix # resolution in pix
    
    return fwhm


def compute_EW(wave,flux,error,theta,plot,show=False):
    transinfo = transitions()
    wav,flx,err = continuum_normalize(flux,wave,error)
    lamred = theta[0] * u.AA
    lamblue = (lamred.value - (transinfo['lamred0'] - transinfo['lamblu0'])) * u.AA
    c = 2.998e5 * u.km/u.s
    bD = theta[2] * u.km / u.s
    
    sigma = 2.5 * (lamred * bD / c).to(u.AA).value
    
    regionr = lamred.value - sigma, lamred.value + sigma
    regionb = lamblue.value - sigma, lamblue.value + sigma
    
    indb = np.where((wav>regionb[0]) & (wav<regionb[1]))[0]
    indr = np.where((wav>regionr[0]) & (wav<regionr[1]))[0]
    indcomb = np.where((wav>regionr[0]) & (wav<regionb[1]))[0]
    
    dlambb = np.insert(np.diff(wav[indb]), 0, np.diff(wav[indb])[0])
    dlambr = np.insert(np.diff(wav[indr]), 0, np.diff(wav[indr])[0])
    
    EWb = np.sum( (np.ones(len(indb)) - flx[indb]) * dlambb )
    EWr = np.sum( (np.ones(len(indr)) - flx[indr])  * dlambr )
    
    sig_EW_l = np.sum( (dlambb * err[indb]) )
    sig_EW_r = np.sum( (dlambr * err[indr]) )
    
    if len(indcomb)>1:
        dlambcomb = np.insert(np.diff(wav[indcomb]), 0, np.diff(wav[indcomb])[0])
        comb = np.sum( (np.ones(len(indcomb)) - flx[indcomb]) * dlambcomb )
        sigcomb = np.sum( (dlambcomb * err[indcomb]) )
        
        EWb -= comb/2
        EWr -= comb/2
        sig_EW_l -= sigcomb
        sig_EW_r -= sigcomb
        
        
    if plot:
        fig, ax = plt.subplots()
        ax.plot(wav,flx,'k',drawstyle='steps-mid')
        ax.plot(wav[indb],flx[indb],'dimgray', drawstyle='steps-mid')
        ax.plot(wav[indr],flx[indr],'dimgray', drawstyle='steps-mid')
        ax.plot(wav[indb],np.ones(len(indb)),'dimgray',lw=1.2,solid_capstyle='projecting')
        ax.plot(wav[indr],np.ones(len(indr)),'dimgray',lw=1.2,solid_capstyle='projecting')
        ax.fill_between(wav[indb],np.ones(len(indb)),flx[indb],color='skyblue',alpha=0.5)
        ax.fill_between(wav[indr],np.ones(len(indr)),flx[indr],color='lightcoral',alpha=0.5)
        
        if len(indcomb>0):
            ax.fill_between(wav[indcomb],np.ones(len(indcomb)),flx[indcomb],color='dimgrey',alpha=0.75)
        
        ax.set_ylim(flx.min()-.5,flx.max()+.5)
        ax.vlines([lamblue.value,lamred.value],ymin=flx.min()-.5,ymax=flx.max()+.5,linestyles='dotted',colors='k',linewidths=1)
        ax.set_xlabel('Wavelength ($\mathrm{\AA}$)')
        ax.set_ylabel('Normalized Flux')
        
        text1 = f'{EWb:.2f}'
        text2 = f'{EWr:.2f}'
        text3 = f'{sigma:.2f}'
        ax.text(0.15,0.925,'$\mathrm{EW_{2796}} = $'+text1+' $\mathrm{\AA}$',fontsize='medium',transform=ax.transAxes)
        ax.text(0.15,0.875,'$\mathrm{EW_{2803}} = $'+text2+' $\mathrm{\AA}$',fontsize='medium',transform=ax.transAxes)
        #ax.text(0.15,0.825,'$\sigma_{\lambda} = $'+text3+' $\mathrm{\AA}$',fontsize='medium',transform=ax.transAxes)
        plt.savefig(plot,bbox_inches='tight',dpi=350)
        if show:
            plt.show()
        else:
            plt.close()
    
    return EWb, sig_EW_l, EWr, sig_EW_r