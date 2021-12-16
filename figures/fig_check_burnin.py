import numpy as np
import scipy.special as sp
import math
import sys
import os
import fnmatch
import os.path
from linetools.spectra.xspectrum1d import XSpectrum1D
from linetools.spectra import convolve as lsc
from astropy import units as u
from astropy.units import Quantity
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn
from astropy.io import ascii
import NaImcmc_read_fits

import json
import pdb

import matplotlib.pyplot as pl
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
pl.rcParams['font.family'] = 'stixgeneral'
pl.rcParams['mathtext.fontset'] = 'stix'
from matplotlib.ticker import MaxNLocator

import model_NaI
import model_fitter
import continuum_normalize_NaI
import corner

def fig_check_burnin(plate=None, ifu=None, binid_for_fit=None):


    outdir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/MCMC/SQUARE2.0/2020jul20/Testing_FluxFit'

    outpattern = '-logN14p4_16p0-mcmc-linetime.pdf'
    outfil = outdir + '/'+str(plate)+'-'+str(ifu)+'-'+str(binid_for_fit)+outpattern
    outfits = outdir + '/'+str(plate)+'-'+str(ifu)+'-'+str(binid_for_fit)+'-logN14p4_16p0-samples.fits'
    
    saspath = '/data/manga/sas/mangawork/'
    drppath = saspath+'manga/spectro/redux/MPL-9/'
    dappath = saspath+'users/u6021943/Alt-DAP/SQUARE2.0_2020jan09/SQUARE2.0-MILESHC-MASTARHC/'

    fits_drp = drppath+str(plate)+'/stack/manga-'+str(plate)+'-'+str(ifu)+\
        '-LOGCUBE.fits.gz'
    fits_dap_cube = dappath+str(plate)+'/'+str(ifu)+'/manga-'+str(plate)+\
        '-'+str(ifu)+'-LOGCUBE-SQUARE2.0-MILESHC-MASTARHC.fits.gz'
    fits_dap_map = dappath+str(plate)+'/'+str(ifu)+'/manga-'+str(plate)+\
        '-'+str(ifu)+'-MAPS-SQUARE2.0-MILESHC-MASTARHC.fits.gz'

    if (os.path.isfile(outfil)==False):

    
        # For continuum-normalization around NaI
        blim = [5850.0,5870.0]
        rlim = [5910.0,5930.0]
        fitlim = [5880.0,5910.0]

        transinfo = model_NaI.transitions()
        sol = 2.998e5

        # Read in the data
        cube = NaImcmc_read_fits.NaImcmc_read_fits(fits_dap_map, fits_dap_cube,\
                                                   fits_drp=fits_drp)

        wave = cube['wave']
        #sres = cube['sres']
        flux = cube['fluxb']
        ivar = cube['ivarb']
        smod = cube['modelb']
        predisp = cube['predispb']
        
        cz = cube['cz']
        stellar_vfield = cube['stellar_vfieldb']
        binid = cube['binidb']
        bb = np.where(binid==binid_for_fit)
        bb = bb[0][0]

        flux_bin = flux[:,bb]
        ivar_bin = ivar[:,bb]
        err_bin = 1.0/np.sqrt(ivar_bin)
        smod_bin = smod[:,bb]
        predisp_bin = predisp[:,bb]
        #bin_z = (cz + stellar_vfield[bb]) / sol
        cosmo_z = cz / sol
        bin_z = cosmo_z + ((1 + cosmo_z) * stellar_vfield[bb] / sol) 
    
        restwave = wave / (1.0 + bin_z)
        ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, \
                                             blim, rlim, FIT_FLG=0)
    

        # Cut out NaI
        ind = np.where((ndata['nwave'] > fitlim[0]) & (ndata['nwave'] < fitlim[1]))
        restwave_NaI = ndata['nwave'][ind]
        flux_NaI = ndata['nflux'][ind]   
        err_NaI = ndata['nerr'][ind]
        #sres_NaI = sres[ind]
        predisp_NaI = predisp_bin[ind]
        wave_NaI = wave[ind]
        #avg_res = sol/np.mean(sres_NaI)
        avg_res = sol * 2.355 * np.mean(predisp_NaI)/np.mean(wave_NaI)
        data = {'wave':restwave_NaI, 'flux':flux_NaI, 'err':err_NaI, 'velres':avg_res}

        
        # Guess good model parameters
        logN = 14.5
        bD = 100.0
        Cf = 0.5
        lamred = 5897.5581
        theta_guess = lamred, logN, bD, Cf
        
        datfit = model_fitter.model_fitter(data, theta_guess, linetimefil=outfil)
        datfit.mcmc()
        lamred_mcmc, logN_mcmc, bD_mcmc, Cf_mcmc = datfit.theta_percentiles

        sv_samples = []
        sv_binnumber = []
        sv_percentiles = []

        sv_binnumber.append(binid_for_fit)
        sv_samples.append(datfit.samples)
        sv_percentiles.append(datfit.theta_percentiles)
        
        t = Table([sv_binnumber, sv_samples, sv_percentiles], names=('bin', 'samples', 'percentiles'))
        fits.writeto(outfits, np.array(t), overwrite=True)


def fig_check_fitandtriangle(plate=None, ifu=None, binid_for_fit=None):


    indir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/MCMC/SQUARE2.0/2020jul20/Testing_FluxFit'

    samplefits = indir + '/'+str(plate)+'-'+str(ifu)+'-'+str(binid_for_fit)+'-logN14p4_16p0-samples.fits'
    
    outtrifil = indir + '/'+str(plate)+'-'+str(ifu)+'-'+str(binid_for_fit)+'-triangle.pdf'
    outfitfil = indir + '/'+str(plate)+'-'+str(ifu)+'-'+str(binid_for_fit)+'-NaImcmcbestfit.pdf'
    
    saspath = '/data/manga/sas/mangawork/'
    drppath = saspath+'manga/spectro/redux/MPL-9/'
    dappath = saspath+'users/u6021943/Alt-DAP/SQUARE2.0_2020jan09/SQUARE2.0-MILESHC-MASTARHC/'

    fits_drp = drppath+str(plate)+'/stack/manga-'+str(plate)+'-'+str(ifu)+\
        '-LOGCUBE.fits.gz'
    fits_dap_cube = dappath+str(plate)+'/'+str(ifu)+'/manga-'+str(plate)+\
        '-'+str(ifu)+'-LOGCUBE-SQUARE2.0-MILESHC-MASTARHC.fits.gz'
    fits_dap_map = dappath+str(plate)+'/'+str(ifu)+'/manga-'+str(plate)+\
        '-'+str(ifu)+'-MAPS-SQUARE2.0-MILESHC-MASTARHC.fits.gz'


    mcmc = fits.getdata(samplefits, 1)
    tmcmc = Table(mcmc)
    samples = tmcmc['samples'][0]
    percentiles_mcmc = tmcmc['percentiles'][0]
    lamred_samples = samples[:,0]
    lamred50 = np.percentile(lamred_samples,50)

    nbins = 30
    logN_samples = samples[:,1]
    logN50 = np.percentile(logN_samples,50)
    #histN, binsN = np.histogram(logN_samples, bins=nbins)
    #imx = np.argmax(histN)
    #logN_mode = (binsN[imx] + binsN[imx+1])/2.0
    #pdb.set_trace()
    
    bD_samples = samples[:,2]
    bD50 = np.percentile(bD_samples, 50)
    histbD, binsbD = np.histogram(bD_samples, bins=nbins)
    imx = np.argmax(histbD)
    bD_mode = (binsbD[imx] + binsbD[imx+1])/2.0

    # Use bD mode to determine logN parameter
    select = np.where((bD_samples > binsbD[imx]) & (bD_samples < binsbD[imx+1]))
    logN_mode = np.mean(logN_samples[select])
    #pdb.set_trace()
    
    Cf_samples = samples[:,3]
    Cf50 = np.percentile(Cf_samples, 50)
    histCf, binsCf = np.histogram(Cf_samples, bins=nbins)
    imx = np.argmax(histCf)
    Cf_mode = (binsCf[imx] + binsCf[imx+1])/2.0
    
    #best_fit = lamred50, logN50, bD50, Cf50
    best_fit = lamred50, logN_mode, bD_mode, Cf_mode
    
    #pdb.set_trace()
    pl.figure(figsize=(6.0,5.0))
    pl.rcParams.update({'font.size': 14})                  
    fig = corner.corner(samples, labels=["$lamred$", "$logN$", "$bD$", "$Cf$"],
                        truths=best_fit)
    fig.tight_layout()
    fig.savefig(outtrifil, format='pdf')

    
    # For continuum-normalization around NaI
    blim = [5850.0,5870.0]
    rlim = [5910.0,5930.0]
    fitlim = [5880.0,5910.0]
    inspectlim = [5850.,5930.]

    transinfo = model_NaI.transitions()
    sol = 2.998e5

    # Read in the data
    cube = NaImcmc_read_fits.NaImcmc_read_fits(fits_dap_map, fits_dap_cube,\
                                               fits_drp=fits_drp)

    wave = cube['wave']
    #sres = cube['sres']
    flux = cube['fluxb']
    ivar = cube['ivarb']
    smod = cube['modelb']
    predisp = cube['predispb']
        
    cz = cube['cz']
    stellar_vfield = cube['stellar_vfieldb']
    binid = cube['binidb']
    bb = np.where(binid==binid_for_fit)
    bb = bb[0][0]

    flux_bin = flux[:,bb]
    ivar_bin = ivar[:,bb]
    err_bin = 1.0/np.sqrt(ivar_bin)
    smod_bin = smod[:,bb]
    predisp_bin = predisp[:,bb]
    #bin_z = (cz + stellar_vfield[bb]) / sol
    cosmo_z = cz / sol
    bin_z = cosmo_z + ((1 + cosmo_z) * stellar_vfield[bb] / sol) 
    
    restwave = wave / (1.0 + bin_z)
    ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, \
                                         blim, rlim, FIT_FLG=0)
    

    # Cut out NaI
    naiind = np.where((restwave > inspectlim[0]) & (restwave < inspectlim[1]))
    restwave_NaI = restwave[naiind]
    flux_NaI = flux_bin[naiind]
    err_NaI = err_bin[naiind]
    #smod_NaI = nsmod['nflux'][ind]
    #smod_norm = smod_bin / ndata['cont']
    #sres_NaI = sres[naiind]
    predisp_NaI = predisp_bin[naiind]
    wave_NaI = wave[naiind]
    smod_NaI = smod_bin[naiind]
    cont_NaI = ndata['cont'][naiind]
    
    avg_cont = np.mean(ndata['cont'][naiind])
    avg_res = sol * 2.355 * np.mean(predisp_NaI)/np.mean(wave_NaI)

    #pdb.set_trace()
    pl.figure(figsize=(6.0,5.0))
    pl.rcParams.update({'font.size': 14})

    fig, ax = pl.subplots(1,1)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.0)

    fitind = np.where((restwave_NaI > fitlim[0]) & (restwave_NaI < fitlim[1]))
    best_mod = model_NaI.model_NaI(best_fit, avg_res, restwave_NaI[fitind])

    ax.plot(restwave_NaI,flux_NaI,drawstyle='steps-mid', color="k")
    ax.plot(restwave_NaI,flux_NaI+err_NaI,drawstyle='steps-mid', color="gray", alpha=0.5)
    ax.plot(restwave_NaI,flux_NaI-err_NaI,drawstyle='steps-mid', color="gray", alpha=0.5)
    ax.plot(restwave_NaI,smod_NaI, color="red", drawstyle='steps-mid')
    ax.plot(restwave_NaI,cont_NaI, color='cyan')
    ax.plot(best_mod['modwv'], cont_NaI[fitind] * best_mod['modflx'], color="blue")
                        
    ax.set_xlabel(r'Rest Wavelength (Ang)')
    ax.set_ylabel(r'Flux')
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(pl.MultipleLocator(5.0))
    #ax.set_xlim(0.5,1.2)
                    
    ylim = ax.get_ylim()
    ax.plot([transinfo['lamblu0'],transinfo['lamblu0']], [ylim[0], ylim[1]], color="gray", ls='--')
    ax.plot([transinfo['lamred0'],transinfo['lamred0']], [ylim[0], ylim[1]], color="gray", ls='--')

    text_str = "{:.2f}".format(best_fit[0])+', '+"{:.2f}".format(best_fit[1])+', '+\
        "{:.2f}".format(best_fit[2])+', '+"{:.2f}".format(best_fit[3])
    ax.text(0.05, 0.11, text_str, ha='left', va='center', transform=ax.transAxes, fontsize=11)
              
    fig.tight_layout()
    fig.savefig(outfitfil, format='pdf')
    
        
def run_check_burnin(infits=None):

    infits = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/NaImcmcfake_outflowbins.fits'
    fakebins = fits.getdata(infits)

    ## TESTING WITH CHAIN OF 1000
    #fig_check_burnin(plate=8250, ifu=12704, binid_for_fit=8)
    #fig_check_burnin(plate=8241, ifu=3704, binid_for_fit=0)
    #fig_check_burnin(plate=8311, ifu=6104, binid_for_fit=0)

    #fig_check_burnin(plate=8311, ifu=6104, binid_for_fit=23)
    #fig_check_burnin(plate=8588, ifu=6101, binid_for_fit=57)
    #fig_check_burnin(plate=8595, ifu=3703, binid_for_fit=19)
    #fig_check_burnin(plate=8595, ifu=3703, binid_for_fit=20)
    #fig_check_burnin(plate=8595, ifu=3703, binid_for_fit=4)

    ## NEXT, TEST THESE, PLUS the following with higher bD limit
    #fig_check_burnin(plate=8250, ifu=12704, binid_for_fit=1)
    
    for qq in range(len(fakebins)):

        #fig_check_burnin(plate=int(fakebins['PLATE'][qq]), ifu=int(fakebins['IFU'][qq]), \
        #                 binid_for_fit=int(fakebins['BINID'][qq]))
        fig_check_fitandtriangle(plate=int(fakebins['PLATE'][qq]), ifu=int(fakebins['IFU'][qq]), \
                                 binid_for_fit=int(fakebins['BINID'][qq]))

        
def main():

    run_check_burnin(infits=None)

main()
