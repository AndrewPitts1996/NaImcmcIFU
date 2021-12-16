from __future__ import print_function

import math
import numpy as np
import os
import sys
import fnmatch
import scipy.optimize as op
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from linetools.spectra.xspectrum1d import XSpectrum1D
import model_NaI
import model_fitter
import continuum_normalize_NaI
#import NaImcmc_read_fits
#import NaImcmc_fitwrapper
#import ew_NaI_allspax
import pdb


def setup_script():

    path = '/data/home/krubin/Projects/MUSE/gistwork/MAD/workingdir/results/'
    outdir = '/data/home/krubin/Projects/MUSE/NaImcmc/plan_files/'
    
    
    gal = 'NGC4030'
    redshift = 0.004887

    binsperrun = 100
    outfil = outdir + gal + '_script'

    # For continuum-normalization around NaI
    blim = [5850.0,5870.0]
    rlim = [5910.0,5930.0]
    fitlim = [5880.0,5910.0]
    sol = 2.998e5


    table_fil = path + gal + '_Testv1.1.SN60/' + gal + '_table.fits'
    ppxfmap_fil = path + gal + '_Testv1.1.SN60/' + gal + '_ppxf.fits'
    vorspec_fil = path + gal + '_Testv1.1.SN60/' + gal + '_VorSpectra.fits'
    ppxffit_fil = path + gal + '_Testv1.1.SN60/' + gal + '_ppxf-bestfit.fits'


    # Read in table file
    #table_hdu = fits.open(table_fil)[1].data
    #_, idxConvertShortToLong = np.unique(np.abs(table.BIN_ID),return_inverse=True)

    hdu = fits.open(ppxfmap_fil)
    binid = np.array(hdu[1].data.BIN_ID)
    ppxf_v = np.array(hdu[1].data.V)


    # Read in binned spectra
    hdu_vorspec = fits.open(vorspec_fil)
    hdr_vorspec = hdu_vorspec[0].header
    spec = hdu_vorspec[1].data.SPEC
    espec = np.sqrt(hdu_vorspec[1].data.ESPEC)

    nbins = spec.shape[0]
    npix = spec.shape[1]
    logwave = np.array(hdu_vorspec[2].data.LOGLAM)
    wave_air = np.exp(logwave)  # this is wavelength relative to z
    obswave_air = (1.0+redshift)*wave_air
    # convert to vacuum
    xspec = XSpectrum1D.from_tuple((obswave_air, 0.0*obswave_air))
    xspec.meta['airvac'] = 'air'
    xspec.airtovac()
    obswave = xspec.wavelength.value
    wave = obswave / (1.0+redshift)

    hdu_ppxfspec = fits.open(ppxffit_fil)
    mod = hdu_ppxfspec[1].data.BESTFIT

    # Read in PPXF map
    #ppxfHDU = fits.open(ppxfmap_fil)[1].data
    #ppxf = np.array( [ppxfHDU.V, ppxfHDU.SIGMA, ppxfHDU.H3, ppxfHDU.H4, ppxfHDU.LAMBDA_R] ).T
    #median_V_stellar = np.nanmedian( ppxf[:,0] )
    #ppxf[:,0] = ppxf[:,0] - median_V_stellar
    #binid = np.array( [ppxfHDU.BIN_ID] ).T
    #binid = binid.flatten()

    # Need LSF in km/s
    # This gives LSF in Ang
    # CAUTION: this does not convert air wavelengths
    # to vacuum, or account for velocity offset of each bin
    LSFdir = '/data/home/krubin/Projects/MUSE/gistwork/MAD/workingdir/configFiles/'
    LSFfil = LSFdir + 'LSF-Config_MUSE_WFM'
    configLSF = np.genfromtxt(LSFfil, comments='#')
    configLSF_wv = configLSF[:,0]
    configLSF_res = configLSF[:,1]
    configLSF_restwv = configLSF_wv / (1.0+redshift)
    whLSF = np.where((configLSF_restwv > fitlim[0]) & (configLSF_restwv < fitlim[1]))
    median_LSFAng = np.median(configLSF_res[whLSF[0]])
    median_LSFvel = sol * median_LSFAng / np.median(configLSF_wv[whLSF[0]])

    LSFvel_str = "{:.2f}".format(median_LSFvel)
    redshift_str = "{:.6f}".format(redshift)
    
    f = open(outfil, "w")
    f.write("#!/bin/sh\n")
    
    # Number of separate "runs"
    nruns = int(nbins / binsperrun)
    for nn in range(nruns+1):

        jobname = 'NaImcmc'+str(nn)
        startbinid = nn*binsperrun
        endbinid = (nn+1)*binsperrun

        if(endbinid > nbins):
            endbinid = nbins

        print(startbinid, endbinid)
        f.write('screen -mdS '+jobname+' sh -c "python NaImcmc_MUSE_analysis.py 1 '\
                +gal+' '+redshift_str+' '+LSFvel_str+' '+\
                str(startbinid)+' '+str(endbinid)+'"\n')
        
    f.close()
    
    # Set up script that lists
    # input root, redshift, LSFvel, startbinid, endbinid
    pdb.set_trace()

def run_mcmc(galname, redshift, LSFvel, startbinid, endbinid):

    path = '/data/home/krubin/Projects/MUSE/gistwork/MAD/workingdir/results/'
    outdir = '/data/home/krubin/Projects/MUSE/NaImcmc/MCMC/'
    outfits = outdir+galname+'-binid-'+str(startbinid)+'-'+\
        str(endbinid)+'-samples.fits'
    outpdf = outdir+galname+'-binid-'+str(startbinid)+'-'+\
        str(endbinid)+'-line_triangle.pdf'
    
    # For continuum-normalization around NaI
    blim = [5850.0,5870.0]
    rlim = [5910.0,5930.0]
    fitlim = [5880.0,5910.0]
    sol = 2.998e5

    ppxfmap_fil = path + galname + '_Testv1.1.SN60/' + galname + '_ppxf.fits'
    vorspec_fil = path + galname + '_Testv1.1.SN60/' + galname + '_VorSpectra.fits'
    ppxffit_fil = path + galname + '_Testv1.1.SN60/' + galname + '_ppxf-bestfit.fits'

    hdu = fits.open(ppxfmap_fil)
    binid = np.array(hdu[1].data.BIN_ID)
    ppxf_v = np.array(hdu[1].data.V)

    # Read in binned spectra
    hdu_vorspec = fits.open(vorspec_fil)
    hdr_vorspec = hdu_vorspec[0].header
    spec = hdu_vorspec[1].data.SPEC
    espec = np.sqrt(hdu_vorspec[1].data.ESPEC)

    nbins = spec.shape[0]
    npix = spec.shape[1]
    logwave = np.array(hdu_vorspec[2].data.LOGLAM)
    wave_air = np.exp(logwave)   # this is wavelength relative to z
    obswave_air = (1.0+redshift)*wave_air
    # convert to vacuum
    xspec = XSpectrum1D.from_tuple((obswave_air, 0.0*obswave_air))
    xspec.meta['airvac'] = 'air'
    xspec.airtovac()
    obswave = xspec.wavelength.value
    wave = obswave / (1.0+redshift)

    hdu_ppxfspec = fits.open(ppxffit_fil)
    mod = hdu_ppxfspec[1].data.BESTFIT

    sv_samples = []
    sv_binnumber = []
    sv_percentiles = []

    # Set up array with all relevant binids
    fitbins = np.arange(startbinid,endbinid)

    for qq in fitbins:

        
        ind = binid==qq

        binvel = ppxf_v[ind]
        flux_bin = np.ma.array(spec[ind,:].flatten())
        err_bin = np.ma.array(espec[ind,:].flatten())
        mod_bin = np.ma.array(mod[ind,:].flatten())

        restwave = wave / ((binvel/sol) + 1.0)
        
        ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, blim, rlim, FIT_FLG=0)

        print("""Beginning fit for bin {0} """.format(qq))
        
        # Cut out NaI
        select = np.where((ndata['nwave'] > fitlim[0]) & (ndata['nwave'] < fitlim[1]))
        restwave_NaI = ndata['nwave'][select]
        flux_NaI = ndata['nflux'][select]
        err_NaI = ndata['nerr'][select]
        sres_NaI = LSFvel

        data = {'wave':restwave_NaI, 'flux':flux_NaI, 'err':err_NaI, 'velres':sres_NaI}
        #pdb.set_trace()

        # Guess good model parameters
        logN = 13.0
        bD = 20.0
        Cf = 0.5
        lamred = 5897.5581
        theta_guess = lamred, logN, bD, Cf
        guess_mod = model_NaI.model_NaI(theta_guess, data['velres'], data['wave'])
        datfit = model_fitter.model_fitter(data, theta_guess)

        # Run the MCMC
        datfit.mcmc()
        lamred_mcmc, logN_mcmc, bD_mcmc, Cf_mcmc = datfit.theta_percentiles
        transinfo = model_NaI.transitions()

        sv_binnumber.append(binid[ind])
        sv_samples.append(datfit.samples)
        sv_percentiles.append(datfit.theta_percentiles)

    t = Table([sv_binnumber, sv_samples, sv_percentiles], \
              names=('bin', 'samples', 'percentiles'))
    fits.writeto(outfits, np.array(t), overwrite=True)
        
def main():

    flg = int(sys.argv[1])
    

    if (flg==0):
        setup_script()

    if (flg==1):

        #pdb.set_trace()
        gal = sys.argv[2]
        redshift = float(sys.argv[3])
        LSFvel = float(sys.argv[4])
        startbin = int(sys.argv[5])
        endbin = int(sys.argv[6])

        run_mcmc(galname=gal, redshift=redshift, LSFvel=LSFvel, \
                 startbinid=startbin, endbinid=endbin)
    
main()
