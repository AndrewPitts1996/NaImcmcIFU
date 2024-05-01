

from __future__ import print_function

import math
import numpy as np
import scipy.optimize as op
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn
from astropy.io import ascii
import corner
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
pl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages
import NaImcmc_read_fits
import ew_NaI_allspax
import model_NaI
import model_fitter
import continuum_normalize_NaI
import logging
import time
import pdb
from IPython import embed

def NaImcmc_fitwrapper(plateifu, wave, prespecres, flux, ivar, smod, predisp, cz, stellar_vfield, binid,
                       blim, rlim, fitlim, outdir=None, FIT_FLG=None):

    # Start timer
    start_time = time.clock()
    
    if(outdir==None):
        outdir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/MCMC/SQUARE2.0/'

    outfits = outdir+plateifu+"-samples.fits"
    outpdf = outdir+plateifu+"-line-triangle.pdf"
    logfil = outdir+plateifu+".log"
    sol = 2.998e5    # km/s
    nbins = flux.shape[1]
    uniq_el, uniq_ind = np.unique(binid, return_index=True)
    uniq_el = uniq_el[1:]
    uniq_ind = uniq_ind[1:]


    sv_samples = []
    sv_binnumber = []
    sv_percentiles = []

        
    #with PdfPages(outpdf) as pdf:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(logfil)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    #logging.basicConfig(filename=logfil, filemode='w', level=10)
    logger.addHandler(file_handler)
    
    logger.info('Beginning fit for {0}\n'.format(plateifu))
    
    print("""Beginning fit for {0}""".format(plateifu))
    
    for qq in uniq_ind:
        
        flux_bin = flux[:,qq]
        ivar_bin = ivar[:,qq]
        err_bin = 1.0/np.sqrt(ivar_bin)
        smod_bin = smod[:,qq]
        predisp_bin = predisp[:,qq]

        # Determine bin redshift: cz in km/s = tstellar_kin[*,0]
        #bin_z = (cz + stellar_vfield[qq]) / sol
        cosmo_z = cz / sol
        bin_z = cosmo_z + ((1 + cosmo_z) * stellar_vfield[qq] / sol) 

        restwave = wave / (1.0 + bin_z)
        logger.info('Fitting bin ID {0} with index {1} of {2}\n'.format(binid[qq], qq, len(uniq_el)))
        print("""Fitting bin ID {0} with index {1} of {2}""".format(binid[qq], qq, len(uniq_el)))

        if(FIT_FLG==0):
            ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, blim, rlim, FIT_FLG=0)
        elif(FIT_FLG==1):
            ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, blim, rlim, FIT_FLG=1, smod=smod_bin)
        elif(FIT_FLG==2):
            ndata = continuum_normalize_NaI.smod_norm(restwave, flux_bin, err_bin, smod_bin, blim, rlim)

   
        if((np.ma.is_masked(bin_z)==True) | (ndata['mskflg']==1)):

            if(np.ma.is_masked(bin_z)==True):
                logger.info('Stellar redshift masked for this bin.')
            elif(ndata['mskflg']==1):
                logger.info('No good continuum regions for this bin.')

        else:
            # Cut out NaI
            ind = np.where((ndata['nwave'] > fitlim[0]) & (ndata['nwave'] < fitlim[1]))
            restwave_NaI = ndata['nwave'][ind]
            flux_NaI = ndata['nflux'][ind]
            err_NaI = ndata['nerr'][ind]
            #sres_NaI = sres[ind]
            predisp_NaI = predisp_bin[ind]  # This is LSF sigma in Angstroms, need to convert to velocity FWHM
            wave_NaI = wave[ind]
            #avg_res = sol/np.mean(sres_NaI)
            avg_res = sol * 2.355 * np.mean(predisp_NaI)/np.mean(wave_NaI)

            print(np.mean(predisp_NaI), avg_res)
            #if(binid[qq]==28):
            #    embed()
            #    exit()

            # If predisp is NaN, punt and use prespecres
            #if(np.isnan(avg_res)):
            #
            #    prespecres_NaI = prespecres[ind]
            #    avg_res = sol/np.mean(prespecres_NaI)
                
            data = {'wave':restwave_NaI, 'flux':flux_NaI, 'err':err_NaI, 'velres':avg_res}

            # Guess good model parameters
            logN = 14.5
            bD = 100.0
            Cf = 0.5
            lamred = 5897.5581
            theta_guess = lamred, logN, bD, Cf
            guess_mod = model_NaI.model_NaI(theta_guess, data['velres'], data['wave'])
            datfit = model_fitter.model_fitter(data, theta_guess)
    
            # Run maximum likelihood fit
            #datfit.maxlikelihood()
            #lamred_ml, logN_ml, bD_ml, Cf_ml = datfit.theta_ml
            #print("""Maximum likelihood result:
            #lamred = {0} (guess: {1})
            #logN = {2} (guess: {3})
            #bD = {4} (guess: {5})
            #Cf = {6} (guess: {7})
            #""".format(lamred_ml, lamred, logN_ml, logN, bD_ml, bD, Cf_ml, Cf))


            # Run the MCMC
            datfit.mcmc()
            # Read in MCMC percentiles
            ## WORKING HERE
            lamred_mcmc, logN_mcmc, bD_mcmc, Cf_mcmc = datfit.theta_percentiles
            # Make triangle plot
            #fig = corner.corner(datfit.samples, labels=["$lamred$", "$logN$", "$bD$", "$Cf$"],
            #                    truths=[lamred_mcmc[0], logN_mcmc[0], bD_mcmc[0], Cf_mcmc[0]])
            #ax = fig.add_subplot(333)
            #ax.plot(restwave_NaI,flux_NaI,drawstyle='steps-mid')
            #ax.set_xlabel(r'Rest Wavelength (Ang)')
            #ax.set_ylabel(r'Normalized Flux')
            #ax.set_ylim(0.5,1.2)

            transinfo = model_NaI.transitions()
            #ax.plot([transinfo['lamblu0'],transinfo['lamblu0']], [0.0,2.0])
            #ax.plot([transinfo['lamred0'],transinfo['lamred0']], [0.0,2.0])
    
            # Print plate, ifu, and bin number
            #bin_str = "{:.0f}".format(binid[qq])
            #pltnote = plateifu+'   Bin:'+bin_str
            #ax.text(0.5,-0.3,pltnote,ha='center',va='center',transform = ax.transAxes,fontsize=20)


            sv_binnumber.append(binid[qq])
            sv_samples.append(datfit.samples)
            sv_percentiles.append(datfit.theta_percentiles)
        
        
            #pdf.savefig()
            #fig.savefig(plate_str+"-"+ifu_str+"-line-triangle.png")

            #pl.figure()
            #pl.plot(restwave_NaI,flux_NaI,drawstyle='steps-mid')
            #pl.plot(guess_mod['modwv'], guess_mod['modflx'], color="r",drawstyle='steps-mid')
            #pl.xlim(5870.,5920.)
            #pl.ylim(0.5,1.3)
            #pl.tight_layout()
            #pl.savefig("NaI-profile.png")
                    
    t = Table([sv_binnumber, sv_samples, sv_percentiles], names=('bin', 'samples', 'percentiles'))
    fits.writeto(outfits, np.array(t), overwrite=True)

    if time.clock() - start_time < 60:
        tstr = '  DURATION: {0:.5f} sec'.format((time.clock() - start_time))
    elif time.clock() - start_time < 3600:
        tstr = '  DURATION: {0:.5f} min'.format((time.clock() - start_time)/60.)
    else:
        tstr = '  DURATION: {0:.5f} hr'.format((time.clock() - start_time)/3600.)
    logger.info('Successfully wrote samples fits file for this object.')
    logger.info(tstr)

    logger.handlers[0].stream.close()
    logger.removeHandler(logger.handlers[0])
