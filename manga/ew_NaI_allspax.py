
from __future__ import print_function

import math
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
import model_NaI
import continuum_normalize_NaI
import NaImcmc_read_fits
import pdb
from linetools.guis import xspecgui as ltxsg
import imp; imp.reload(ltxsg)
from linetools.spectra.xspectrum1d import XSpectrum1D

# Computes EW at NaI in each spaxel
# Reports median value for stellar pop model and data
# Inputs below refer to the whole cube!
#def ew(wave, flux, ivar, smod, stkinfit, blim, rlim):

def ew(fits_dap_map, fits_dap_cube, fits_drp, blim, rlim, FIT_FLG=None):

    sol = 2.998e5    # km/s

    cube = NaImcmc_read_fits.NaImcmc_read_fits(fits_dap_map, fits_dap_cube, fits_drp)

    wave = cube['wave']
    flux = cube['fluxb']
    ivar = cube['ivarb']
    smod = cube['modelb']
    cz = cube['cz']
    stellar_vfield = cube['stellar_vfieldb']
    binid = cube['binidb']
    
    nbins = flux.shape[1]
    uniq_el, uniq_ind = np.unique(binid, return_index=True)
    uniq_el = uniq_el[1:]
    uniq_ind = uniq_ind[1:]
    
    allobsew = []
    allsigobsew = []
    allmodew = []
    alls2n = []
    allbinid = []
    
    for qq in uniq_ind:

        flux_bin = flux[:,qq]
        ivar_bin = ivar[:,qq]
        err_bin = 1.0/np.sqrt(ivar_bin)
        smod_bin = smod[:,qq]

        #if(qq==3741):
        #pdb.set_trace()
        if(np.ma.is_masked(stellar_vfield[qq])):

            continue

        else:
            
            # Determine bin redshift: cz in km/s = tstellar_kin[*,0]
            #bin_z = (cz + stellar_vfield[qq]) / sol
            cosmo_z = cz / sol
            bin_z = cosmo_z + ((1 + cosmo_z) * stellar_vfield[qq] / sol) 
            
            restwave = wave / (1.0 + bin_z)

            #pl.step(restwave, flux_bin)
            #pl.plot(restwave, smod_bin, color='r')
            #pl.axis([5850, 5930, 0, 2])
            #transinfo = model_NaI.transitions()
            #pl.plot(transinfo['lamred0'] + np.zeros(10), np.arange(10))
            #pl.plot(transinfo['lamblu0'] + np.zeros(10), np.arange(10))
            #pl.show()
            #pdb.set_trace()
      

            if((FIT_FLG==0) | (FIT_FLG==1)):
                ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, blim, rlim, FIT_FLG=0)

            elif(FIT_FLG==2):
                ndata = continuum_normalize_NaI.smod_norm(restwave, flux_bin, err_bin, smod_bin)
                
            restwave_NaI = ndata['nwave']
            flux_NaI = ndata['nflux']
            err_NaI = ndata['nerr']
            smod_NaI = smod_bin / ndata['cont']

            transinfo = model_NaI.transitions()
            vel_NaI = (restwave_NaI - transinfo['lamred0']) * sol / transinfo['lamred0']

            ind = np.where((vel_NaI > -600.0 ) & (vel_NaI < 300.0))
            restwave_NaI_forEW = restwave_NaI[ind]
            flux_NaI_forEW = flux_NaI[ind]
            err_NaI_forEW = err_NaI[ind]
            smod_NaI_forEW = smod_NaI[ind]
            dwv = restwave_NaI_forEW - np.roll(restwave_NaI_forEW,1)
            
            dwv[0] = dwv[1]
            obsEW = np.sum((1.0-flux_NaI_forEW) * dwv)
            sigobsEW = (np.sum((err_NaI_forEW * dwv)**2))**0.5
            modEW = np.sum((1.0-smod_NaI_forEW) * dwv)

            
            
            
            allbinid.append(binid[qq])
            allobsew.append(obsEW)
            allsigobsew.append(sigobsEW)
            alls2n.append(ndata['s2n'])
            allmodew.append(modEW)

    return {'binid':allbinid, 'obsew':allobsew, 'sigobsew':allsigobsew, 'modew':allmodew, 's2n':alls2n}
