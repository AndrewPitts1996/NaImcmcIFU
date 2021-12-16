from __future__ import print_function

import os
import sys
import math
import errno
import copy
import numpy as np
import fnmatch

import matplotlib
matplotlib.use('Agg')
from imp import reload
from astropy.io import fits
from astropy.table import Table
#from mangadap.util.fileio import channel_dictionary
import model_NaI
#import ew_NaI_allspax
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
#from marvin.tools.maps import Maps
#import marvin.utils.plot.map as mapplot
from plotbin.sauron_colormap import register_sauron_colormap
register_sauron_colormap()
import pdb



def NaImcmc_MUSE_write_fitinfo(outdir=None, overwrite=False):

    gistpath = '/data/home/krubin/Projects/MUSE/gistwork/MAD/workingdir/results/'
    mcmcpath = '/data/home/krubin/Projects/MUSE/NaImcmc/MCMC/'

    sol = 2.998e5    # km/s

    pattern = '*samples*.fits'
    outdirroot = mcmcpath
    outroot = '-NaImaps.fits'

       
    # Find all bins which have been fit
    allmcmc = []
    allsamplesfils = []

    for root, dirs, files in os.walk(mcmcpath):
        for filename in fnmatch.filter(files,pattern):

            #pdb.set_trace()
            allsamplesfils.append(filename)
            allmcmc.append(os.path.join(root,filename))

    print("Found this many MCMC fit outputs:")
    print(len(allmcmc))
    
    #if overwrite:
    #
    newmcmc = allmcmc
    #
    #else:
    
    #    newmcmc = []

    galname_fits = []
    startbinid_fits = []
    endbinid_fits = []
    for nn in range(len(allsamplesfils)):

        nnsamples = allsamplesfils[nn]
        nnfil = nnsamples.split('-')
        galname = nnfil[0]
        startbinid = int(nnfil[2])
        endbinid = int(nnfil[3])

        galname_fits.append(galname)
        startbinid_fits.append(startbinid)
        endbinid_fits.append(endbinid)

    galname_fits = np.array(galname_fits)
    startbinid_fits = np.array(startbinid_fits)
    endbinid_fits = np.array(endbinid_fits)
        
    #        mapname = galname+'-'+outroot
    #        dum = os.path.exists(outdirroot+mapname)
    #        if(dum):
    #            print("Not overwriting ", mapname)
    #        else:
    #            newmcmc.append(os.path.join(root,nnsamples))
    #
    #    print("Found this many new MCMC fit outputs to map:")
    #    print(len(newmcmc))
        

    # For continuum-normalization around NaI
    blim = [5850.0,5870.0]
    rlim = [5910.0,5930.0]
    fitlim = [5880.0,5910.0]
    
    # Identify unique galaxies that have been fit
    ugal = np.unique(galname_fits)
    
    for nn in range(len(ugal)):
  
        ## Set up file names
        table_fil = gistpath + ugal[nn] + '_Testv1.1.SN60/' + ugal[nn] + '_table.fits'
        ppxfmap_fil = gistpath + ugal[nn] + '_Testv1.1.SN60/' + ugal[nn] + '_ppxf.fits'
        vorspec_fil = gistpath + ugal[nn] + '_Testv1.1.SN60/' + ugal[nn] + '_VorSpectra.fits'
        ppxffit_fil = gistpath + ugal[nn] + '_Testv1.1.SN60/' + ugal[nn] + '_ppxf-bestfit.fits'
        hdu = fits.open(ppxfmap_fil)
        binid = np.array(hdu[1].data.BIN_ID)
        ppxf_v = np.array(hdu[1].data.V)

        #binid = np.unique( np.abs(np.array(binid_all)))
        #pdb.set_trace()
        
        dum_data = np.zeros(binid.shape)
        NaIvel50_map = np.zeros(binid.shape)
        NaIvel16_map = np.zeros(binid.shape)
        NaIvel84_map = np.zeros(binid.shape)
        NaIvel05_map = np.zeros(binid.shape)
        NaIvel95_map = np.zeros(binid.shape)
        NaIvel01_map = np.zeros(binid.shape)
        NaIvel99_map = np.zeros(binid.shape)
        
        NaIlogN50_map = np.zeros(binid.shape)
        NaIlogN16_map = np.zeros(binid.shape)
        NaIlogN84_map = np.zeros(binid.shape)
        NaIlogN05_map = np.zeros(binid.shape)
        NaIlogN95_map = np.zeros(binid.shape)
        NaIlogN01_map = np.zeros(binid.shape)
        NaIlogN99_map = np.zeros(binid.shape)

        NaICf50_map = np.zeros(binid.shape)
        NaICflo_map = np.zeros(binid.shape)
        NaICfhi_map = np.zeros(binid.shape)

        NaIbD50_map = np.zeros(binid.shape)
        NaIbDlo_map = np.zeros(binid.shape)
        NaIbDhi_map = np.zeros(binid.shape)

        #ewNaI_obs_map = np.zeros(binid.shape)
        #ewNaI_mod_map = np.zeros(binid.shape)
        #s2n_map = np.zeros(binid.shape)

        
        # Select samples files for this object
        wh = np.where(galname_fits==ugal[nn])
        outfits = mcmcpath + ugal[nn] + outroot
        
        ustartbin = startbinid_fits[wh]
        uendbin = endbinid_fits[wh]
        
        for mm in range(len(ustartbin)):

            ustartbin_str = str(ustartbin[mm])
            uendbin_str = str(uendbin[mm])
            mcmcfits = mcmcpath+ugal[nn]+"-binid-"+ustartbin_str+"-"+uendbin_str+"-samples.fits"
            mcmc = fits.getdata(mcmcfits, 1)
            tmcmc = Table(mcmc)
            nubins = len(tmcmc['bin'])
            
            print(mcmcfits, nubins)

            # Calculate velocities from fit
            transinfo = model_NaI.transitions()
            #vel50_bin = np.zeros(nubins) - 999.0
            #sigflow_bin = np.zeros(nubins) - 999.0

            for qq in range(nubins):

                binid_mcmc = tmcmc['bin'][qq]
                samples = tmcmc['samples'][qq]
                percentiles_mcmc = tmcmc['percentiles'][qq]
                lamred_mcmc = percentiles_mcmc[0,:]
                logN_mcmc = percentiles_mcmc[1,:]
                bD_mcmc = percentiles_mcmc[2,:]
                Cf_mcmc = percentiles_mcmc[3,:]
            

                # Calculate 95th-percentile limits
                lamred_samples = samples[:,0]
                lamred05 = np.percentile(lamred_samples,5)
                lamred95 = np.percentile(lamred_samples,95)
                lamred01 = np.percentile(lamred_samples,1)
                lamred99 = np.percentile(lamred_samples,99)

                logN_samples = samples[:,1]
                logN05 = np.percentile(logN_samples,5)
                logN95 = np.percentile(logN_samples,95)
                logN01 = np.percentile(logN_samples,1)
                logN99 = np.percentile(logN_samples,99)
                logN50 = logN_mcmc[0]
                logN16 = np.percentile(logN_samples,16)
                logN84 = np.percentile(logN_samples,84)

                Cf50 = Cf_mcmc[0]
                Cflo = Cf_mcmc[2]
                Cfhi = Cf_mcmc[1]

                bD50 = bD_mcmc[0]
                bDlo = bD_mcmc[2]
                bDhi = bD_mcmc[1]
            
                vel50 = (lamred_mcmc[0] - transinfo['lamred0']) * sol / transinfo['lamred0']
                vel16 = ((lamred_mcmc[0] - lamred_mcmc[2]) - transinfo['lamred0']) * sol / transinfo['lamred0']
                vel84 = (lamred_mcmc[0] + lamred_mcmc[1] - transinfo['lamred0']) * sol / transinfo['lamred0']
                vel05 = (lamred05 - transinfo['lamred0']) * sol / transinfo['lamred0']
                vel95 = (lamred95 - transinfo['lamred0']) * sol / transinfo['lamred0']
                vel01 = (lamred01 - transinfo['lamred0']) * sol / transinfo['lamred0']
                vel99 = (lamred99 - transinfo['lamred0']) * sol / transinfo['lamred0']

                ind = binid==binid_mcmc
                
                NaIvel50_map[ind] = vel50
                NaIvel16_map[ind] = vel16
                NaIvel84_map[ind] = vel84
                NaIvel05_map[ind] = vel05
                NaIvel95_map[ind] = vel95
                NaIvel01_map[ind] = vel01
                NaIvel99_map[ind] = vel99
                
                NaIlogN50_map[ind] = logN50
                NaIlogN16_map[ind] = logN16
                NaIlogN84_map[ind] = logN84
                NaIlogN05_map[ind] = logN05
                NaIlogN95_map[ind] = logN95
                NaIlogN01_map[ind] = logN01
                NaIlogN99_map[ind] = logN99
            
                NaICf50_map[ind] = Cf50
                NaICflo_map[ind] = Cflo
                NaICfhi_map[ind] = Cfhi
                
                NaIbD50_map[ind] = bD50
                NaIbDlo_map[ind] = bDlo
                NaIbDhi_map[ind] = bDhi

            
        # Write fits file!
        hdr = fits.Header()
        primary_hdu = fits.PrimaryHDU(header=hdr)
        nai_hdu = fits.HDUList()
        nai_hdu.append(fits.ImageHDU(binid, name='BINID'))
        nai_hdu.append(fits.ImageHDU(ppxf_v, name='PPXF_V'))

        nai_hdu.append(fits.ImageHDU(NaIvel01_map, name='NAIVEL01_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIvel05_map, name='NAIVEL05_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIvel16_map, name='NAIVEL16_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIvel50_map, name='NAIVEL50_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIvel84_map, name='NAIVEL84_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIvel95_map, name='NAIVEL95_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIvel99_map, name='NAIVEL99_MAP'))
        
        nai_hdu.append(fits.ImageHDU(NaIlogN50_map, name='NAILOGN50_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIlogN16_map, name='NAILOGN16_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIlogN84_map, name='NAILOGN84_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIlogN05_map, name='NAILOGN05_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIlogN95_map, name='NAILOGN95_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIlogN01_map, name='NAILOGN01_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIlogN99_map, name='NAILOGN99_MAP'))

        nai_hdu.append(fits.ImageHDU(NaICf50_map, name='NAICF50_MAP'))
        nai_hdu.append(fits.ImageHDU(NaICflo_map, name='NAICFLO_MAP'))
        nai_hdu.append(fits.ImageHDU(NaICfhi_map, name='NAICFHI_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIbD50_map, name='NAIBD50_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIbDlo_map, name='NAIBDLO_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIbDhi_map, name='NAIBDHI_MAP'))

        
        #nai_hdu.append(fits.ImageHDU(ewNaI_obs_map, name='EWNAI_OBS_MAP'))
        #nai_hdu.append(fits.ImageHDU(ewNaI_mod_map, name='EWNAI_MOD_MAP'))
        #nai_hdu.append(fits.ImageHDU(s2n_map, name='S2N_MAP'))
        
        nai_hdu.writeto(outfits, overwrite=True)
        # First, plot vel50
        # Then, adjust color for bins with central velocities outside of 95% limits

def main():

    NaImcmc_MUSE_write_fitinfo(outdir=None, overwrite=True)

main()
