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
from mangadap.util.fileio import channel_dictionary
import model_NaI
import ew_NaI_allspax
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from marvin.tools.maps import Maps
import marvin.utils.plot.map as mapplot
import pdb



def NaImcmc_write_fitinfo(outdir=None, overwrite=False, FIT_FLG=None):


    sol = 2.998e5    # km/s

    # Read in DAP file
    mcmcdirroot = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/MCMC/SQUARE2.0/2020mar01/'
    saspath = '/data/manga/sas/mangawork/'
    pattern = '*samples*.fits'
    outdirroot = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/MCMC/SQUARE2.0/2020mar01/'
    outroot = '-NaImaps.fits'

    if(FIT_FLG==0):
        mcmcdir = mcmcdirroot + 'FluxFit/'
        outdir = outdirroot + 'FluxFit/'
        #mcmcdir = mcmcdirroot + 'Extra/'
        #outfir = outdirroot + 'Extra/'
    elif(FIT_FLG==1):
        mcmcdir = mcmcdirroot + 'SModFit/'
        outdir = outdirroot + 'SModFit/'
    elif(FIT_FLG==2):
        mcmcdir = mcmcdirroot + 'FluxNoSModFit/'
        outdir = outdirroot + 'FluxNoSModFit/'
    
    # Find all objects which have been fit
    allmcmc = []
    allsamplesfils = []

    for root, dirs, files in os.walk(mcmcdir):
        for filename in fnmatch.filter(files,pattern):

            #pdb.set_trace()
            allsamplesfils.append(filename)
            allmcmc.append(os.path.join(root,filename))

    print("Found this many MCMC fit outputs:")
    print(len(allmcmc))

    if overwrite:

        newmcmc = allmcmc

    else:
    
        newmcmc = []
    
        for nn in range(len(allsamplesfils)):

            nnsamples = allsamplesfils[nn]
            nnfil = nnsamples.split('-')
            plate = nnfil[0]
            fiber = nnfil[1]
            mapname = plate+'-'+fiber+outroot
            dum = os.path.exists(outdir+mapname)
            if(dum):
                print("Not overwriting ", mapname)
            else:
                newmcmc.append(os.path.join(root,nnsamples))

        print("Found this many new MCMC fit outputs to map:")
        print(len(newmcmc))
        #pdb.set_trace()
        

    # For continuum-normalization around NaI
    blim = [5850.0,5870.0]
    rlim = [5910.0,5930.0]
    fitlim = [5880.0,5910.0]
    
    # Pixels are 69 km/s wide --> only mark flows > 35 km/s (half pixel)
    #vflow_min = 35.0

    for nn in range(len(newmcmc)):

        nnmcmc = newmcmc[nn]
        nnfil = os.path.basename(nnmcmc)
        print(nnfil)
        nnfilsplit = nnfil.split('-')
        plate = int(nnfilsplit[0])
        ifu = int(nnfilsplit[1])

        ## Set up file names
        plate_str = "{:.0f}".format(plate)
        ifu_str = "{:.0f}".format(ifu)

        dap = 'users/u6021943/Alt-DAP/SQUARE2.0_2020jan09/SQUARE2.0-MILESHC-MASTARHC/'+plate_str+'/'+ifu_str+'/manga-'+plate_str+'-'+ifu_str
        fits_dap_cube = saspath + dap + '-LOGCUBE-SQUARE2.0-MILESHC-MASTARHC.fits.gz'
        fits_dap_map = saspath + dap + '-MAPS-SQUARE2.0-MILESHC-MASTARHC.fits.gz'

        drp = 'manga/spectro/redux/MPL-9/'+plate_str+'/stack/manga-'+plate_str+'-'+ifu_str
        fits_drp = saspath + drp + '-LOGCUBE.fits.gz'
        
        ## Read in data
        hdu_maps = fits.open(fits_dap_map)
        hdu_hdr = hdu_maps['PRIMARY'].header
        cz = hdu_hdr['SCINPVEL']  # in km/s, z=v/c
        emlc = channel_dictionary(hdu_maps, 'EMLINE_GVEL')
        binid = hdu_maps['BINID'].data[1,:,:]
        binarea = hdu_maps['BIN_AREA'].data
        binsnr = hdu_maps['BIN_SNR'].data
        spx_ellcoo = hdu_maps['SPX_ELLCOO'].data
        bin_lwskycoo = hdu_maps['BIN_LWSKYCOO'].data
        bin_lwellcoo = hdu_maps['BIN_LWELLCOO'].data

        #mask_ext = hdu_maps['STELLAR_VEL'].header['QUALDATA']
        stellar_vfield = hdu_maps['STELLAR_VEL'].data
        stellar_vfield_mask = hdu_maps['STELLAR_VEL_MASK'].data
        stellar_vfield_ivar = hdu_maps['STELLAR_VEL_IVAR'].data
        #mask=hdu_maps[mask_ext].data>0)
        mask_ext = hdu_maps['EMLINE_GVEL'].header['QUALDATA']
        emission_vfield = hdu_maps['EMLINE_GVEL'].data[emlc['Ha-6564'],:,:]
        emission_vfield_mask = hdu_maps['EMLINE_GVEL_MASK'].data[emlc['Ha-6564'],:,:]
        emission_vfield_ivar = hdu_maps['EMLINE_GVEL_IVAR'].data[emlc['Ha-6564'],:,:]
        #mask=hdu_maps[mask_ext].data[emlc['Ha-6564'],:,:] > 0)

        #mask_ext = hdu_maps['EMLINE_GFLUX'].header['QUALDATA']
        Ha_gflux = hdu_maps['EMLINE_GFLUX'].data[emlc['Ha-6564'],:,:]
        Ha_gflux_mask = hdu_maps['EMLINE_GFLUX_MASK'].data[emlc['Ha-6564'],:,:]
        Ha_gflux_ivar = hdu_maps['EMLINE_GFLUX_IVAR'].data[emlc['Ha-6564'],:,:]
        Hb_gflux = hdu_maps['EMLINE_GFLUX'].data[emlc['Hb-4862'],:,:]
        Hb_gflux_mask = hdu_maps['EMLINE_GFLUX_MASK'].data[emlc['Hb-4862'],:,:]
        Hb_gflux_ivar = hdu_maps['EMLINE_GFLUX_IVAR'].data[emlc['Hb-4862'],:,:]
        OIII5008_gflux = hdu_maps['EMLINE_GFLUX'].data[emlc['OIII-5008'],:,:]
        OIII5008_gflux_mask = hdu_maps['EMLINE_GFLUX_MASK'].data[emlc['OIII-5008'],:,:]
        OIII5008_gflux_ivar = hdu_maps['EMLINE_GFLUX_IVAR'].data[emlc['OIII-5008'],:,:]
        NII6585_gflux = hdu_maps['EMLINE_GFLUX'].data[emlc['NII-6585'],:,:]
        NII6585_gflux_mask = hdu_maps['EMLINE_GFLUX_MASK'].data[emlc['NII-6585'],:,:]
        NII6585_gflux_ivar = hdu_maps['EMLINE_GFLUX_IVAR'].data[emlc['NII-6585'],:,:]

        
        # Calculate EWs
        if((FIT_FLG==0) | (FIT_FLG==1)):
            ewNaI = ew_NaI_allspax.ew(fits_dap_map, fits_dap_cube, fits_drp, blim, rlim, FIT_FLG=FIT_FLG)
        elif(FIT_FLG==2):
            ewNaI = ew_NaI_allspax.ew(fits_dap_map, fits_dap_cube, fits_drp, blim, rlim, FIT_FLG=FIT_FLG)
        
        # Read in NaI fit results
        outfits = mcmcdir+plate_str+"-"+ifu_str+outroot
        mcmcfits = mcmcdir+plate_str+"-"+ifu_str+"-samples.fits"
        mcmc = fits.getdata(mcmcfits, 1)
        tmcmc = Table(mcmc)
        nubins = len(tmcmc['bin'])
        print(mcmcfits, nubins)

        # Calculate velocities from fit
        transinfo = model_NaI.transitions()
        vel50_bin = np.zeros(nubins) - 999.0
        sigflow_bin = np.zeros(nubins) - 999.0

        dum_data = np.zeros(binid.shape)
        NaIvel50_map = np.zeros(binid.shape)
        NaIvel16_map = np.zeros(binid.shape)
        NaIvel84_map = np.zeros(binid.shape)
        NaIvel05_map = np.zeros(binid.shape)
        NaIvel95_map = np.zeros(binid.shape)
        NaIvel01_map = np.zeros(binid.shape)
        NaIvel99_map = np.zeros(binid.shape)
        NaIelvel50_map = np.zeros(binid.shape)
        NaIelvel16_map = np.zeros(binid.shape)
        NaIelvel84_map = np.zeros(binid.shape)
        NaIelvel05_map = np.zeros(binid.shape)
        NaIelvel95_map = np.zeros(binid.shape)
        NaIelvel01_map = np.zeros(binid.shape)
        NaIelvel99_map = np.zeros(binid.shape)

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

        ewNaI_obs_map = np.zeros(binid.shape)
        ewNaI_mod_map = np.zeros(binid.shape)
        s2n_map = np.zeros(binid.shape)
        
        #NaIvel50_msk = np.zeros(binid.shape, dtype=bool)
        #NaIsig_map = np.zeros(binid.shape)

        for qq in range(len(ewNaI['binid'])):

            ewind = np.where(binid==ewNaI['binid'][qq])
            ewNaI_obs_map[ewind] = ewNaI['obsew'][qq]
            ewNaI_mod_map[ewind] = ewNaI['modew'][qq]
            s2n_map[ewind] = ewNaI['s2n'][qq]
            
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

            
            vel50_bin[qq] = vel50

            
            # Also calc velocity relative to emission-line redshift
            ind = np.where(binid==binid_mcmc)
            qqsvf = stellar_vfield[ind]
            qqevf = emission_vfield[ind]

            cosmo_z = cz / sol
            #bin_z = (cz + qqsvf) / sol
            bin_z = cosmo_z + ((1 + cosmo_z) * qqsvf / sol) 
            #bin_elz = (cz + qqevf) / sol
            bin_elz = cosmo_z + ((1 + cosmo_z) * qqevf / sol) 
            #pdb.set_trace()
            obslamred_mcmc = lamred_mcmc * (1.0 + bin_z[0])
            elvel50 = (obslamred_mcmc[0] - ((1.0 + bin_elz[0])*transinfo['lamred0'])) * sol / \
                ((1.0 + bin_elz[0])*transinfo['lamred0'])
            elvel16 = ((obslamred_mcmc[0] - obslamred_mcmc[2]) - ((1.0 + bin_elz[0])*transinfo['lamred0'])) * sol / \
                ((1.0 + bin_elz[0])*transinfo['lamred0'])
            elvel84 = (obslamred_mcmc[0] + obslamred_mcmc[1] - ((1.0 + bin_elz[0])*transinfo['lamred0'])) * sol / \
                ((1.0 + bin_elz[0])*transinfo['lamred0'])

            obslamred05 = lamred05 * (1.0 + bin_z[0])
            obslamred95 = lamred95 * (1.0 + bin_z[0])
            obslamred01 = lamred01 * (1.0 + bin_z[0])
            obslamred99 = lamred99 * (1.0 + bin_z[0])

            elvel05 = (obslamred05 - ((1.0 + bin_elz[0])*transinfo['lamred0'])) * sol / \
                ((1.0 + bin_elz[0])*transinfo['lamred0'])
            elvel95 = (obslamred95 - ((1.0 + bin_elz[0])*transinfo['lamred0'])) * sol / \
                ((1.0 + bin_elz[0])*transinfo['lamred0'])
            elvel01 = (obslamred01 - ((1.0 + bin_elz[0])*transinfo['lamred0'])) * sol / \
                ((1.0 + bin_elz[0])*transinfo['lamred0'])
            elvel99 = (obslamred99 - ((1.0 + bin_elz[0])*transinfo['lamred0'])) * sol / \
                ((1.0 + bin_elz[0])*transinfo['lamred0'])
 

            
            NaIvel50_map[ind] = vel50_bin[qq]
            NaIvel16_map[ind] = vel16
            NaIvel84_map[ind] = vel84
            NaIelvel50_map[ind] = elvel50
            NaIelvel16_map[ind] = elvel16
            NaIelvel84_map[ind] = elvel84
             

            NaIvel05_map[ind] = vel05
            NaIvel95_map[ind] = vel95
            NaIelvel05_map[ind] = elvel05
            NaIelvel95_map[ind] = elvel95

            NaIvel01_map[ind] = vel01
            NaIvel99_map[ind] = vel99
            NaIelvel01_map[ind] = elvel01
            NaIelvel99_map[ind] = elvel99

            
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
        nai_hdu.append(fits.ImageHDU(binarea, name='BINAREA'))
        nai_hdu.append(fits.ImageHDU(binsnr, name='BINSNR'))
        nai_hdu.append(fits.ImageHDU(spx_ellcoo, name='SPX_ELLCOO'))
        nai_hdu.append(fits.ImageHDU(bin_lwskycoo, name='BIN_LWSKYCOO'))
        nai_hdu.append(fits.ImageHDU(bin_lwellcoo, name='BIN_LWELLCOO'))
        nai_hdu.append(fits.ImageHDU(stellar_vfield, name='STELLAR_VFIELD'))
        nai_hdu.append(fits.ImageHDU(stellar_vfield_ivar, name='STELLAR_VFIELD_IVAR'))
        nai_hdu.append(fits.ImageHDU(stellar_vfield_mask, name='STELLAR_VFIELD_MASK'))
        nai_hdu.append(fits.ImageHDU(emission_vfield, name='EMISSION_VFIELD'))
        nai_hdu.append(fits.ImageHDU(emission_vfield_ivar, name='EMISSION_VFIELD_IVAR'))
        nai_hdu.append(fits.ImageHDU(emission_vfield_mask, name='EMISSION_VFIELD_MASK'))
        nai_hdu.append(fits.ImageHDU(Ha_gflux, name='HA_GFLUX'))
        nai_hdu.append(fits.ImageHDU(Ha_gflux_mask, name='HA_GFLUX_MASK'))
        nai_hdu.append(fits.ImageHDU(Ha_gflux_ivar, name='HA_GFLUX_IVAR'))
        nai_hdu.append(fits.ImageHDU(Hb_gflux, name='HB_GFLUX'))
        nai_hdu.append(fits.ImageHDU(Hb_gflux_mask, name='HB_GFLUX_MASK'))
        nai_hdu.append(fits.ImageHDU(Hb_gflux_ivar, name='HB_GFLUX_IVAR'))
        nai_hdu.append(fits.ImageHDU(OIII5008_gflux, name='OIII5008_GFLUX'))
        nai_hdu.append(fits.ImageHDU(OIII5008_gflux_mask, name='OIII5008_GFLUX_MASK'))
        nai_hdu.append(fits.ImageHDU(OIII5008_gflux_ivar, name='OIII5008_GFLUX_IVAR'))
        nai_hdu.append(fits.ImageHDU(NII6585_gflux, name='NII6585_GFLUX'))
        nai_hdu.append(fits.ImageHDU(NII6585_gflux_mask, name='NII6585_GFLUX_MASK'))
        nai_hdu.append(fits.ImageHDU(NII6585_gflux_ivar, name='NII6585_GFLUX_IVAR'))

        
        nai_hdu.append(fits.ImageHDU(NaIvel01_map, name='NAIVEL01_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIvel05_map, name='NAIVEL05_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIvel16_map, name='NAIVEL16_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIvel50_map, name='NAIVEL50_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIvel84_map, name='NAIVEL84_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIvel95_map, name='NAIVEL95_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIvel99_map, name='NAIVEL99_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIelvel01_map, name='NAIELVEL01_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIelvel05_map, name='NAIELVEL05_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIelvel16_map, name='NAIELVEL16_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIelvel50_map, name='NAIELVEL50_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIelvel84_map, name='NAIELVEL84_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIelvel95_map, name='NAIELVEL95_MAP'))
        nai_hdu.append(fits.ImageHDU(NaIelvel99_map, name='NAIELVEL99_MAP'))
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

        
        nai_hdu.append(fits.ImageHDU(ewNaI_obs_map, name='EWNAI_OBS_MAP'))
        nai_hdu.append(fits.ImageHDU(ewNaI_mod_map, name='EWNAI_MOD_MAP'))
        nai_hdu.append(fits.ImageHDU(s2n_map, name='S2N_MAP'))
        
        nai_hdu.writeto(outfits, overwrite=overwrite)
        # First, plot vel50
        # Then, adjust color for bins with central velocities outside of 95% limits

def main():

    NaImcmc_write_fitinfo(outdir=None, overwrite=False, FIT_FLG=0)

main()
