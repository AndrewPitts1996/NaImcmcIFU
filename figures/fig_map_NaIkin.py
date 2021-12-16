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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from marvin.tools.maps import Maps
import marvin.utils.plot.map as mapplot
import pdb


sol = 2.998e5    # km/s
overwrite = True

# Read in DAP file
mcmcdir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/MCMC/SQUARE2.0/2020mar01/FluxFit/'
saspath = '/data/manga/sas/mangawork/'
pattern = '*samples*.fits'
mapfilpattern = '*NaImaps*.fits'

# Find all objects which have been fit
allmcmc = []

for root, dirs, files in os.walk(mcmcdir):
    for filename in fnmatch.filter(files,mapfilpattern):
        allmcmc.append(os.path.join(root,filename))

print("Found this many NaI map outputs to plot:")
print(len(allmcmc))

# Pixels are 69 km/s wide --> only mark flows > 35 km/s (half pixel)
vflow_outmin = 40.0
vflow_inmin = 50.0

for nn in range(len(allmcmc)):

    nnmcmc = allmcmc[nn]
    nnfil = os.path.basename(nnmcmc)
    print(nnfil)
    nnfilsplit = nnfil.split('-')
    plate = int(nnfilsplit[0])
    ifu = int(nnfilsplit[1])

    ## Set up file names
    plate_str = "{:.0f}".format(plate)
    ifu_str = "{:.0f}".format(ifu)
    fout = mcmcdir+plate_str+"-"+ifu_str+"-fit-map-3sigma.pdf"

    if(((os.path.exists(fout)) & (overwrite==False)) | (plate_str+"-"+ifu_str=='9501-12705')):

        print("Skipping ", plate_str+"-"+ifu_str)

    #elif(plate_str+"-"+ifu_str=='8440-6104'):

    else:

        dap = 'users/u6021943/Alt-DAP/SQUARE2.0_2020jan09/SQUARE2.0-MILESHC-MASTARHC/'+plate_str+'/'+ifu_str+'/manga-'+plate_str+'-'+ifu_str
        fits_dap_cube = saspath + dap + '-LOGCUBE-SQUARE2.0-MILESHC-MASTARHC.fits.gz'
        fits_dap_map = saspath + dap + '-MAPS-SQUARE2.0-MILESHC-MASTARHC.fits.gz'
        fits_NaI_map = mcmcdir + plate_str+'-'+ifu_str+'-NaImaps.fits'

        ## Read in data
        #hdu_maps = fits.open(fits_dap_map)
        #hdu_hdr = hdu_maps['PRIMARY'].header
        #cz = hdu_hdr['SCINPVEL']  # in km/s, z=v/c
        #emlc = channel_dictionary(hdu_maps, 'EMLINE_GVEL')
        #binid = hdu_maps['BINID'].data[1,:,:]

        #mask_ext = hdu_maps['STELLAR_VEL'].header['QUALDATA']
        #stellar_vfield = np.ma.MaskedArray(hdu_maps['STELLAR_VEL'].data,
        #                                   mask=hdu_maps[mask_ext].data >0)
        #mask_ext = hdu_maps['EMLINE_GVEL'].header['QUALDATA']
        #emission_vfield = np.ma.MaskedArray(hdu_maps['EMLINE_GVEL'].data[emlc['Ha-6564'],:,:],
        #                                    mask=hdu_maps[mask_ext].data[emlc['Ha-6564'],:,:] > 0)

        with fits.open(fits_NaI_map) as hdu_NaImaps:
            #hdu_NaImaps = fits.open(fits_NaI_map)
            binid = hdu_NaImaps['BINID'].data
            svf = hdu_NaImaps['STELLAR_VFIELD'].data
            evf = hdu_NaImaps['EMISSION_VFIELD'].data
            NaIvel50_map = hdu_NaImaps['NAIVEL50_MAP'].data
            NaIvel01_map = hdu_NaImaps['NAIVEL01_MAP'].data
            NaIvel05_map = hdu_NaImaps['NAIVEL05_MAP'].data
            NaIvel16_map = hdu_NaImaps['NAIVEL16_MAP'].data
            NaIvel84_map = hdu_NaImaps['NAIVEL84_MAP'].data
            NaIvel95_map = hdu_NaImaps['NAIVEL95_MAP'].data
            NaIvel99_map = hdu_NaImaps['NAIVEL99_MAP'].data
            NaIelvel01_map = hdu_NaImaps['NAIELVEL01_MAP'].data
            NaIelvel05_map = hdu_NaImaps['NAIELVEL05_MAP'].data
            NaIelvel95_map = hdu_NaImaps['NAIELVEL95_MAP'].data
            NaIelvel99_map = hdu_NaImaps['NAIELVEL99_MAP'].data
            s2n = hdu_NaImaps['S2N_MAP'].data
            
            NaIvel50_msk = np.zeros(binid.shape,dtype=bool)
            NaIvel50_sigflowmsk = np.ones(binid.shape,dtype=bool)
        
            bdbin = np.where((NaIvel84_map - NaIvel16_map) > 200.0)
            NaIvel50_msk[bdbin] = True
                
            ind_outflow = np.where((NaIvel99_map < 0.0) & (NaIelvel99_map < 0.0) &
                                   (np.fabs(NaIvel50_map) > vflow_outmin) & (s2n > 30.0) &
                                   ((NaIvel84_map - NaIvel16_map) < 200.0))

            ind_inflow = np.where((NaIvel01_map > 0.0) & (NaIelvel01_map > 0.0) &
                                  (np.fabs(NaIvel50_map) > vflow_inmin) & (s2n > 30.0) &
                                  ((NaIvel84_map - NaIvel16_map) < 200.0))
            NaIvel50_sigflowmsk[ind_inflow] = False
            NaIvel50_sigflowmsk[ind_outflow] = False
        
            NaIvel50_mapmsk = np.ma.MaskedArray(NaIvel50_map, mask=NaIvel50_msk)
            NaIvel50_sigflow = np.ma.MaskedArray(NaIvel50_map, mask=NaIvel50_sigflowmsk)
                                     

            #if(nn==3):
            #    pdb.set_trace()
            # Read in NaI fit results
            #mcmcfits = mcmcdir+plate_str+"-"+ifu_str+"-samples.fits"
            #mcmc = fits.getdata(mcmcfits, 1)
            #tmcmc = Table(mcmc)
            #nubins = len(tmcmc['bin'])

            # Calculate velocities from fit
            #transinfo = model_NaI.transitions()
            #vel50_bin = np.zeros(nubins) - 999.0
            #sigflow_bin = np.zeros(nubins) - 999.0

            #dum_data = np.zeros(binid.shape)
            #NaIvel50_map = np.zeros(binid.shape)
            #NaIvel50_msk = np.zeros(binid.shape, dtype=bool)
            #NaIsig_map = np.zeros(binid.shape)
            #for qq in range(nubins):

            #    binid_mcmc = tmcmc['bin'][qq]
            #    samples = tmcmc['samples'][qq]
            #    percentiles_mcmc = tmcmc['percentiles'][qq]
            #    lamred_mcmc = percentiles_mcmc[0,:]
            
            #    vel50 = (lamred_mcmc[0] - transinfo['lamred0']) * sol / transinfo['lamred0']
            #    vel16 = ((lamred_mcmc[0] - lamred_mcmc[2]) - transinfo['lamred0']) * sol / transinfo['lamred0']
            #    vel84 = (lamred_mcmc[0] + lamred_mcmc[1] - transinfo['lamred0']) * sol / transinfo['lamred0']
            
            #    vel50_bin[qq] = vel50
            
            
            #    # Also calc velocity relative to emission-line redshift
            #    ind = np.where(binid==binid_mcmc)
            #    qqsvf = stellar_vfield[ind]
            #    qqevf = emission_vfield[ind]
            
            #    bin_z = (cz + qqsvf) / sol
            #    bin_elz = (cz + qqevf) / sol
            #pdb.set_trace()
            #    obslamred_mcmc = lamred_mcmc * (1.0 + bin_z[0])
            #    elvel50 = (obslamred_mcmc[0] - ((1.0 + bin_elz[0])*transinfo['lamred0'])) * sol / \
                #        ((1.0 + bin_elz[0])*transinfo['lamred0'])
            #    elvel16 = ((obslamred_mcmc[0] - obslamred_mcmc[2]) - ((1.0 + bin_elz[0])*transinfo['lamred0'])) * sol / \
                #        ((1.0 + bin_elz[0])*transinfo['lamred0'])
            #    elvel84 = (obslamred_mcmc[0] + obslamred_mcmc[1] - ((1.0 + bin_elz[0])*transinfo['lamred0'])) * sol / \
                #        ((1.0 + bin_elz[0])*transinfo['lamred0'])
            
            #    NaIvel50_map[ind] = vel50_bin[qq]
            #    NaIsig_map[ind] = math.fabs(vel50_bin[qq] / ((vel84 - vel16)/2.0))
            # Mask bins
            #    if((vel84-vel16) > 50.0):
            #print('Masking!')
            #        NaIvel50_msk[ind] = True

            #    NaIvel50_map = np.ma.MaskedArray(NaIvel50_map, mask=NaIvel50_msk)
        
            ## Also make maps of velocity offset significance!
        
            # Calculate 95th-percentile limits
            #    lamred_samples = samples[:,0]
            #    lamred05 = np.percentile(lamred_samples,5)
            #    lamred95 = np.percentile(lamred_samples,95)
            
            #    if((transinfo['lamred0'] > lamred95) & (math.fabs(vel50) > vflow_min)):
            #        sigflow_qq = vel50               
            
         
         
                
            #    elif((transinfo['lamred0'] < lamred05) & (math.fabs(vel50) > vflow_min)):
            #        sigflow_qq = vel50
            
         
               
            #    else:
            #        sigflow_qq = 0.0

            #    sigflow_bin[qq] = sigflow_qq
            #pdb.set_trace()

  

            # Make plot - of vel50
            #patch_kws = mapplot.set_patch_style()
            fig, axes = plt.subplots(1, 4, figsize=(16,4))
            #mapplot.plot(value=(evf-svf), title='Relative Emission Line Velocity',
            #             cblabel='Relative Velocity (km/s)', cmap=cm.coolwarm, cbrange=[-80,80], fig=fig, ax=axes[0])
            mapplot.plot(value=svf, title='Stellar Velocity Field',
                         cblabel='Relative Velocity (km/s)', cmap=cm.coolwarm, cbrange=[-200,200], fig=fig, ax=axes[0])
            mapplot.plot(value=s2n, title='S/N (NaI)', cblabel='per XX', cmap=cm.viridis, cbrange=[0,30], fig=fig, ax=axes[1])
            mapplot.plot(value=NaIvel50_mapmsk, mask=NaIvel50_msk, title='NaI Absorption Velocity',
                         cblabel='Relative Velocity (km/s)', cmap=cm.coolwarm, cbrange=[-90,90], fig=fig, ax=axes[2])
            mapplot.plot(value=NaIvel50_sigflow, mask=NaIvel50_sigflowmsk, title='Significant In/Outflow',
                         cblabel='Relative Velocity (km/s)', cmap=cm.coolwarm, cbrange=[-90,90], fig=fig, ax=axes[3])
            fig.tight_layout()
          
    
            plt.savefig(fout)
   
    
# Make plot - of bins w/ significant inflow/outflow
#mapval = np.array(sigflow_bin)
#print(mapval)
#kwargs = dict(cblabel='Velocity (km/s)', cmap=cmap, title_text='NaI Absorption', nodots=True, spaxel_num=True)
#qa.plot_map(mapval, **kwargs)
#fout = dir+plate_str+"-"+ifu_str+"-sigflow-map.pdf"
#plt.savefig(fout)


# First, plot vel50
# Then, adjust color for bins with central velocities outside of 95% limits
