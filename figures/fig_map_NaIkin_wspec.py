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
from mangadap.util.fitsutil import DAPFitsUtil
import model_NaI
import NaImcmc_read_fits
import continuum_normalize_NaI
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
plt.rcParams['font.family'] = 'stixgeneral'
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages
from marvin.tools.maps import Maps
import marvin.utils.plot.map as mapplot
import pdb


sol = 2.998e5    # km/s
overwrite = True

# Set object
plate = 8982
ifu = 6104

# Read in DAP file
mcmcdir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/MCMC/SQUARE2.0/2019feb04/FluxFit/'
saspath = '/data/home/krubin/sas/mangawork/'
pattern = '*samples*.fits'
mapfilpattern = '*NaImaps*.fits'


# Pixels are 69 km/s wide --> only mark flows > 35 km/s (half pixel)
vflow_min = 60.0

## Set up file names
plate_str = "{:.0f}".format(plate)
ifu_str = "{:.0f}".format(ifu)
fout = mcmcdir+plate_str+"-"+ifu_str+"-fit-map-3sigma-NSF.pdf"

dap = 'users/u6021943/Alt-DAP/SQUARE2.0_2018oct25/SQUARE2.0-GAU-MILESHC/'+plate_str+'/'+ifu_str+'/manga-'+plate_str+'-'+ifu_str
drp = 'manga/spectro/redux/MPL-7/'+plate_str+'/stack/manga-'+plate_str+'-'+ifu_str
fits_dap_cube = saspath + dap + '-LOGCUBE-SQUARE2.0-GAU-MILESHC.fits.gz'
fits_dap_map = saspath + dap + '-MAPS-SQUARE2.0-GAU-MILESHC.fits.gz'
fits_drp = saspath + drp + '-LOGCUBE.fits.gz'
fits_NaI_map = mcmcdir + plate_str+'-'+ifu_str+'-NaImaps.fits'

#with fits.open(fits_NaI_map) as hdu_NaImaps:
hdu_NaImaps = fits.open(fits_NaI_map)
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
            
NaIvel50_msk = np.zeros(binid.shape,dtype=bool)
NaIvel50_sigflowmsk = np.ones(binid.shape,dtype=bool)
        
bdbin = np.where((NaIvel84_map - NaIvel16_map) > 200.0)
NaIvel50_msk[bdbin] = True
                
ind_outflow = np.where((NaIvel99_map < 0.0) & (NaIelvel99_map < 0.0) &
                       (np.fabs(NaIvel50_map) > vflow_min) &
                       ((NaIvel84_map - NaIvel16_map) < 200.0))

ind_inflow = np.where((NaIvel01_map > 0.0) & (NaIelvel01_map > 0.0) &
                      (np.fabs(NaIvel50_map) > vflow_min) &
                      ((NaIvel84_map - NaIvel16_map) < 200.0))
NaIvel50_sigflowmsk[ind_inflow] = False
NaIvel50_sigflowmsk[ind_outflow] = False
        
NaIvel50_mapmsk = np.ma.MaskedArray(NaIvel50_map, mask=NaIvel50_msk)
NaIvel50_sigflow = np.ma.MaskedArray(NaIvel50_map, mask=NaIvel50_sigflowmsk)
                                     
# Make plot - of vel50
#patch_kws = mapplot.set_patch_style()
fig, axes = plt.subplots(1, 3, figsize=(13,4))
plt.subplots_adjust(wspace=0.6)
plt.rcParams.update({'font.size': 13})
cbfontsize = 12

pos = axes[0].get_position()
axpad = 0.09
cbpos = [pos.x1+0.01, pos.y0+axpad, 0.01, pos.height-0.18]
mapplot.plot(value=svf, title='Stellar Velocity Field',
             cblabel='Relative Velocity (km/s)', cmap=cm.coolwarm,
             cbrange=[-200,200], fig=fig, ax=axes[0], sky_coords=True,
             cb_kws={'axloc':cbpos, 'label_kws':{'size':cbfontsize, 'family':'stixgeneral'},
                     'tick_params_kws':{'labelsize':cbfontsize-2}})

pos = axes[1].get_position()
axpad = 0.09
cbpos = [pos.x1+0.01, pos.y0+axpad, 0.01, pos.height-0.18]
mapplot.plot(value=NaIvel50_mapmsk, mask=NaIvel50_msk, title='NaI Absorption Velocity',
             cblabel='Relative Velocity (km/s)', cmap=cm.coolwarm,
             cbrange=[-120,120], fig=fig, ax=axes[1], sky_coords=True,
             cb_kws={'axloc':cbpos, 'label_kws':{'size':cbfontsize, 'family':'stixgeneral'},
                     'tick_params_kws':{'labelsize':cbfontsize-2}})

# Create a Rectangle patch
#rect = patches.Rectangle((20.0,20.0),5.0,5.0,linewidth=1,edgecolor='white',facecolor='white')#,transform=axes[1].get_transform('fk5'))
#axes[1].add_patch(rect)
#pdb.set_trace()
mapplot.plot(value=NaIvel50_sigflow, mask=NaIvel50_sigflowmsk, title='Significant In/Outflow',
             cblabel='Relative Velocity (km/s)', cmap=cm.coolwarm,
             cbrange=[-120,120], fig=fig, ax=axes[2], sky_coords=True)
#fig.tight_layout()
          
    
plt.savefig(fout)

#pdb.set_trace()

## Now plot the spectra
foutspec = mcmcdir+plate_str+"-"+ifu_str+"-fit-spec-3sigma-NSF.pdf"


## Choose bins to plot
pltbins = [58,52,51,57,53,26]

cube = NaImcmc_read_fits.NaImcmc_read_fits(fits_dap_map, fits_dap_cube, fits_drp=fits_drp)

wave = cube['wave']
sres = cube['sres']
flux = cube['fluxb']
ivar = cube['ivarb']
smod = cube['modelb']
cz = cube['cz']
stellar_vfield = cube['stellar_vfieldb']
stellar_vfield_ivar = cube['stellar_vfield_ivarb']
emission_vfield = cube['emission_vfieldb']
emission_vfield_ivar = cube['emission_vfield_ivarb']
binid_cube = cube['binidb']

    
# For continuum-normalization around NaI
blim = [5850.0,5870.0]
rlim = [5910.0,5930.0]
fitlim = [5880.0,5910.0]
inspectlim = [5850.,5930.]

fig, axes = plt.subplots(3, 2, sharex='col', sharey='row', figsize=(4,4))
#plt.figure(figsize=(4.0,4.0))
plt.rcParams.update({'font.size': 5})
nrows = axes.shape[0]
ncols = axes.shape[1]
row = 0
col = 0

ubins, uindx = DAPFitsUtil.unique_bins(hdu_NaImaps['BINID'].data, return_index=True)
binid_map = hdu_NaImaps['BINID'].data.ravel()[uindx]
vel50 = hdu_NaImaps['NAIVEL50_MAP'].data.ravel()[uindx]

for qq in range(len(pltbins)):

    if(row==nrows):
        row = 0    
        col = col + 1
    ax = axes[row,col]
    
    ind = np.where(binid_cube==pltbins[qq])
    ind = ind[0][0]

    indmap = np.where(binid_map==pltbins[qq])
    #pdb.set_trace()
    print(vel50[indmap[0]])
    
    flux_bin = flux[:,ind]
    smod_bin = smod[:,ind]
    ivar_bin = ivar[:,ind]
    err_bin = 1.0/np.sqrt(ivar_bin)
    bin_z = (cz + stellar_vfield[ind]) / sol
    restwave = wave / (1.0+bin_z)

    ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, blim, rlim)

    # Cut out NaI
    naiind = np.where((restwave > inspectlim[0]) & (restwave < inspectlim[1]))
    restwave_NaI = restwave[naiind]
    flux_NaI = flux_bin[naiind]
    err_NaI = err_bin[naiind]
    #smod_NaI = nsmod['nflux'][ind]
    #smod_norm = smod_bin / ndata['cont']
    #sres_NaI = sres[naiind]
    smod_NaI = smod_bin[naiind]
    cont_NaI = ndata['cont'][naiind]
    normflux_NaI = flux_NaI / cont_NaI
    normsmod_NaI = smod_NaI / cont_NaI
    transinfo = model_NaI.transitions()
    vel_NaI = (restwave_NaI - transinfo['lamred0']) * sol / transinfo['lamred0']

    vel_NaIblue = (transinfo['lamblu0'] - transinfo['lamred0']) * sol / transinfo['lamred0']

    #pdb.set_trace()
    #ax = plt.subplot(3,2,qq+1, sharex=True, sharey=True)
    #ax = axes[qq]
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.0)

    ax.plot(vel_NaI,normsmod_NaI, color="red", drawstyle='steps-mid')    
    ax.plot(vel_NaI,normflux_NaI,drawstyle='steps-mid', color="k")
    #ax.plot(restwave_NaI,flux_NaI+err_NaI,drawstyle='steps-mid', color="gray", alpha=0.5)
    #ax.plot(restwave_NaI,flux_NaI-err_NaI,drawstyle='steps-mid', color="gray", alpha=0.5)
    #ax.plot(restwave_NaI,cont_NaI, color='cyan')
    #ax.plot(best_mod['modwv'], cont_NaI[fitind] * best_mod['modflx'], color="cyan")

    ax.set_xlim(-1500.0,1200.0)
    ax.set_ylim(0.65,1.1)
    ylim = ax.get_ylim()
    ax.plot([vel_NaIblue,vel_NaIblue], [ylim[0], ylim[1]], color="gray", ls='--')
    ax.plot([0.0,0.0], [ylim[0], ylim[1]], color="gray", ls='--')
    #ax.set_xlabel(r'Rest Wavelength (Ang)')
    #ax.set_ylabel(r'Flux')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(4.0))
    

    if(row==2):
        ax.set_xlabel(r'Relative Velocity (km/s)')
    if((row==1) & (col==0)):
        ax.set_ylabel(r'Normalized Flux')

    row = row + 1


    cmap=cm.coolwarm
    cnorm = matplotlib.colors.Normalize(vmin=-120.0, vmax=120.0)
    qqcolor = cnorm(vel50[indmap[0]][0])
    # Create a Rectangle patch
    rect = patches.Rectangle((400,0.7),500,0.1,linewidth=1,edgecolor='white',facecolor=cmap(qqcolor))

    # Add the patch to the Axes
    ax.add_patch(rect)
    
plt.subplots_adjust(wspace=0, hspace=0)    
#plt.tight_layout()
plt.savefig(foutspec)
pdb.set_trace()
    
# Make plot - of bins w/ significant inflow/outflow
#mapval = np.array(sigflow_bin)
#print(mapval)
#kwargs = dict(cblabel='Velocity (km/s)', cmap=cmap, title_text='NaI Absorption', nodots=True, spaxel_num=True)
#qa.plot_map(mapval, **kwargs)
#fout = dir+plate_str+"-"+ifu_str+"-sigflow-map.pdf"
#plt.savefig(fout)


# First, plot vel50
# Then, adjust color for bins with central velocities outside of 95% limits
