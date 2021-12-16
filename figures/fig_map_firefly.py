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

# Read in FIREFLY file
saspath = '/data/home/krubin/sas/mangawork/'
fireflyfil = saspath+'manga/spectro/analysis/manga_firefly-v2_1_2-STELLARPOP.fits'
ff_maps = fits.open(fireflyfil)

# Set desired object
plate = 8440
ifu = 6104
x = [2.6,4.7,2.7]
y = [1.2,-0.7,-2.7]
#plate=7443
#ifu=6102
#x = [-4.7,1.2]
#y = [4.7,-5.1]
#plate=8549
#ifu=12705
#x = [3.2, 1.2]
#y = [3.2, -2.6]
win = 1.0    ## arcsec

## Set up file names
plate_str = "{:.0f}".format(plate)
ifu_str = "{:.0f}".format(ifu)
plateifu = plate_str+'-'+ifu_str
fout = '/data/home/krubin/Projects/MaNGA/NaImcmc/Figures/Maps/FIREFLY/'+\
    plate_str+"-"+ifu_str+"-FIREFLY-map.pdf"

# Read in FIREFLY info
hdu_basic = ff_maps['GALAXY_INFO'].data
hdu_spatial = ff_maps['SPATIAL_INFO'].data
hdu_binid = ff_maps['SPAXEL_BINID'].data
hdu_lwage = ff_maps['LW_AGE_VORONOI'].data
hdu_mwage = ff_maps['MW_AGE_VORONOI'].data
hdu_lwZ = ff_maps['LW_Z_VORONOI'].data
hdu_mwZ = ff_maps['MW_Z_VORONOI'].data

ff_maps.close()

# Select target galaxy
wh = np.where(hdu_basic['PLATEIFU']==plateifu)
#pdb.set_trace()
wh = wh[0][0]

bin_number = hdu_spatial[wh,:,0]
x_pos = hdu_spatial[wh,:,1]
y_pos = hdu_spatial[wh,:,2]

lwage = hdu_lwage[wh,:,0]
mwage = hdu_mwage[wh,:,0]
lwZ = hdu_lwZ[wh,:,0]
mwZ = hdu_mwZ[wh,:,0]

# Fill 2D array
dx = 0.5
dy = 0.5

nrows = np.int(((np.nanmax(y_pos) - np.nanmin(y_pos)) / dy)) + 2
ncols = np.int(((np.nanmax(x_pos) - np.nanmin(x_pos)) / dx)) + 2
idx = np.round((np.nanmax(x_pos) - x_pos)/dx).astype(np.int)
idy = np.round((y_pos - np.nanmin(y_pos))/dy).astype(np.int)

grid_xpos = np.zeros((nrows,ncols),dtype=np.float) - 99.
grid_ypos = np.zeros((nrows,ncols),dtype=np.float) - 99.
grid_lwage = np.zeros((nrows,ncols),dtype=np.float) - 99.
grid_mwage = np.zeros((nrows,ncols),dtype=np.float) - 99.
grid_lwZ = np.zeros((nrows,ncols),dtype=np.float) - 99.
grid_mwZ = np.zeros((nrows,ncols),dtype=np.float) - 99.
fin = np.isfinite(x_pos)

#pdb.set_trace()
grid_xpos[idy[fin],idx[fin]] = x_pos[fin]
grid_ypos[idy[fin],idx[fin]] = y_pos[fin]
grid_lwage[idy[fin],idx[fin]] = lwage[fin]
grid_mwage[idy[fin],idx[fin]] = mwage[fin]
grid_lwZ[idy[fin],idx[fin]] = lwZ[fin]
grid_mwZ[idy[fin],idx[fin]] = mwZ[fin]


# Make plot - of vel50
fig, axes = plt.subplots(1, 4, figsize=(13,4))
#mapplot.plot(value=grid_xpos, title='x position',
#             cblabel='arcsec', cmap=cm.viridis, cbrange=[-20,20], fig=fig, ax=axes[0])
#mapplot.plot(value=grid_ypos, title='y position',
#             cblabel='arcsec', cmap=cm.viridis, cbrange=[-20,20], fig=fig, ax=axes[1])
mapplot.plot(value=grid_lwage, title='Light-Weighted Age',
             cblabel='log Age (Gyr)', cmap=cm.viridis, cbrange=[0,1.5], fig=fig, ax=axes[0])
#mapplot.plot(value=grid_mwage, title='Mass-Weighted Age',
#             cblabel='log Age (Gyr)', cmap=cm.viridis, cbrange=[0,1.5], fig=fig, ax=axes[1])


#pdb.set_trace()
ax = axes[1]
binsz = 0.1 
age_bins = np.arange(min(lwage), max(lwage)+binsz, binsz)
ax.hist(lwage[np.isfinite(lwage)], bins=age_bins, histtype='stepfilled', \
        alpha=0.2, color='blue')
ax.set_xlabel('log Age (Gyr)')

for ii in range(len(x)):
    whwin = np.where((x_pos>(x[ii]-win)) & (x_pos<(x[ii]+win)) \
                     & (y_pos>(y[ii]-win)) & (y_pos<(y[ii]+win)))
    ax.hist(lwage[whwin],bins=age_bins,color='red')
    print("For x, y ", x[ii], y[ii])
    print("Mean age = ", np.mean(lwage[whwin]), np.std(lwage[whwin]))

#pdb.set_trace()

mean_age = np.nanmean(lwage)
stddev_age = np.nanstd(lwage)
ax.plot(mean_age + np.zeros(500), np.arange(0,500), color='blue')
ax.plot(mean_age + stddev_age + np.zeros(500), np.arange(0,500), color='orange')
ax.plot((mean_age-stddev_age) + np.zeros(500), np.arange(0,500), color='orange')
ax.set_ylim(0,300)

print("Age statistics (log Gyr): ", mean_age, stddev_age)
pltnote = 'mean = '+"{:.2f}".format(mean_age)+"  stddev = "+"{:.2f}".format(stddev_age)
ax.text(0.5,1.02,pltnote,ha='center',va='center',transform=ax.transAxes,fontsize=10)

mapplot.plot(value=grid_lwZ, title='Light-Weighted Metallicty',
             cblabel='[Z/H]', cmap=cm.viridis, cbrange=[-1.5,0.5], fig=fig, ax=axes[2])
#mapplot.plot(value=grid_mwZ, title='Mass-Weighted Metallicty',
#             cblabel='[Z/H]', cmap=cm.viridis, cbrange=[-1.5,0.5], fig=fig, ax=axes[3])


ax = axes[3]
binsz = 0.1
Z_bins = np.arange(min(lwZ), max(lwZ)+binsz, binsz)
ax.hist(lwZ[np.isfinite(lwZ)], bins=Z_bins, histtype='stepfilled', alpha=0.5)
ax.set_xlabel('[Z/H]')

for ii in range(len(x)):
    whwin = np.where((x_pos>(x[ii]-win)) & (x_pos<(x[ii]+win)) \
                     & (y_pos>(y[ii]-win)) & (y_pos<(y[ii]+win)))
    ax.hist(lwZ[whwin],bins=age_bins,color='red')
    print("For x, y ", x[ii], y[ii])
    print("Mean [Z/H] = ", np.mean(lwZ[whwin]), np.std(lwZ[whwin]))


mean_Z = np.nanmean(lwZ)
stddev_Z = np.nanstd(lwZ)
ax.plot(mean_Z + np.zeros(800), np.arange(0,800), color='blue')
ax.plot(mean_Z + stddev_Z + np.zeros(800), np.arange(0,800), color='orange')
ax.plot((mean_Z-stddev_Z) + np.zeros(800), np.arange(0,800), color='orange')
ax.set_ylim(0,700)

print("Z statistics ([Z/H]): ", mean_Z, stddev_Z)
pltnote = 'mean = '+"{:.2f}".format(mean_Z)+"  stddev = "+"{:.2f}".format(stddev_Z)
ax.text(0.5,1.02,pltnote,ha='center',va='center',transform=ax.transAxes,fontsize=10)

#pdb.set_trace()

fig.tight_layout()
plt.savefig(fout)
   
    

