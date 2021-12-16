from __future__ import print_function

import math
import numpy as np
import scipy.optimize as op
import os
import fnmatch
from astropy.io import ascii
import matplotlib
matplotlib.use('Agg')
from mangadap.util.fitsutil import DAPFitsUtil#unique_bins
import matplotlib.pyplot as pl
import matplotlib.image as mpimg
pl.rcParams['font.family'] = 'stixgeneral'
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
#matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages


def fig_NaImcmc_inspect_morphologies(INFIL=None, OUTFIL=None):

    wrkpath = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/plan_files/SQUARE2.0/2020mar01/FluxFit/'

    if(INFIL==None):
        INFIL = wrkpath + 'NaImcmc_select_fitsample_GZSpirals_LSF_extended.plan'

    if(OUTFIL==None):
        OUTFIL = wrkpath + 'fig_NaImcmc_select_fitsample_GZSpirals_LSF_extended.pdf'

    saspath = '/data/manga/sas/mangawork/'
    table = ascii.read(INFIL)
    drp = table['drp']
    dap = table['dap']

    ngal = len(drp)

    pltct = 0
    nperpage = 72
    
    with PdfPages(OUTFIL) as pdf:

        for qq in range(len(drp)):
        #for qq in range(560):

            if((qq % nperpage)==0):

                pl.figure(figsize=(13.0,11.0))
                pl.rcParams.update({'font.size':14})
                pltct = 0

            ax = pl.subplot(9,8,pltct+1)
            
            #fits_drp = saspath + drp[qq] + '-LOGCUBE.fits.gz'
            dum1, dum2, plate, ifu = drp[qq].split('-')
            plateifu = plate+'-'+ifu

            impng_fil = saspath + 'manga/spectro/redux/MPL-9/'+plate+'/images/'+ifu+'.png'

            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            ax.yaxis.set_major_formatter(ticker.NullFormatter())
            ax.set_xticks([])
            ax.set_yticks([])
            impng = mpimg.imread(impng_fil)

            img = ax.imshow(impng, interpolation='nearest')
            ax.grid(False)

            ax.text(0.05,0.925, plateifu, horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes, fontsize=9, color='white')

            if((pltct==nperpage-1) | (qq==len(drp)-1)):

                pl.tight_layout()
                pdf.savefig()

            pltct = pltct+1

def main():

    fig_NaImcmc_inspect_morphologies(INFIL=None, OUTFIL=None)

if __name__ == '__main__':

    main()
