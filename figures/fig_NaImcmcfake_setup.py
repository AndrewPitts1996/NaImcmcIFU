import numpy as np
import scipy.special as sp
import math
from linetools.spectra.xspectrum1d import XSpectrum1D
from linetools.spectra import convolve as lsc
from astropy import units as u
from astropy.units import Quantity
from astropy.io import fits
from astropy.table import Table
from mangadap.util.fileio import rec_to_fits_type
from mangadap.util.fileio import rec_to_fits_col_dim
import json
import sys
import pdb

import matplotlib.pyplot as pl
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages
pl.rcParams['font.family'] = 'stixgeneral'
pl.rcParams['mathtext.fontset'] = 'stix'

import model_NaI
import continuum_normalize_NaI
import NaImcmc_read_fits
import ew_NaI_allspax

def makefig_NaImcmcfake_setup(plate=None, ifu=None, binid_for_fit=None, s2n=None, setup_type=None):

    if(setup_type=='NoiseOnly'):
        filtype = 'noiseonly'
    elif(setup_type=='AddAbs'):
        filtype = 'addabs'
    else:
        print("Illegitimate setup_type")
        pdb.set_trace()

    dir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/FakeSets/'+setup_type

    s2n_str = '{0:.0f}'.format(s2n)
    infil_npy = dir + '/'+str(plate)+'-'+str(ifu)+'-'+\
        str(binid_for_fit)+'-'+filtype+'-fakeset-SN'+s2n_str+'.npy'
    outfil_pdf = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/Plots/'+\
        'fig_examine_fakeabslines'+'-'+str(plate)+'-'+str(ifu)+'-'+\
        str(binid_for_fit)+'-'+filtype+'-fakeset-SN'+s2n_str+'.pdf'

    dictList = np.load(infil_npy)

    
    pltct = 0
    nperpage = 30

    with PdfPages(outfil_pdf) as pdf:

        for nn in range(len(dictList)):

            ## SOMETHING!!!!
            dict = dictList[nn]
            wv = dict['wave']
            flx = dict['flux_fkline']
            mod = dict['flux']

            if((pltct==0) | (pltct==nperpage)):
                pltct=0
                pl.figure(figsize=(13.0,11.0))
                pl.rcParams.update({'font.size': 12})

            transinfo = model_NaI.transitions()
            ax = pl.subplot(5,6,pltct+1)
            ax.xaxis.set_minor_locator(pl.MultipleLocator(10))
            ax.set_ylim(0.5,1.1)
            ax.plot(wv,mod,drawstyle='steps-mid',color='red')
            ax.plot(wv,flx,drawstyle='steps-mid',color='black')
            #pl.text(0.0,1.08,str(fake_Cf[ii])+'_'+str(fake_logN[jj])+\
            #        '_'+str(fake_bD[kk])+'_'+str(fake_v[ll]),\
            #        ha='left', va='center', transform = ax.transAxes)
                        
            ax.plot(np.zeros(10)+transinfo['lamblu0'], np.arange(10), \
                    color='gray')
            ax.plot(np.zeros(10)+transinfo['lamred0'], np.arange(10), \
                    color='gray')
                            
            if(pltct==nperpage-1):
                pl.tight_layout()
                pdf.savefig()
                        
            pltct = pltct+1


def main():

    plate = 9194
    ifu = 9101
    binid = 115
    s2n = 30.0
    setup_type = 'NoiseOnly'
    
    makefig_NaImcmcfake_setup(plate=plate, ifu=ifu, \
                              binid_for_fit=binid, \
                              s2n=s2n, setup_type=setup_type)
    
main()
