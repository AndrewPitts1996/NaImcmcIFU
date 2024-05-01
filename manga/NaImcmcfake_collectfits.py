import numpy as np
import scipy.special as sp
import math
import os
import fnmatch
from linetools.spectra.xspectrum1d import XSpectrum1D
from linetools.spectra import convolve as lsc
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity
from mangadap.util.fileio import rec_to_fits_type
from mangadap.util.fileio import rec_to_fits_col_dim
import json
import pdb

import matplotlib.pyplot as pl
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
pl.rcParams['font.family'] = 'stixgeneral'
pl.rcParams['mathtext.fontset'] = 'stix'
import model_NaI
import corner

def NaImcmcfake_collectfits(infil=None, s2n=None, setup_type=None, fakesetsdir=None, outfil=None, outtriangle=False):

    if(infil==None):
        print("Need to choose input file")
        pdb.set_trace()
    
    if(s2n==None):
        print("Need to request a S/N level")
        pdb.set_trace()

    if(outfil==None):
        outfil = 'NaImcmcfakeresults.fits'

    if(setup_type=='NoiseOnly'):
        setup_type_root = 'noiseonly'
    elif(setup_type=='AddNoise'):
        setup_type_root = 'addnoise'
    else:
        print('Setup type not defined!')

    if(fakesetsdir==None):
        print("Need to choose fake sets directory")
        pdb.set_trace()

    if(outtriangle==True):
        triplotpattern = '-logN14p4_16p0-triangle.pdf'

    #galinfo_fits = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/NaImcmcfake_galselect_binselect.fits'
    #galinfo_fits = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/NaImcmcfake_galselect_binselect_SNmin30.fits'
    #galinfo_fits = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/NaImcmcfake_inflowbins.fits'
    galinfo = fits.getdata(infil)

    if(s2n>0.0):
        s2n_str = '{0:.0f}'.format(s2n)
    else:
        s2n_str = '*'
    
    #fakesetsdir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/FakeSets/'+setup_type+'/Detected_Inflows/'

    fitpattern = '*'+setup_type_root+'-fakeset-SN'+s2n_str+'-logN14p1_16p0-wsamples.npy'

    findall_fit = []
    plate_fit = []
    ifu_fit = []
    binid_fit = []

    for root, dirs, files in os.walk(fakesetsdir):
        for filename in fnmatch.filter(files,fitpattern):
            findall_fit.append(filename)
    print("Found this many fake datasets with fits:", len(findall_fit))

    fit_fils = []

    default_fracout = []
    default_fracin = []

    vflow_consmin = 0.0
    vflow_inmin = 50.0 
    vflow_outmin = 40.0

    vflow_min_fracin5 = []
    vflow_min_fracout5 = []

    
    for nn in range(len(findall_fit)):
    #for nn in range(0,10):

        nnfakeset = findall_fit[nn]
        nnfil = nnfakeset.split('-')
        found_plate = int(nnfil[0])
        found_ifu = int(nnfil[1])
        found_binid = int(nnfil[2])
        found_plateifubin = "{:.0f}".format(found_plate)+"-"+"{:.0f}".format(found_ifu)+"-"+"{:.0f}".format(found_binid)

               
        fit_fils.append(findall_fit[nn])
        plate_fit.append(found_plate)
        ifu_fit.append(found_ifu)
        binid_fit.append(found_binid)

        dictList = np.load(fakesetsdir+nnfakeset, allow_pickle=True)
        nsims = len(dictList)

        vsim = np.zeros(nsims) - 9999.0
        Cfsim = np.zeros(nsims) - 9999.0
        logNsim = np.zeros(nsims) - 9999.0

        vel01 = np.zeros(nsims) - 9999.0
        vel05 = np.zeros(nsims) - 9999.0
        vel16 = np.zeros(nsims) - 9999.0
        vel50 = np.zeros(nsims) - 9999.0
        vel84 = np.zeros(nsims) - 9999.0
        vel95 = np.zeros(nsims) - 9999.0
        vel99 = np.zeros(nsims) - 9999.0
        
        sol = 2.998e5    # km/s
        transinfo = model_NaI.transitions()
        
        
        for qq in range(nsims):
        
            data = dictList[qq]

            vsim[qq] = data['v']
            Cfsim[qq] = data['Cf']
            logNsim[qq] = data['logN']

            samples = data['samples']
            
            # Calculate 95th-percentile limits
            lamred_samples = samples[:,0]
            lamred50 = np.percentile(lamred_samples,50)
            lamred16 = np.percentile(lamred_samples,16)
            lamred84 = np.percentile(lamred_samples,84)
            lamred05 = np.percentile(lamred_samples,5)
            lamred95 = np.percentile(lamred_samples,95)
            lamred01 = np.percentile(lamred_samples,1)
            lamred99 = np.percentile(lamred_samples,99)
    
            vel50[qq] = (lamred50 - transinfo['lamred0']) * sol / transinfo['lamred0']
            vel16[qq] = (lamred16 - transinfo['lamred0']) * sol / transinfo['lamred0']
            vel84[qq] = (lamred84 - transinfo['lamred0']) * sol / transinfo['lamred0']
            vel05[qq] = (lamred05 - transinfo['lamred0']) * sol / transinfo['lamred0']
            vel95[qq] = (lamred95 - transinfo['lamred0']) * sol / transinfo['lamred0']
            vel01[qq] = (lamred01 - transinfo['lamred0']) * sol / transinfo['lamred0']
            vel99[qq] = (lamred99 - transinfo['lamred0']) * sol / transinfo['lamred0']

            logN_samples = samples[:,1]
            logN50 = np.percentile(logN_samples,50)
            bD_samples = samples[:,2]
            bD50 = np.percentile(bD_samples, 50)
            Cf_samples = samples[:,3]
            Cf50 = np.percentile(Cf_samples, 50)
            print(found_plateifubin, vel01, vel50, vel99, logN50, bD50)
            best_fit = lamred50, logN50, bD50, Cf50

            if(outtriangle==True):
                outtriplotfil = fakesetsdir + found_plateifubin + triplotpattern
                pl.figure(figsize=(6.0,5.0))
                pl.rcParams.update({'font.size': 14})                  
                fig = corner.corner(samples, labels=["$lamred$", "$logN$", "$bD$", "$Cf$"],
                                    truths=best_fit)
                fig.tight_layout()
                fig.savefig(outtriplotfil, format='pdf')
            
        #pdb.set_trace()
        ## Find significant flows
        ind_outflow3sig = np.where((vel99 < -1.0*vflow_consmin) & (np.fabs(vel50) > vflow_outmin) &
                                   ((vel84-vel16) < 200.0))
        ind_inflow3sig = np.where((vel01 > vflow_consmin) & (np.fabs(vel50) > vflow_inmin) &
                                  ((vel84-vel16) < 200.0))

        default_fracout.append(len(ind_outflow3sig[0]))
        default_fracin.append(len(ind_inflow3sig[0]))

        if(len(ind_inflow3sig[0])>5.0):

            # Which bins are in the running?
            ind_allin = np.where((vel01 > vflow_consmin) & ((vel84-vel16) < 200.0))
            n_allin = len(ind_allin[0])

            percentile_in = 100.0 * (1.0 - (5.0/float(n_allin)))
            percentile_where_fracin5 = np.percentile(vel50[ind_allin[0]],percentile_in)

            #if(nn==22):
            #    pdb.set_trace()
                
            vflow_min_fracin5.append(percentile_where_fracin5)
        else:
            vflow_min_fracin5.append(0.0)


        if(len(ind_outflow3sig[0])>5.0):

            # Which bins are in the running?
            ind_allout = np.where((vel99 < -1.0*vflow_consmin) & ((vel84-vel16) < 200.0))
            n_allout = len(ind_allout[0])

            percentile_out = 100.0 * (1.0 - (5.0/float(n_allout)))
            percentile_where_fracout5 = np.percentile(-1.0*vel50[ind_allout[0]],percentile_out)
            #pdb.set_trace()
                
            vflow_min_fracout5.append(percentile_where_fracout5)
        else:
            vflow_min_fracout5.append(0.0)


    plate_fit = np.array(plate_fit)
    ifu_fit = np.array(ifu_fit)
    binid_fit = np.array(binid_fit)
    default_fracin = np.array(default_fracin)
    default_fracout = np.array(default_fracout)
    vflow_min_fracin5 = np.array(vflow_min_fracin5)
    vflow_min_fracout5 = np.array(vflow_min_fracout5)
    pdb.set_trace()
            
    ## Save info
    hdr = fits.Header()
    primary_hdu = fits.PrimaryHDU(header=hdr)
    col1 = fits.Column(name='PLATE',format=rec_to_fits_type(plate_fit), array=plate_fit)
    col2 = fits.Column(name='IFU',format=rec_to_fits_type(ifu_fit), array=ifu_fit)
    col3 = fits.Column(name='BINID',format=rec_to_fits_type(binid_fit), array=binid_fit)
    col4 = fits.Column(name='DEFAULT_FRACIN',format=rec_to_fits_type(default_fracin), array=default_fracin)
    col5 = fits.Column(name='DEFAULT_FRACOUT',format=rec_to_fits_type(default_fracout), array=default_fracout)
    col6 = fits.Column(name='VFLOW_FRACIN5',format=rec_to_fits_type(vflow_min_fracin5), array=vflow_min_fracin5)
    col7 = fits.Column(name='VFLOW_FRACOUT5',format=rec_to_fits_type(vflow_min_fracout5), array=vflow_min_fracout5)

    cols = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7])
    fitres_hdu = fits.BinTableHDU.from_columns(cols)
    hdu1 = fits.HDUList([primary_hdu, fitres_hdu])
    hdu1.writeto(outfil, overwrite=True)
    pdb.set_trace()
        

def main():

    #s2n = -1.0
    s2n = 50.0
    setup_type = 'NoiseOnly'
    galinfo_fits = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/NaImcmcfake_galselect_binselect_SNmin30.fits'
    fakesetsdir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/FakeSets/'+setup_type+'/Representative_logN14p1_16p0Test/'
    
    #outfil = 'NaImcmcfakeresultsinflow_velinfo_2019feb04_FluxFit_SN30_vflowmin40-50.fits'
    outfil = 'donotsave.fits'
    
    
    NaImcmcfake_collectfits(infil=galinfo_fits, s2n=s2n, setup_type=setup_type, fakesetsdir=fakesetsdir, \
                            outfil=outfil, outtriangle=False)

main()
