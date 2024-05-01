from __future__ import print_function

import math
import numpy as np
import scipy.optimize as op
import os
import fnmatch
import matplotlib
matplotlib.use('Agg')
from mangadap.util.fitsutil import DAPFitsUtil#unique_bins
import matplotlib.pyplot as pl
pl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.ticker import MaxNLocator
#matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from astropy.table import Table
import NaImcmc_read_fits
import model_NaI
import model_fitter
import continuum_normalize_NaI
import ew_NaI_allspax
import pdb


def fig_NaImcmcfake_makecheckfit(plate_plot=None, ifu_plot=None, binid_plot=None, \
                                 setup_type=None, s2n_str=None, fakesetsdir=None, \
                                 theta_plot=None):

    sol = 2.998e5    # km/s 
    saspath = '/data/manga/sas/mangawork/'
    nperpage = 1
    
    outfilpattern = '-logN14p4_16p0-NaImcmcfake-checkfit.pdf'


    if(setup_type=='NoiseOnly'):
        setup_type_root = 'noiseonly'
    elif(setup_type=='AddNoise'):
        setup_type_root = 'addnoise'
    else:
        print('Setup type not defined!')

    infil_npy = fakesetsdir + '/'+str(plate_plot)+'-'+str(ifu_plot)+'-'+str(binid_plot)+'-'+setup_type_root+'-fakeset-SN'+s2n_str+'.npy'
    outfil = fakesetsdir + '/'+str(plate_plot)+'-'+str(ifu_plot)+'-'+str(binid_plot)+'-'+setup_type_root+'-fakeset-SN'+s2n_str+outfilpattern

    dictList = np.load(infil_npy, allow_pickle=True)
    data = dictList[0]
                
    # For continuum-normalization around NaI
    #blim = [5850.0,5870.0]
    #rlim = [5910.0,5930.0]
    #fitlim = [5880.0,5910.0]
    #inspectlim = [5850.,5930.]

    bins_for_loop = np.array(binid_plot)
    
    
    pl.figure(figsize=(6.0,5.0))
    pl.rcParams.update({'font.size': 14})
    
    
    flux_NaI = data['flux_fkline']
    err_NaI = data['err_fkline']
    restwave_NaI = data['wave']
                
    transinfo = model_NaI.transitions()
    vel_NaI = (restwave_NaI - transinfo['lamred0']) * sol / transinfo['lamred0']

    best_mod = model_NaI.model_NaI(theta_plot, data['velres'], restwave_NaI)
    #pdb.set_trace()
    
    ## PLAYING WITH MODEL PARAMETERS
    #lamred_guess = 5897.5581
    #logN_guess = 13.0
    #bD_guess = 150.0
    #Cf_guess = 0.2
    #theta_best = lamred_guess, logN_guess, bD_guess, Cf_guess
    #fitind = np.where((restwave_NaI > 5880.0) & (restwave_NaI < 5910.0))
    #best_mod1 = model_NaI.model_NaI(theta_best, avg_res, restwave_NaI[fitind])
    #logN_guess = 13.5
    #theta_best = lamred_guess, logN_guess, bD_guess, Cf_guess
    #best_mod2 = model_NaI.model_NaI(theta_best, avg_res, restwave_NaI[fitind])
    #logN_guess = 14.0
    #theta_best = lamred_guess, logN_guess, bD_guess, Cf_guess
    #best_mod3 = model_NaI.model_NaI(theta_best, avg_res, restwave_NaI[fitind])
                    
    #if(qq==2):
    #    pdb.set_trace()
    fig, ax = pl.subplots(1,1)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.0)
        
    ax.plot(restwave_NaI,flux_NaI,drawstyle='steps-mid', color="k")
    ax.plot(restwave_NaI,flux_NaI+err_NaI,drawstyle='steps-mid', color="gray", alpha=0.5)
    ax.plot(restwave_NaI,flux_NaI-err_NaI,drawstyle='steps-mid', color="gray", alpha=0.5)
    #ax.plot(restwave_NaI,smod_NaI, color="red", drawstyle='steps-mid')
    #ax.plot(restwave_NaI,cont_NaI, color='cyan')
    ax.plot(best_mod['modwv'], best_mod['modflx'], color="blue")
                        
    ax.set_xlabel(r'Rest Wavelength (Ang)')
    ax.set_ylabel(r'Flux')
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(pl.MultipleLocator(5.0))
    #ax.set_ylim(0.5,1.2)
                    
    vel_NaIblue = (transinfo['lamblu0'] - transinfo['lamred0']) * sol / transinfo['lamred0']

    ylim = ax.get_ylim()
    ax.plot([transinfo['lamblu0'],transinfo['lamblu0']], [ylim[0], ylim[1]], color="gray", ls='--')
    ax.plot([transinfo['lamred0'],transinfo['lamred0']], [ylim[0], ylim[1]], color="gray", ls='--')
    #ax.plot([blim[0],blim[0]], [1.1*avg_cont, 1.1*avg_cont + 0.02], color="blue")
    #ax.plot([blim[1],blim[1]], [1.1*avg_cont, 1.1*avg_cont + 0.02], color="blue")
    #ax.plot([rlim[0],rlim[0]], [1.1*avg_cont, 1.1*avg_cont + 0.02], color="red")
    #ax.plot([rlim[1],rlim[1]], [1.1*avg_cont, 1.1*avg_cont + 0.02], color="red")

    HeI = 5877.243
    #ax.plot([HeI, HeI], [1.1*avg_cont, 1.1*avg_cont + 0.02], color="purple")
    #ax.plot([vel50,vel50],[0.0,2.0], color="red")
    #ax.plot([vel16,vel16],[0.0,2.0], color="orange")
    #ax.plot([vel84,vel84],[0.0,2.0], color="orange")

    bin_str = "{:.0f}".format(binid_plot)
    ax.text(0.85,0.1,bin_str, ha='left', va='center', transform = ax.transAxes)

    text_str = "{:.2f}".format(theta_plot[0])+', '+"{:.2f}".format(theta_plot[1])+', '+\
        "{:.2f}".format(theta_plot[2])+', '+"{:.2f}".format(theta_plot[3])
    
    ax.text(0.05, 0.11, text_str, ha='left', va='center', transform=ax.transAxes, fontsize=11)
              
    fig.tight_layout()
    fig.savefig(outfil, format='pdf')

    


       

def main():

    s2n = 50.0
    setup_type = 'NoiseOnly'
    fakesetsdir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/FakeSets/NoiseOnly/Representative_logN14p4_16p0Test/'

    if(setup_type=='NoiseOnly'):
        setup_type_root = 'noiseonly'
    elif(setup_type=='AddNoise'):
        setup_type_root = 'addnoise'

    if(s2n>0.0):
        s2n_str = '{0:.0f}'.format(s2n)
    else:
        s2n_str = '*'

    fitpattern = '*'+setup_type_root+'-fakeset-SN'+s2n_str+'-logN14p4_16p0-wsamples.npy'

    findall_fit = []
    plate_fit = []
    ifu_fit = []
    binid_fit = []

    for root, dirs, files in os.walk(fakesetsdir):
        for filename in fnmatch.filter(files,fitpattern):
            findall_fit.append(filename)
    print("Found this many fake datasets with fits:", len(findall_fit))


    for nn in range(len(findall_fit)):
  
        nnfakeset = findall_fit[nn]
        nnfil = nnfakeset.split('-')
        found_plate = int(nnfil[0])
        found_ifu = int(nnfil[1])
        found_binid = int(nnfil[2])
        found_plateifubin = "{:.0f}".format(found_plate)+"-"+"{:.0f}".format(found_ifu)+"-"+"{:.0f}".format(found_binid)
        dictList = np.load(fakesetsdir+nnfakeset, allow_pickle=True)
        data = dictList[0]

        #pdb.set_trace()
        samples = data['samples']
        lamred_samples = samples[:,0]
        lamred50 = np.percentile(lamred_samples,50)
        logN_samples = samples[:,1]
        logN50 = np.percentile(logN_samples,50)
        Cf_samples = samples[:,3]
        Cf50 = np.percentile(Cf_samples,50)
        bD_samples = samples[:,2]
        bD50 = np.percentile(bD_samples,50)
        theta_best = lamred50, logN50, bD50, Cf50
        
        #pdb.set_trace()
        fig_NaImcmcfake_makecheckfit(plate_plot=found_plate, ifu_plot=found_ifu, binid_plot=found_binid, \
                                     setup_type=setup_type, s2n_str=s2n_str, fakesetsdir=fakesetsdir, \
                                     theta_plot=theta_best)


    #plate = 8257
    #ifu = 12701
    #binid = None
    #outdir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/Plots/'
    #fig_check_continuum(mcmcdirroot=None, overwrite=True, FIT_FLG=0, \
    #                    plate_plot=plate, ifu_plot=ifu, binid_plot=binid, NOFITINFO=True, outdir=outdir)
    
    # check bin 10


if __name__ == '__main__':

    main()
