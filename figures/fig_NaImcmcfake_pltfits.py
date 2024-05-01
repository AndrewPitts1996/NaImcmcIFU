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
import json
import pdb

import matplotlib.pyplot as pl
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
pl.rcParams['font.family'] = 'stixgeneral'
pl.rcParams['mathtext.fontset'] = 'stix'
import model_NaI

def fig_NaImcmcfake_pltfits(s2n=None, setup_type=None):


    if(s2n==None):
        print("Need to request a S/N level")
        pdb.set_trace()

    if(setup_type=='NoiseOnly'):
        setup_type_root = 'noiseonly'
    elif(setup_type=='AddNoise'):
        setup_type_root = 'addnoise'
    else:
        print('Setup type not defined!')


    #galinfo_fits = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/NaImcmcfake_galselect_binselect.fits'
    galinfo_fits = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/NaImcmcfake_galselect_binselect_SNmin30.fits'
    #galinfo_fits = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/NaImcmcfake_inflowbins.fits'
    galinfo = fits.getdata(galinfo_fits)

    if(s2n>0.0):
        s2n_str = '{0:.0f}'.format(s2n)
    else:
        s2n_str = '*'
    
    #fakesetsdir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/FakeSets/'+setup_type+'/Detected_Inflows/'
    fakesetsdir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/FakeSets/'+setup_type+'/Representative/'
    
    fitpattern = '*'+setup_type_root+'-fakeset-SN'+s2n_str+'-wsamples.npy'


    findall_fit = []
    plate_fit = []
    ifu_fit = []
    binid_fit = []

    for root, dirs, files in os.walk(fakesetsdir):
        for filename in fnmatch.filter(files,fitpattern):
            findall_fit.append(filename)
    print("Found this many fake datasets with fits:", len(findall_fit))

    fit_fils = []
    gal_StellarMass = []#np.zeros(len(findall_fit))
    gal_Inclination = []#np.zeros(len(findall_fit))
    gal_logSFR = []#np.zeros(len(findall_fit))
    gal_VelDisp = []#np.zeros(len(findall_fit))
    gal_EWNaI = []#np.zeros(len(findall_fit))
    
    frac_out = []#np.zeros(len(findall_fit))
    frac_in = []#np.zeros(len(findall_fit))
    
    vflow_consmin = 0.0
    #vflow_inmin = 55.0 
    #vflow_outmin = 50.0
    vflow_inmin = 50.0
    vflow_outmin = 40.0


    
    for nn in range(len(findall_fit)):
    #for nn in range(0,10):

        nnfakeset = findall_fit[nn]
        nnfil = nnfakeset.split('-')
        found_plate = int(nnfil[0])
        found_ifu = int(nnfil[1])
        found_binid = int(nnfil[2])
        found_plateifubin = "{:.0f}".format(found_plate)+"-"+"{:.0f}".format(found_ifu)+"-"+"{:.0f}".format(found_binid)
        

        # Match to get galaxy info
        wh = np.where((galinfo['PLATE']==found_plate) & \
                      (galinfo['IFU']==found_ifu) & \
                      (galinfo['BINID']==found_binid))

        print(wh[0])
        #pdb.set_trace()
        if(len(wh[0] > 0)):
            print("This is included")
            fit_fils.append(findall_fit[nn])
            plate_fit.append(found_plate)
            ifu_fit.append(found_ifu)
            binid_fit.append(found_binid)
            
            gal_StellarMass.append(galinfo['STELLAR_MASS'][wh[0][0]])
            gal_Inclination.append(galinfo['INCLINATION'][wh[0][0]])
            gal_logSFR.append(galinfo['LOGSFR'][wh[0][0]])
            gal_VelDisp.append(galinfo['VELDISP'][wh[0][0]])
            gal_EWNaI.append(galinfo['EWNAI'][wh[0][0]])
            print("Matched to ", galinfo['PLATE'][wh[0]], galinfo['IFU'][wh[0]], galinfo['BINID'][wh[0]])
        
            print("Plotting ", nnfakeset)
            #plate = 79
            #ifu = 6102
            #binid_for_fit = 26

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

        
            offset = -0.5

            for qq in range(nsims):
            #for qq in range(0,10):

                data = dictList[qq]

                vsim[qq] = data['v']
                Cfsim[qq] = data['Cf']
                logNsim[qq] = data['logN']

                samples = data['samples']
                #percentiles_mcmc = data['theta_percentiles']
                #lamred_mcmc = percentiles_mcmc[0,:]
                #logN_mcmc = percentiles_mcmc[1,:]
                #bD_mcmc = percentiles_mcmc[2,:]
                #Cf_mcmc = percentiles_mcmc[3,:]

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

                #if(logNsim[qq] == 11.5):
                #    symcolor='red'
                #elif(logNsim[qq] == 12.0):
                #    symcolor='orange'
                #elif(logNsim[qq] == 12.5):
                #    symcolor='green'
                #elif(logNsim[qq] == 13.0):
                #    symcolor='cyan'
                #elif(logNsim[qq] == 13.5):
                symcolor='blue'

                #if(Cfsim[qq] == 0.25):
                symsz = 1
                #elif(Cfsim[qq] == 0.5):
                #    symsz = 3
                #elif(Cfsim[qq] == 0.75):
                #    symsz = 5
                #elif(Cfsim[qq] == 1.0):
                #    symsz = 7

                #ax.plot([nn+offset], [vel50[qq]], color='black', linestyle="none", marker="s")
                #ax.plot([nn+offset], [vel99[qq]], color='red', linestyle="none", marker="s")
                #ax.plot([nn+offset], [vel01[qq]], color='blue', linestyle="none", marker="s")
            
            
                #pl.errorbar([nn], [vel50[qq]], yerr=[vel99[qq]-vel50[qq]], marker="s",
                #            linestyle="none", lolims=True, capsize=0, color=symcolor,\
                    #            ecolor=symcolor, markersize=symsz,markeredgewidth=1, alpha=0.5)
                #pl.errorbar([nn], [vel50[qq]], yerr=[vel50[qq]-vel01[qq]], marker="s",
                #            linestyle="none", uplims=True, capsize=0, color=symcolor, \
                    #            ecolor=symcolor, markersize=symsz,markeredgewidth=1, alpha=0.5)
                
            if(offset > 0.5):
                offset = -0.5

            ## Find significant flows
            ind_outflow3sig = np.where((vel99 < -1.0*vflow_consmin) & (np.fabs(vel50) > vflow_outmin) &
                                       ((vel84-vel16) < 200.0))
            ind_inflow3sig = np.where((vel01 > vflow_consmin) & (np.fabs(vel50) > vflow_inmin) &
                                      ((vel84-vel16) < 200.0))
            frac_out.append(len(ind_outflow3sig[0]))
            frac_in.append(len(ind_inflow3sig[0]))

    gal_VelDisp = np.array(gal_VelDisp)
    gal_EWNaI = np.array(gal_EWNaI)
    gal_StellarMass = np.array(gal_StellarMass)
    gal_Inclination = np.array(gal_Inclination)
    
    frac_out = np.array(frac_out)
    frac_in = np.array(frac_in)
            
    # Set up plotting
    #outfil= 'fig_NaImcmcfake_pltfits_DetectedInflows.pdf'
    outfil = 'fig_NaImcmcfake_pltfits_SNmin30.pdf'
    pl.figure(figsize=(8,5))
    pl.rcParams.update({'font.size': 15})

    xlim = [0,260]
    ylim = [-1,30]

    ax = pl.subplot(221)
    #for axis in ['top','bottom','left','right']:
    #    ax.spines[axis].set_linewidth(2.0)
    #ax.tick_params(which='major', axis='both', width=1.5, length=4, direction='in', top=True, right=True)
    #ax.tick_params(which='minor', axis='both', direction='in', top=True, right=True)
    ax.set_xlim(xlim)
    ax.set_xlabel("Bin Velocity Dispersion (km/s)")
    ax.plot(gal_VelDisp, frac_out, color='blue', linestyle="none", marker="s", markersize=2)
    ax.plot(gal_VelDisp, frac_in, color='red', linestyle="none", marker="s", markersize=2, alpha=0.5)

    #xlim = [0,2]
    #ax.set_xlabel("log Galaxy SFR")
    #ax.plot(gal_logSFR, frac_out, color='blue', linestyle="none", marker="s", markersize=2)
    #ax.plot(gal_logSFR, frac_in, color='red', linestyle="none", marker="s", markersize=2, alpha=0.5)
    
    
    ax = pl.subplot(222)
    xlim = [-2,10]
    ax.set_xlim(xlim)
    ax.set_xlabel("Stellar EW(NaI) (Ang)")
    ax.plot(gal_EWNaI, frac_out, color='blue', linestyle="none", marker="s", markersize=2)
    ax.plot(gal_EWNaI, frac_in, color='red', linestyle="none", marker="s", markersize=2, alpha=0.5)

    ax = pl.subplot(223)
    xlim = [8,12]
    ax.set_xlim(xlim)
    ax.set_xlabel("log Galaxy Stellar Mass")
    ax.plot(gal_StellarMass, frac_out, color='blue', linestyle="none", marker="s", markersize=2)
    ax.plot(gal_StellarMass, frac_in, color='red', linestyle="none", marker="s", markersize=2, alpha=0.5)

    ax = pl.subplot(224)
    xlim = [0,90]
    ax.set_xlim(xlim)
    ax.set_xlabel("Inclination")
    ax.plot(gal_Inclination, frac_out, color='blue', linestyle="none", marker="s", markersize=2)
    ax.plot(gal_Inclination, frac_in, color='red', linestyle="none", marker="s", markersize=2, alpha=0.5) 
   

    #ax.plot(np.arange(-120,120,1), 0.0*np.arange(-120,120,1))
    #ax.plot(np.arange(-120,120,1), 30.0+0.0*np.arange(-120,120,1), color='gray')
    #ax.plot(np.arange(-120,120,1), -30.0+0.0*np.arange(-120,120,1), color='gray')
    #ax.plot(np.arange(-120,120,1), 60.0+0.0*np.arange(-120,120,1), color='gray')
    #ax.plot(np.arange(-120,120,1), -60.0+0.0*np.arange(-120,120,1), color='gray') 
    pl.tight_layout()
    pl.savefig(outfil, format='pdf')

    pdb.set_trace()
    
    outfil= 'fig_NaImcmcfake_pltfits_veldispEW_DetectedInflows.pdf'
    pl.figure(figsize=(8,5))
    pl.rcParams.update({'font.size': 15})

    xlim = [0,260]
    ylim = [-1,80]

    ax = pl.subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.0)
    ax.tick_params(which='major', axis='both', width=1.5, length=4, direction='in', top=True, right=True)
    ax.tick_params(which='minor', axis='both', direction='in', top=True, right=True)
    #ax.yaxis.set_major_locator(MaxNLocator(5))
    #ax.xaxis.set_minor_locator(pl.MultipleLocator(0.1))
    #ax.yaxis.set_minor_locator(pl.MultipleLocator(10))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    wh = np.where(gal_EWNaI < 1.5)
    ax.plot(gal_VelDisp[wh], frac_out[wh], color='blue', linestyle="none", marker="s", markersize=2)
    ax.plot(gal_VelDisp[wh], frac_in[wh], color='red', linestyle="none", marker="s", markersize=2, alpha=0.5)

    wh = np.where(gal_EWNaI > 3.5)
    ax.plot(gal_VelDisp[wh], frac_out[wh], color='blue', linestyle="none", marker="o", markersize=4)
    ax.plot(gal_VelDisp[wh], frac_in[wh], color='red', linestyle="none", marker="o", markersize=4, alpha=0.5)

    pl.tight_layout()
    pl.savefig(outfil, format='pdf')

    
    pdb.set_trace()


def main():


    #s2n = 30.0
    s2n = -1.0
    setup_type = 'NoiseOnly'

    fig_NaImcmcfake_pltfits(s2n=s2n, setup_type=setup_type)

main()
