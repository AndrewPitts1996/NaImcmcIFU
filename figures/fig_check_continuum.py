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


def fig_check_continuum(mcmcdirroot=None, overwrite=False, FIT_FLG=None, plate_plot=None, ifu_plot=None, binid_plot=None, NOFITINFO=False, outdir=None):

    sol = 2.998e5    # km/s 
    saspath = '/data/manga/sas/mangawork/'
    nperpage = 1
    
    if((mcmcdirroot==None) & (NOFITINFO==False)):
        mcmcdirroot = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/MCMC/SQUARE2.0/2020jul20/'
    pattern = '*samples*.fits'
    outpattern = '-check-continuum.pdf'

    if(NOFITINFO==False):
        if(FIT_FLG==0):
            mcmcdir = mcmcdirroot + 'FluxFit/'
        elif(FIT_FLG==1):
            mcmcdir = mcmcdirroot + 'SModFit/'
        elif(FIT_FLG==2):
            mcmcdir = mcmcdirroot + 'FluxNoSModFit/'

    # NEED TO IMPLEMENT THIS OPTION
    if((plate_plot==None) & (ifu_plot==None)):
        # Find all objects which have been fit
        allmcmc = []

        for root, dirs, files in os.walk(mcmcdir):
            for filename in fnmatch.filter(files,pattern):
                allmcmc.append(os.path.join(root,filename))

        print("Found this many MCMC fit outputs to plot:")
        print(len(allmcmc))
        n_allmcmc = len(allmcmc)
        
    else:
        #plate_str = "{:.0f}".format(plate_plot)
        #ifu_str = "{:.0f}".format(ifu_plot)

        #if(binid_plot != None):
        #    binid_str = "{:.0f}".format(binid_plot)
        #mcmcfits = mcmcdir
        n_allmcmc = 1
        
    for nn in range(n_allmcmc):

        if((plate_plot==None) & (ifu_plot==None)):
            nnmcmc = allmcmc[nn]
            nnfil = os.path.basename(nnmcmc)
            print(nnfil)
            nnfilsplit = nnfil.split('-')
            plate = int(nnfilsplit[0])
            ifu = int(nnfilsplit[1])

        else:
            plate = plate_plot
            ifu = ifu_plot
            
            if(binid_plot != None):
                binid_str = "{:.0f}".format(binid_plot)
                bins_for_loop = np.array([binid_plot])
                #pdb.set_trace()
            else:
                binid_str = 'all'

        #pdb.set_trace()   
        plate_str = "{:.0f}".format(plate)
        ifu_str = "{:.0f}".format(ifu)

        if(NOFITINFO==False):
            mcmcfits = mcmcdir+plate_str+"-"+ifu_str+"-samples.fits"
            mapfits = mcmcdir+plate_str+"-"+ifu_str+"-NaImaps.fits"

        if(outdir==None):
            outdir = mcmcdir+'Plots/'
            
        outfil = outdir+plate_str+"-"+ifu_str+"-"+binid_str+outpattern
      
        if(((os.path.exists(outfil)) & (overwrite==False)) | (plate_str+"-"+ifu_str=='9501-12705') | (plate_str+"-"+ifu_str=='8719-9101') | (plate_str+"-"+ifu_str=='8940-9101')):

            print("Skipping ", plate_str+"-"+ifu_str)

        else:
        
            dap = 'users/u6021943/Alt-DAP/SQUARE2.0_2020jan09/SQUARE2.0-MILESHC-MASTARHC/'+plate_str+'/'+ifu_str+'/manga-'+plate_str+'-'+ifu_str
            drp = 'manga/spectro/redux/MPL-9/'+plate_str+'/stack/manga-'+plate_str+'-'+ifu_str
            fits_dap_cube = saspath + dap + '-LOGCUBE-SQUARE2.0-MILESHC-MASTARHC.fits.gz'
            fits_dap_map = saspath + dap + '-MAPS-SQUARE2.0-MILESHC-MASTARHC.fits.gz'
            fits_drp = saspath + drp + '-LOGCUBE.fits.gz'
            
            cube = NaImcmc_read_fits.NaImcmc_read_fits(fits_dap_map, fits_dap_cube, fits_drp=fits_drp)

            wave = cube['wave']
            #sres = cube['sres']
            flux = cube['fluxb']
            ivar = cube['ivarb']
            smod = cube['modelb']
            predisp = cube['predispb']
            cz = cube['cz']
            stellar_vfield = cube['stellar_vfieldb']
            stellar_vfield_ivar = cube['stellar_vfield_ivarb']
            emission_vfield = cube['emission_vfieldb']
            emission_vfield_ivar = cube['emission_vfield_ivarb']
            binid_cube = cube['binidb']

            #pdb.set_trace()
            #elmod = dapf.hdu['ELOMFB'].data # emission-line fit
            #elfit = dapf.hdu['ELOFIT'].data
            #ha_ew = dapf.hdu['ELOFIT'].data['EW_FB'][:,5]
            #ha_ewerr = dapf.hdu['ELOFIT'].data['EWERR_FB'][:,5]
            #elkinfit = elfit.field('KIN_FB')
            #elkinerrfit = elfit.field('KINERR_FB')
            #nbins = len(flux[0])

            
            if(NOFITINFO==True):
                mapfits = fits_dap_map

            hdu_maps = fits.open(mapfits)
            ubins, uindx = DAPFitsUtil.unique_bins(hdu_maps['BINID'].data, return_index=True)
            binid_map = hdu_maps['BINID'].data.ravel()[uindx]
            if(binid_plot==None):
                bins_for_loop = binid_map
            
            if(NOFITINFO==False):
                mcmc = fits.getdata(mcmcfits, 1)
                tmcmc = Table(mcmc)
                vel50 = hdu_maps['NAIVEL50_MAP'].data.ravel()[uindx]
                vel05 = hdu_maps['NAIVEL05_MAP'].data.ravel()[uindx]
                vel16 = hdu_maps['NAIVEL16_MAP'].data.ravel()[uindx]
                vel84 = hdu_maps['NAIVEL84_MAP'].data.ravel()[uindx]
                vel95 = hdu_maps['NAIVEL95_MAP'].data.ravel()[uindx]
                elvel05 = hdu_maps['NAIELVEL05_MAP'].data.ravel()[uindx]
                elvel95 = hdu_maps['NAIELVEL95_MAP'].data.ravel()[uindx]
                elvel50 = hdu_maps['NAIELVEL50_MAP'].data.ravel()[uindx]
                logN50 = hdu_maps['NAILOGN50_MAP'].data.ravel()[uindx]
                Cf50 = hdu_maps['NAICF50_MAP'].data.ravel()[uindx]
                bD50 = hdu_maps['NAIBD50_MAP'].data.ravel()[uindx]
                bins_for_loop = tmcmc['bin']
            
            # For continuum-normalization around NaI
            blim = [5850.0,5870.0]
            rlim = [5910.0,5930.0]
            fitlim = [5880.0,5910.0]
            inspectlim = [5850.,5930.]
        
            pltct = 0
            with PdfPages(outfil) as pdf:
    
                #for qq in range(len(tmcmc['bin'])):
                for qq in range(len(bins_for_loop)):

                    if((qq % nperpage)==0):

                        # Plot lamred histogram, NaI profile, other test transitions
                        #pl.figure(figsize=(13.0,11.0))
                        pl.figure(figsize=(6.0,5.0))
                        pl.rcParams.update({'font.size': 14})
                        pltct = 0

                    
                    
                    print(plate, ifu)
                    #print(qq, tmcmc['bin'][qq])
                    #binid_bin = tmcmc['bin'][qq]
                    print(qq, bins_for_loop[qq])
                    binid_bin = bins_for_loop[qq]
                
                    ind = np.where(binid_cube==binid_bin)
                    ind = ind[0][0]
                
                    flux_bin = flux[:,ind]
                    smod_bin = smod[:,ind]
                    #elmod_bin = elmod[:,qq]
                    ivar_bin = ivar[:,ind]
                    err_bin = 1.0/np.sqrt(ivar_bin)
                    predisp_bin = predisp[:,ind]
            

                    # Determine bin redshift: cz in km/s = tstellar_kin[*,0]
                    # bin_z = (cz + stellar_vfield[ind]) / sol
                    cosmo_z = cz / sol
                    bin_z = cosmo_z + ((1 + cosmo_z) * stellar_vfield[ind] / sol) 
                    restwave = wave / (1.0 + bin_z)

                    # Other kinematic measurements
                    bin_stkinerr = 1.0/np.sqrt(stellar_vfield_ivar[ind])
                    #bin_elz = (cz + emission_vfield[ind]) / sol
                    bin_elz = cosmo_z + ((1 + cosmo_z) * emission_vfield[ind] / sol) 
                    bin_elkinerr = 1.0/np.sqrt(emission_vfield_ivar[ind])

                    ## FOR TESTING SMOD FITTING ONLY
                    #flux_bin = smod_bin
                                               
                    if((FIT_FLG==0) | (FIT_FLG==1)):
                        #ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, blim, rlim)
                        ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, \
                                         blim, rlim, FIT_FLG=1, smod=smod_bin)

                    elif(FIT_FLG==2):
                        ndata = continuum_normalize_NaI.smod_norm(restwave, flux_bin, err_bin, smod_bin)
    
                    # Cut out NaI
                    naiind = np.where((restwave > inspectlim[0]) & (restwave < inspectlim[1]))
                    restwave_NaI = restwave[naiind]
                    flux_NaI = flux_bin[naiind]
                    err_NaI = err_bin[naiind]
                    #smod_NaI = nsmod['nflux'][ind]
                    #smod_norm = smod_bin / ndata['cont']
                    #sres_NaI = sres[naiind]
                    predisp_NaI = predisp[naiind]
                    wave_NaI = wave[naiind]
                    smod_NaI = smod_bin[naiind]
                    cont_NaI = ndata['cont'][naiind]
                    transinfo = model_NaI.transitions()
                    vel_NaI = (restwave_NaI - transinfo['lamred0']) * sol / transinfo['lamred0']

                    avg_cont = np.mean(ndata['cont'][naiind])
                    #avg_res = sol / np.mean(sres_NaI)
                    avg_res = sol * 2.355 * np.mean(predisp_NaI)/np.mean(wave_NaI)
                    
                    
                    shiftvel_NaI = vel_NaI[1:]
                    dvel = shiftvel_NaI - vel_NaI[:-1]

                    if(NOFITINFO==False):
                        samples = tmcmc['samples'][qq]
                        percentiles_mcmc = tmcmc['percentiles'][qq]
                        lamred_mcmc = percentiles_mcmc[0,:]
                   
                        #lamred05 = np.percentile(lamred_samples,5)
                        #lamred95 = np.percentile(lamred_samples,95)

                        # Overplot median model
                        ind_map = np.where(binid_map==binid_bin)
                        ind_map = ind_map[0][0]
                        theta_best = lamred_mcmc[0], logN50[ind_map], bD50[ind_map], Cf50[ind_map]

                        fitind = np.where((restwave_NaI > 5880.0) & (restwave_NaI < 5910.0))
                        best_mod = model_NaI.model_NaI(theta_best, avg_res, restwave_NaI[fitind])

                    ## PLAYING WITH MODEL PARAMETERS
                    lamred_guess = 5897.5581
                    logN_guess = 13.5
                    bD_guess = 60.0
                    Cf_guess = 0.2
                    theta_best = lamred_guess, logN_guess, bD_guess, Cf_guess
                    fitind = np.where((restwave_NaI > 5880.0) & (restwave_NaI < 5910.0))
                    best_mod1 = model_NaI.model_NaI(theta_best, avg_res, restwave_NaI[fitind])
                    logN_guess = 14.0
                    theta_best = lamred_guess, logN_guess, bD_guess, Cf_guess
                    best_mod2 = model_NaI.model_NaI(theta_best, avg_res, restwave_NaI[fitind])
                    logN_guess = 14.4
                    theta_best = lamred_guess, logN_guess, bD_guess, Cf_guess
                    best_mod3 = model_NaI.model_NaI(theta_best, avg_res, restwave_NaI[fitind])
                    
                    #if(qq==2):
                    #    pdb.set_trace()
                    ax = pl.subplot(1,1,pltct+1)
                    for axis in ['top','bottom','left','right']:
                        ax.spines[axis].set_linewidth(2.0)
        
                    ax.plot(restwave_NaI,flux_NaI,drawstyle='steps-mid', color="k")
                    ax.plot(restwave_NaI,flux_NaI+err_NaI,drawstyle='steps-mid', color="gray", alpha=0.5)
                    ax.plot(restwave_NaI,flux_NaI-err_NaI,drawstyle='steps-mid', color="gray", alpha=0.5)
                    ax.plot(restwave_NaI,smod_NaI, color="red", drawstyle='steps-mid')
                    ax.plot(restwave_NaI,cont_NaI, color='cyan')
                    ax.plot(best_mod1['modwv'], cont_NaI[fitind] * best_mod1['modflx'], color="blue")
                    ax.plot(best_mod2['modwv'], cont_NaI[fitind] * best_mod2['modflx'], color="indigo")
                    ax.plot(best_mod3['modwv'], cont_NaI[fitind] * best_mod3['modflx'], color="violet")
                    
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
                    ax.plot([blim[0],blim[0]], [1.1*avg_cont, 1.1*avg_cont + 0.02], color="blue")
                    ax.plot([blim[1],blim[1]], [1.1*avg_cont, 1.1*avg_cont + 0.02], color="blue")
                    ax.plot([rlim[0],rlim[0]], [1.1*avg_cont, 1.1*avg_cont + 0.02], color="red")
                    ax.plot([rlim[1],rlim[1]], [1.1*avg_cont, 1.1*avg_cont + 0.02], color="red")

                    HeI = 5877.243
                    ax.plot([HeI, HeI], [1.1*avg_cont, 1.1*avg_cont + 0.02], color="purple")
                    #ax.plot([vel50,vel50],[0.0,2.0], color="red")
                    #ax.plot([vel16,vel16],[0.0,2.0], color="orange")
                    #ax.plot([vel84,vel84],[0.0,2.0], color="orange")

                    bin_str = "{:.0f}".format(binid_bin)
                    ax.text(0.85,0.1,bin_str, ha='left', va='center', transform = ax.transAxes)

          
                    #if(binid_bin==4):
                    #    pdb.set_trace()
                    #pdb.set_trace()
                    if(NOFITINFO==False):
                        vel_str = "{:.1f}".format(vel05[ind_map])+","+"{:.1f}".format(vel50[ind_map])+\
                            ","+"{:.1f}".format(vel95[ind_map])+","+\
                            "{:.1f}".format(vel84[ind_map]-vel16[ind_map])
                        ax.text(0.05, 0.11, vel_str, ha='left', va='center', transform=ax.transAxes, fontsize=11)
                        elvel_str = "{:.1f}".format(elvel05[ind_map])+","+"{:.1f}".format(elvel50[ind_map])+\
                            ","+"{:.1f}".format(elvel95[ind_map])
                        ax.text(0.05, 0.05, elvel_str, ha='left', va='center', transform=ax.transAxes, fontsize=11)

                    
                        if((vel95[ind_map] < 0.0) & (elvel95[ind_map] < 0.0) &
                           (np.fabs(vel50[ind_map]) > 40.0) &
                           ((vel84[ind_map]-vel16[ind_map])<200.0)):
                            ax.text(0.05,0.17,'OUTFLOW',ha='left',va='center',
                                    transform = ax.transAxes,color="blue")
                    
                        if((vel05[ind_map] > 0.0) & (elvel05[ind_map] > 0.0) &
                           (np.fabs(vel50[ind_map]) > 40.0) &
                           ((vel84[ind_map]-vel16[ind_map])<200.0)):
                            ax.text(0.05,0.17,'INFLOW',ha='left',va='center',
                                    transform = ax.transAxes,color="red")
                
                    if(pltct==nperpage-1):
          
                        pl.tight_layout()
                        pdf.savefig()

                    pltct = pltct+1


       

def main():

    # FIT_FLG=0  : plot standard fits
    # FIT_FLG=1  : plot SModFits
    # FIT_FLG=2  : plot stellar-model-normalized fits
    #fig_check_continuum(mcmcdirroot=None, overwrite=True, FIT_FLG=0)


    plate = 8257
    ifu = 12701
    binid = None
    outdir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/Plots/'
    fig_check_continuum(mcmcdirroot=None, overwrite=True, FIT_FLG=0, \
                        plate_plot=plate, ifu_plot=ifu, binid_plot=binid, NOFITINFO=True, outdir=outdir)
    
    # check bin 10


if __name__ == '__main__':

    main()
