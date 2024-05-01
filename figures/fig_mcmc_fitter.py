from __future__ import print_function

import math
import numpy as np
import scipy.optimize as op
import os
import fnmatch
import matplotlib
matplotlib.use('Agg')
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


def fig_mcmc_fitter(mcmcdirroot=None, overwrite=False, FIT_FLG=None):

    sol = 2.998e5    # km/s 
    saspath = '/data/home/krubin/sas/mangawork/'
    
    if(mcmcdirroot==None):
        mcmcdirroot = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/MCMC/SQUARE2.0/'
    pattern = '*samples*.fits'
    outpattern = '-fit-inspect.pdf'

    if(FIT_FLG==None):
        mcmcdir = mcmcdirroot + 'FluxFit/'
    elif(FIT_FLG==1):
        mcmcdir = mcmcdirroot + 'SModFit/'
    elif(FIT_FLG==2):
        mcmcdir = mcmcdirroot + 'FluxNoSModFit/'

    
    # Find all objects which have been fit
    allmcmc = []

    for root, dirs, files in os.walk(mcmcdir):
        for filename in fnmatch.filter(files,pattern):
            allmcmc.append(os.path.join(root,filename))

    print("Found this many MCMC fit outputs to plot:")
    print(len(allmcmc))

    for nn in range(len(allmcmc)):

        nnmcmc = allmcmc[nn]
        nnfil = os.path.basename(nnmcmc)
        print(nnfil)
        nnfilsplit = nnfil.split('-')
        plate = int(nnfilsplit[0])
        ifu = int(nnfilsplit[1])

        plate_str = "{:.0f}".format(plate)
        ifu_str = "{:.0f}".format(ifu)
        mcmcfits = mcmcdir+plate_str+"-"+ifu_str+"-samples.fits"
        outfil = mcmcdir+'Plots/'+plate_str+"-"+ifu_str+outpattern

        
        if(((os.path.exists(outfil)) & (overwrite==False)) | (plate_str+"-"+ifu_str=='9501-12705') | (plate_str+"-"+ifu_str=='8719-9101') | (plate_str+"-"+ifu_str=='8940-9101')):

            print("Skipping ", plate_str+"-"+ifu_str)

        else:
        
            #plate = 7443
            #ifu = 12704

            dap = 'users/u6021943/Alt-DAP/SQUARE2.0_2018oct25/SQUARE2.0-GAU-MILESHC/'+plate_str+'/'+ifu_str+'/manga-'+plate_str+'-'+ifu_str
            fits_dap_cube = saspath + dap + '-LOGCUBE-SQUARE2.0-GAU-MILESHC.fits.gz'
            fits_dap_map = saspath + dap + '-MAPS-SQUARE2.0-GAU-MILESHC.fits.gz'
            cube = NaImcmc_read_fits.NaImcmc_read_fits(fits_dap_map, fits_dap_cube)

            wave = cube['wave']
            flux = cube['fluxb']
            ivar = cube['ivarb']
            smod = cube['modelb']
            cz = cube['cz']
            stellar_vfield = cube['stellar_vfieldb']
            stellar_vfield_ivar = cube['stellar_vfield_ivarb']
            emission_vfield = cube['emission_vfieldb']
            emission_vfield_ivar = cube['emission_vfield_ivarb']
            binid = cube['binidb']
    
            #elmod = dapf.hdu['ELOMFB'].data # emission-line fit
            #elfit = dapf.hdu['ELOFIT'].data
            #ha_ew = dapf.hdu['ELOFIT'].data['EW_FB'][:,5]
            #ha_ewerr = dapf.hdu['ELOFIT'].data['EWERR_FB'][:,5]
            #elkinfit = elfit.field('KIN_FB')
            #elkinerrfit = elfit.field('KINERR_FB')
            #nbins = len(flux[0])


            mcmc = fits.getdata(mcmcfits, 1)
            tmcmc = Table(mcmc)
            
            
            # For continuum-normalization around NaI
            blim = [5850.0,5860.0]
            rlim = [5910.0,5920.0]
            fitlim = [5880.0,5910.0]

            ewNaI = ew_NaI_allspax.ew(fits_dap_map, fits_dap_cube, blim, rlim)
        
    
            with PdfPages(outfil) as pdf:
    
                for qq in range(len(tmcmc['bin'])):

                    print(plate, ifu)
                    print(qq, tmcmc['bin'][qq])
                    binid_bin = tmcmc['bin'][qq]
                
                
                    ind = np.where(binid==binid_bin)
                    ind = ind[0][0]
                
                    flux_bin = flux[:,ind]
                    smod_bin = smod[:,ind]
                    #elmod_bin = elmod[:,qq]
                    ivar_bin = ivar[:,ind]
                    err_bin = 1.0/np.sqrt(ivar_bin)
            

                    # Determine bin redshift: cz in km/s = tstellar_kin[*,0]
                    bin_z = (cz + stellar_vfield[ind]) / sol
                    restwave = wave / (1.0 + bin_z)

                    # Other kinematic measurements
                    bin_stkinerr = 1.0/np.sqrt(stellar_vfield_ivar[ind])
                    bin_elz = (cz + emission_vfield[ind]) / sol
                    bin_elkinerr = 1.0/np.sqrt(emission_vfield_ivar[ind])

                    #restwave_el = wave / (1.0 + bin_elz)
                    #HaSN = ha_ew[qq] / ha_ewerr[qq]

                    #pdb.set_trace()

                    if((FIT_FLG==0) | (FIT_FLG==1)):
                        ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, blim, rlim)

                    elif(FIT_FLG==2):
                        ndata = continuum_normalize_NaI.smod_norm(restwave, flux_bin, err_bin, smod_bin)
    
                    # Cut out NaI
                    naiind = np.where((ndata['nwave'] > fitlim[0]) & (ndata['nwave'] < fitlim[1]))
                    restwave_NaI = ndata['nwave'][naiind]
                    flux_NaI = ndata['nflux'][naiind]
                    err_NaI = ndata['nerr'][naiind]
                    #smod_NaI = nsmod['nflux'][ind]
                    smod_norm = smod_bin / ndata['cont']
                    smod_NaI = smod_norm[naiind]
                    transinfo = model_NaI.transitions()
                    vel_NaI = (restwave_NaI - transinfo['lamred0']) * sol / transinfo['lamred0']

                    #print(vel_NaI[:-1])
                    shiftvel_NaI = vel_NaI[1:]
                    #print(shiftvel_NaI)
                    dvel = shiftvel_NaI - vel_NaI[:-1]
                    print(np.median(dvel))


                    #pdb.set_trace()
                    # Calculate velocities from fit
                    #idx = np.where(tmcmc['bin'] == qq)
                    samples = tmcmc['samples'][qq]
                    percentiles_mcmc = tmcmc['percentiles'][qq]
                    lamred_mcmc = percentiles_mcmc[0,:]

                    vel50 = (lamred_mcmc[0] - transinfo['lamred0']) * sol / transinfo['lamred0']
                    vel16 = ((lamred_mcmc[0] - lamred_mcmc[2]) - transinfo['lamred0']) * sol / transinfo['lamred0']
                    vel84 = (lamred_mcmc[0] + lamred_mcmc[1] - transinfo['lamred0']) * sol / transinfo['lamred0']

                    # Calculate velocities relative to emission lines
                    obslamred_mcmc = lamred_mcmc * (1.0 + bin_z)
                    elvel50 = (obslamred_mcmc[0] - ((1.0 + bin_elz)*transinfo['lamred0'])) * sol / ((1.0 + bin_elz)*transinfo['lamred0'])
                    elvel16 = ((obslamred_mcmc[0] - obslamred_mcmc[2]) - ((1.0 + bin_elz)*transinfo['lamred0'])) * sol / ((1.0 + bin_elz)*transinfo['lamred0'])
                    elvel84 = (obslamred_mcmc[0] + obslamred_mcmc[1] - ((1.0 + bin_elz)*transinfo['lamred0'])) * sol / ((1.0 + bin_elz)*transinfo['lamred0'])

                    # Calculate 95th-percentile limits
                    lamred_samples = samples[:,0]
                    lamred05 = np.percentile(lamred_samples,5)
                    lamred95 = np.percentile(lamred_samples,95)

            
                    # Plot lamred histogram, NaI profile, other test transitions
                    pl.figure(figsize=(13.0,8.0))
                    pl.rcParams.update({'font.size': 14})

                    ax = pl.subplot(2,4,1)
                    for axis in ['top','bottom','left','right']:
                        ax.spines[axis].set_linewidth(2.0)
        
                    ax.plot(vel_NaI,flux_NaI,drawstyle='steps-mid', color="k")
                    ax.plot(vel_NaI,flux_NaI+err_NaI,drawstyle='steps-mid', color="gray")
                    ax.plot(vel_NaI,flux_NaI-err_NaI,drawstyle='steps-mid', color="gray")
                    ax.plot(vel_NaI,smod_NaI, color="red", drawstyle='steps-mid')
                    ax.set_xlabel(r'Relative Velocity (km/s)')
                    ax.set_ylabel(r'Normalized Flux')
                    ax.xaxis.set_major_locator(MaxNLocator(5))
                    ax.set_ylim(0.5,1.2)
                    
                    vel_NaIblue = (transinfo['lamblu0'] - transinfo['lamred0']) * sol / transinfo['lamred0']
                    ax.plot([vel_NaIblue,vel_NaIblue], [0.0,2.0], color="gray")
                    ax.plot([0.0,0.0], [0.0,2.0], color="gray")
                    ax.plot([vel50,vel50],[0.0,2.0], color="red")
                    ax.plot([vel16,vel16],[0.0,2.0], color="orange")
                    ax.plot([vel84,vel84],[0.0,2.0], color="orange")
                    
                    if(transinfo['lamred0'] > lamred95):
                        ax.text(0.2,0.1,'OUTFLOW',ha='left',va='center',transform = ax.transAxes,color="blue")

                    if((1.0 + bin_elz)*transinfo['lamred0'] > (1.0 + bin_z)*lamred95):
                        ax.text(0.2,0.05,'ELOUTFLOW',ha='left',va='center',transform = ax.transAxes,color="blue")
                
                    if(transinfo['lamred0'] < lamred05):
                        ax.text(0.2,0.1,'INFLOW',ha='left',va='center',transform = ax.transAxes,color="red")

                    if((1.0 + bin_elz)*transinfo['lamred0'] < (1.0 + bin_z)*lamred05):
                        ax.text(0.2,0.05,'ELINFLOW',ha='left',va='center',transform = ax.transAxes,color="red")

                    # Need to find correct bins for EW
                    ind_ew = np.where(ewNaI['binid']==binid_bin)
                    ind_ew = ind_ew[0][0]
                    ew_str = "{:.2f}".format(ewNaI['obsew'][ind_ew])
                    sigew_str = "{:.2f}".format(ewNaI['sigobsew'][ind_ew])
                    modew_str = "{:.2f}".format(ewNaI['modew'][ind_ew])
                    ax.text(0.2,0.9, ew_str+'+/-'+sigew_str,ha='center',va='center',transform = ax.transAxes,fontsize=14)
                    ax.text(0.2,0.8, modew_str,ha='center',va='center',transform = ax.transAxes,fontsize=14,color="red")

                    #pdb.set_trace()
                    ax = pl.subplot(2,4,2)
                    for axis in ['top','bottom','left','right']:
                        ax.spines[axis].set_linewidth(2.0)

                    bin_sz = 0.02
                    lbins = np.arange(min(lamred_samples), max(lamred_samples) + bin_sz, bin_sz)
                    pl.hist(lamred_samples,bins=lbins,histtype='step',normed=1, color="k")
                    ax.set_ylim(0.0,3.0)
                    ax.plot([transinfo['lamred0'],transinfo['lamred0']], [0.0,5.0], color="gray", lw=2)
                    ax.plot([lamred_mcmc[0], lamred_mcmc[0]], [0.0,5.0], color="red")
                    ax.plot([lamred_mcmc[0]-lamred_mcmc[2], lamred_mcmc[0]-lamred_mcmc[2]], [0.0,5.0], color="orange")
                    ax.plot([lamred_mcmc[0]+lamred_mcmc[1], lamred_mcmc[0]+lamred_mcmc[1]], [0.0,5.0], color="orange")

                    vel_str = "{:.1f}".format(vel50)
                    pltnote = vel_str + ' km/s'
                    ax.text(0.8,0.9,pltnote,ha='center',va='center',transform = ax.transAxes,fontsize=14)
                    vel_str = "{:.1f}".format(elvel50)
                    pltnote = vel_str + ' km/s'
                    ax.text(0.8,0.8,pltnote,ha='center',va='center',transform = ax.transAxes,fontsize=14,color='blue')

                    ############################
                    ## Plot CaII K
                    ax = pl.subplot(2,4,3)
                    for axis in ['top','bottom','left','right']:
                        ax.spines[axis].set_linewidth(2.0)
        
                    ax.plot(restwave,flux_bin,drawstyle='steps-mid', color="k")
                    ax.plot(restwave,smod_bin, color="red")
                    #ax.plot(restwave,elmod_bin, color="blue")
                    ax.set_xlabel(r'Rest Wavelength (Ang)')
                    ax.set_ylabel(r'Flux')
                    ax.xaxis.set_major_locator(MaxNLocator(5))
                    ax.set_xlim(3919.0,3949.0)

                    lamCaK = 3934.79
                    lamCaH = 3969.61
      
                    ax.plot([lamCaK, lamCaK], [-0.1,5.0], color="gray")
                    ax.plot([lamCaH, lamCaH], [0.0,5.0], color="gray")
                    caiiind = np.where((restwave > 3919.0) & (restwave < 3949.0))
                    ax.set_ylim(-0.1,0.2+max(flux_bin[caiiind]))

                    # Now plot CaK in rest-frame of emission line fit
                    el_lamCaK = lamCaK * (1.0 + bin_elz) / (1.0 + bin_z) 
                    ax.plot([el_lamCaK, el_lamCaK], [-0.1,5.0], color="blue")
                    stzerr_str = "{:.1f}".format(bin_stkinerr)
                    elzerr_str = "{:.1f}".format(bin_elkinerr)
                    pltnote = 'Stellar vel err = '+stzerr_str+' km/s'
                    ax.text(0.1,0.1,pltnote,ha='left',va='center',transform = ax.transAxes,fontsize=14,color="gray")

                    pltnote = 'EL vel err = '+elzerr_str+' km/s'
                    ax.text(0.1,0.2,pltnote,ha='left',va='center',transform = ax.transAxes,fontsize=14,color="blue")
                    
                    ax.text(0.7,0.9,'CaII K',ha='left',va='center',transform = ax.transAxes,fontsize=14,color="k")

                    ############################
                    ## Plot Hdelta
                    ax = pl.subplot(2,4,4)
                    for axis in ['top','bottom','left','right']:
                        ax.spines[axis].set_linewidth(2.0)
        
                    ax.plot(restwave,flux_bin,drawstyle='steps-mid', color="k")
                    ax.plot(restwave,smod_bin, color="red")
                    #ax.plot(restwave,elmod_bin, color="blue")
                    ax.set_xlabel(r'Rest Wavelength (Ang)')
                    ax.set_ylabel(r'Flux')
                    ax.xaxis.set_major_locator(MaxNLocator(5))
                    ax.set_xlim(4087.0,4120.0)

                    lamHd = 4102.90
      
                    ax.plot([lamHd, lamHd], [-0.1,5.0], color="gray")
                    hdind = np.where((restwave > 4087.0) & (restwave < 4120.0))
                    ax.set_ylim(-0.1,0.2+max(flux_bin[hdind]))

                    # Now plot Hd in rest-frame of emission line fit
                    el_lamHd = lamHd * (1.0 + bin_elz) / (1.0 + bin_z) 
                    ax.plot([el_lamHd, el_lamHd], [-0.1,5.0], color="blue")
                    ax.text(0.7,0.9,'Hd',ha='left',va='center',transform = ax.transAxes,fontsize=14,color="k")

                
                    ############################
                    ## Plot [OIII] 5007
                    ax = pl.subplot(2,4,5)
                    for axis in ['top','bottom','left','right']:
                        ax.spines[axis].set_linewidth(2.0)
        
                    ax.plot(restwave,flux_bin,drawstyle='steps-mid', color="k")
                    ax.plot(restwave,smod_bin, color="red")
                    #ax.plot(restwave,elmod_bin, color="blue")
                    ax.set_xlabel(r'Rest Wavelength (Ang)')
                    ax.set_ylabel(r'Flux')
                    ax.xaxis.set_major_locator(MaxNLocator(5))
                    ax.set_xlim(4995.0,5024.0)

                    lamOIII = 5008.24
      
                    ax.plot([lamOIII, lamOIII], [-0.1,5.0], color="gray")
                    oiiiind = np.where((restwave > 4995.0) & (restwave < 5024.0))
                    ax.set_ylim(-0.1,0.2+max(flux_bin[oiiiind]))

                    # Now plot OIII in rest-frame of emission line fit
                    el_lamOIII = lamOIII * (1.0 + bin_elz) / (1.0 + bin_z) 
                    ax.plot([el_lamOIII, el_lamOIII], [-0.1,5.0], color="blue")
                    ax.text(0.7,0.9,'[OIII]',ha='left',va='center',transform = ax.transAxes,fontsize=14,color="k")

                    ################################
                    ## Plot Mgb
                    ax = pl.subplot(2,4,6)
                    for axis in ['top','bottom','left','right']:
                        ax.spines[axis].set_linewidth(2.0)

                    ax.plot(restwave,flux_bin,drawstyle='steps-mid', color="k")
                    ax.plot(restwave,smod_bin, color="red")
                    #ax.plot(restwave,elmod_bin, color="blue")
                    ax.set_xlabel(r'Rest Wavelength (Ang)')
                    ax.set_ylabel(r'Flux')
                    ax.xaxis.set_major_locator(MaxNLocator(5))
                    ax.set_xlim(5158.0,5195.0)

                    lamMgI1 = 5168.74
                    lamMgI2 = 5174.14
                    lamMgI3 = 5185.04
            
                    ax.plot([lamMgI1,lamMgI1], [-0.1,5.0], color="gray")
                    ax.plot([lamMgI2,lamMgI2], [-0.1,5.0], color="gray")
                    ax.plot([lamMgI3,lamMgI3], [-0.1,5.0], color="gray")
                    mgbind = np.where((restwave > 5158.0) & (restwave < 5195.0))
                    ax.set_ylim(-0.1,0.2+max(flux_bin[mgbind]))

                    ax.text(0.7,0.9,'MgI',ha='left',va='center',transform = ax.transAxes,fontsize=14,color="k")
      
            
                    ################################
                    ## Plot Halpha
                    ax = pl.subplot(2,4,7)
                    for axis in ['top','bottom','left','right']:
                        ax.spines[axis].set_linewidth(2.0)
        
                    ax.plot(restwave,flux_bin,drawstyle='steps-mid', color="k")
                    ax.plot(restwave,smod_bin, color="red")
                    #ax.plot(restwave,elmod_bin, color="blue")
                    ax.set_xlabel(r'Rest Wavelength (Ang)')
                    ax.set_ylabel(r'Flux')
                    ax.xaxis.set_major_locator(MaxNLocator(5))
                    ax.set_xlim(6550.0,6578.0)

                    lamHa = 6564.633

                    ax.plot([lamHa, lamHa], [-0.2,15.0], color="gray")
                    haind = np.where((restwave > 6550.0) & (restwave < 6578.0))
                    ax.set_ylim(-0.1,0.2+max(flux_bin[haind]))

                    # Now plot Ha in rest-frame of emission line fit
                    el_lamHa = lamHa * (1.0 + bin_elz) / (1.0 + bin_z) 
                    ax.plot([el_lamHa, el_lamHa], [-0.1,5.0], color="blue")
            
                    stz_str = "{:.5f}".format(bin_z)
                    elz_str = "{:.5f}".format(bin_elz)
                    pltnote = 'Stellar redshift = '+stz_str
                    ax.text(0.05,0.1,pltnote,ha='left',va='center',transform = ax.transAxes,fontsize=12,color="gray")

                    pltnote = 'EL redshift = '+elz_str
                    ax.text(0.05,0.2,pltnote,ha='left',va='center',transform = ax.transAxes,fontsize=12,color="blue")
                    ax.text(0.7,0.9,'Ha',ha='left',va='center',transform = ax.transAxes,fontsize=12,color="k")
                    
                    #HaSN_str = "{:.2f}".format(ha_ew[qq])
                    #pltnote = 'EW = '+HaSN_str
                    #ax.text(0.1,0.9,pltnote,ha='left',va='center',transform = ax.transAxes,fontsize=12,color="k")

                    medobsew = np.median(ewNaI['obsew'])
                    medmodew = np.median(ewNaI['modew'])
                    
                    medobsew_str = "{:.2f}".format(medobsew)
                    medmodew_str = "{:.2f}".format(medmodew)
                    bin_str = "{:.1f}".format(binid_bin)

                    #ax.text(1.5,0.9,'Median obs. EW = '+medobsew_str,ha='left',va='center',transform = ax.transAxes,fontsize=14,color="k")
                    #ax.text(1.5,0.8,'Median mod. EW = '+medmodew_str,ha='left',va='center',transform = ax.transAxes,fontsize=14,color="red")
                    ax.text(1.5,0.7,'BINID = '+bin_str,ha='left',va='center',transform = ax.transAxes,fontsize=14,color="k")
            
                    pl.tight_layout()
                    pdf.savefig()




def main():

    # FIT_FLG=0  : plot standard fits
    # FIT_FLG=1  : plot SModFits
    # FIT_FLG=2  : plot stellar-model-normalized fits
    fig_mcmc_fitter(mcmcdirroot=None, overwrite=False, FIT_FLG=2)


if __name__ == '__main__':

    main()
