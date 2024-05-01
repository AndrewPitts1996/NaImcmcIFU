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

def fake_setup_noiseonly(plate=None, ifu=None, binid_for_fit=None, s2n=None):
    
    #plate = 7443
    #ifu = 6102
    #binid_for_fit = 26
    if((plate==None) | (ifu==None) | (binid_for_fit==None)):
        print("Need to select bin properly!")
        pdb.set_trace()

    if(s2n==None):
        print("Need to request a S/N level!")
        pdb.set_trace()
    
    outdir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/FakeSets/NoiseOnly/Representative_logNTest/'

    if(s2n>0.0):
        s2n_str = '{0:.0f}'.format(s2n)
    else:
        s2n_str = 'Variable'
    outfil_npy = outdir + '/'+str(plate)+'-'+str(ifu)+'-'+\
        str(binid_for_fit)+'-noiseonly-fakeset-SN'+s2n_str+'.npy'
    outfil_pdf = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/Plots/'+\
        'fig_examine_fakeabslines'+'-'+str(plate)+'-'+str(ifu)+'-'+\
        str(binid_for_fit)+'.pdf'

    print("Generating fake spectra for ", str(plate)+'-'+str(ifu)+'-'+str(binid_for_fit))
    # Set up fake line parameters
    # CHECK THAT DIFFERENT REALIZATIONS ACTUALLY LOOK DIFFERENT
 
    #nreal = 100
    nreal = 1
    
    saspath = '/data/manga/sas/mangawork/'
    drppath = saspath+'manga/spectro/redux/MPL-9/'
    dappath = saspath+'users/u6021943/Alt-DAP/SQUARE2.0_2020jan09/SQUARE2.0-MILESHC-MASTARHC/'

    fits_drp = drppath+str(plate)+'/stack/manga-'+str(plate)+'-'+str(ifu)+\
        '-LOGCUBE.fits.gz'
    fits_dap_cube = dappath+str(plate)+'/'+str(ifu)+'/manga-'+str(plate)+\
        '-'+str(ifu)+'-LOGCUBE-SQUARE2.0-MILESHC-MASTARHC.fits.gz'
    fits_dap_map = dappath+str(plate)+'/'+str(ifu)+'/manga-'+str(plate)+\
        '-'+str(ifu)+'-MAPS-SQUARE2.0-MILESHC-MASTARHC.fits.gz'

    # For continuum-normalization around NaI
    blim = [5850.0,5870.0]
    rlim = [5910.0,5930.0]
    fitlim = [5880.0,5910.0]

    transinfo = model_NaI.transitions()
    sol = 2.998e5

    # Read in the data
    cube = NaImcmc_read_fits.NaImcmc_read_fits(fits_dap_map, fits_dap_cube,\
                                               fits_drp=fits_drp)

    wave = cube['wave']
    # sres = cube['sres']
    flux = cube['fluxb']
    ivar = cube['ivarb']
    smod = cube['modelb']
    predisp = cube['predispb']
    
    cz = cube['cz']
    stellar_vfield = cube['stellar_vfieldb']
    binid = cube['binidb']
    bb = np.where(binid==binid_for_fit)
    bb = bb[0][0]
    
    flux_bin = flux[:,bb]
    ivar_bin = ivar[:,bb]
    err_bin = 1.0/np.sqrt(ivar_bin)
    smod_bin = smod[:,bb]
    predisp_bin = predisp[:,bb]
    #bin_z = (cz + stellar_vfield[bb]) / sol
    cosmo_z = cz / sol
    bin_z = cosmo_z + ((1 + cosmo_z) * stellar_vfield[bb] / sol) 
    
    restwave = wave / (1.0 + bin_z)
    ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, \
                                         blim, rlim, FIT_FLG=1, smod=smod_bin)
    

    # Cut out NaI
    ind = np.where((ndata['nwave'] > fitlim[0]) & (ndata['nwave'] < fitlim[1]))
    restwave_NaI = ndata['nwave'][ind]
    flux_NaI = ndata['nflux'][ind]     # this is smod
    err_NaI = ndata['nerr'][ind]
    #sres_NaI = sres[ind]
    predisp_NaI = predisp_bin[ind]
    wave_NaI = wave[ind]
    #avg_res = sol/np.mean(sres_NaI)
    avg_res = sol * 2.355 * np.mean(predisp_NaI)/np.mean(wave_NaI)
    data = {'wave':restwave_NaI, 'flux':flux_NaI, 'err':err_NaI, 'velres':avg_res}
    
    # Construct new fake cube
    dictList = []
      
    clean_wave = data['wave']
    clean_flux = data['flux']  # again, this is smod
    clean_err = data['err']
    clean_velres = data['velres']

    # Deal with some values being masked
    clean_err = clean_err.filled()
    
    wv_unit = u.AA
    restwave_unit = u.Quantity(clean_wave,unit=wv_unit)
    xspec = XSpectrum1D.from_tuple((restwave_unit,clean_flux,clean_err))

    #pdb.set_trace()
    for nn in range(nreal):
        #print(nn)
        xspec_noisy = xspec.add_noise(s2n=s2n)

        # Noise array should be sig = 1./s2n
        err_fkline = np.full(clean_err.shape,1.0/s2n)

        clean_dict = {'wave':clean_wave, 'flux':clean_flux, \
                      'err':clean_err, 'velres':clean_velres, \
                      'flux_fkline':xspec_noisy.flux.value, \
                      'err_fkline':err_fkline, \
                      'Cf':0.0, 'logN':0.0, \
                      'bD':0.0, 'v':0.0, 's2n':s2n}
        dictList.append(clean_dict)
        
        #pdb.set_trace()
    np.save(outfil_npy, dictList)
    print("Saving ", outfil_npy)


def fake_setup_addabs(plate=None, ifu=None, binid_for_fit=None, s2n=None):
    
    #plate = 7443
    #ifu = 6102
    #binid_for_fit = 26
    if((plate==None) | (ifu==None) | (binid_for_fit==None)):
        print("Need to select bin properly!")
        pdb.set_trace()

    if(s2n==None):
        print("Need to request S/N level!")
        pdb.set_trace()
    
    outdir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/FakeSets/AddAbs/'

    s2n_str = '{0:.0f}'.format(s2n)
    outfil_npy = outdir + '/'+str(plate)+'-'+str(ifu)+'-'+\
        str(binid_for_fit)+'-addabs-fakeset-SN'+s2n_str+'.npy'

    # Set up fake line parameters
    # CHECK THAT DIFFERENT REALIZATIONS ACTUALLY LOOK DIFFERENT
 
    nreal = 10
    #fake_s2n = np.array([20.0, 30.0, 40.0])
    fake_Cf = np.array([0.25, 0.5, 0.75, 1.0])
    fake_logN = np.array([11.5, 12.0, 12.5, 13.0])
    fake_bD = np.array([10.0, 30.0])
    fake_v = np.arange(-100.0, 120.0, 20.0)

    saspath = '/data/home/krubin/sas/mangawork/'
    drppath = saspath+'manga/spectro/redux/MPL-7/'
    dappath = saspath+'users/u6021943/Alt-DAP/SQUARE2.0_2018oct25/SQUARE2.0-GAU-MILESHC/'

    fits_drp = drppath+str(plate)+'/stack/manga-'+str(plate)+'-'+\
        str(ifu)+'-LOGCUBE.fits.gz'
    fits_dap_cube = dappath+str(plate)+'/'+str(ifu)+'/manga-'+str(plate)+'-'+\
        str(ifu)+'-LOGCUBE-SQUARE2.0-GAU-MILESHC.fits.gz'
    fits_dap_map = dappath+str(plate)+'/'+str(ifu)+'/manga-'+str(plate)+'-'+\
        str(ifu)+'-MAPS-SQUARE2.0-GAU-MILESHC.fits.gz'


    # For continuum-normalization around NaI
    blim = [5850.0,5870.0]
    rlim = [5910.0,5930.0]
    fitlim = [5880.0,5910.0]

    transinfo = model_NaI.transitions()
    sol = 2.998e5

    # Read in the data
    cube = NaImcmc_read_fits.NaImcmc_read_fits(fits_dap_map, fits_dap_cube, \
                                               fits_drp=fits_drp)

    wave = cube['wave']
    sres = cube['sres']
    flux = cube['fluxb']
    ivar = cube['ivarb']
    smod = cube['modelb']
    predisp = cube['predispb']
    cz = cube['cz']
    stellar_vfield = cube['stellar_vfieldb']
    binid = cube['binidb']
    bb = np.where(binid==binid_for_fit)
    bb = bb[0][0]
    
    flux_bin = flux[:,bb]
    ivar_bin = ivar[:,bb]
    err_bin = 1.0/np.sqrt(ivar_bin)
    smod_bin = smod[:,bb]
    predisp_bin = predisp[:,bb]
    #bin_z = (cz + stellar_vfield[bb]) / sol

    cosmo_z = cz / sol
    bin_z = cosmo_z + ((1 + cosmo_z) * stellar_vfield[bb] / sol) 
    restwave = wave / (1.0 + bin_z)
    ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, \
                                         blim, rlim, FIT_FLG=1, smod=smod_bin)
    


    # Cut out NaI
    ind = np.where((ndata['nwave'] > fitlim[0]) & (ndata['nwave'] < fitlim[1]))
    restwave_NaI = ndata['nwave'][ind]
    flux_NaI = ndata['nflux'][ind]     # this is smod
    err_NaI = ndata['nerr'][ind]
    #sres_NaI = sres[ind]
    predisp_NaI = predisp_bin[ind]  # This is LSF sigma in Angstroms, need to convert to velocity FWHM
    wave_NaI = wave[ind]

    #avg_res = sol/np.mean(sres_NaI)
    avg_res = sol * 2.355 * np.mean(predisp_NaI)/np.mean(wave_NaI)
    data = {'wave':restwave_NaI, 'flux':flux_NaI, 'err':err_NaI, 'velres':avg_res}
    
    # Construct new fake cube
    dictList = []

    clean_wave = data['wave']
    clean_flux = data['flux']  # again, this is smod
    clean_err = data['err']
    clean_velres = data['velres']

    wv_unit = u.AA
    restwave_unit = u.Quantity(clean_wave,unit=wv_unit)
    xspec = XSpectrum1D.from_tuple((restwave_unit,clean_flux,clean_err))        

    for ii in range(len(fake_Cf)):
        for jj in range(len(fake_logN)):
            for kk in range(len(fake_bD)):
                for ll in range(len(fake_v)):
                                            
                    lamred = ((fake_v[ll] / sol) * transinfo['lamred0']) + transinfo['lamred0']
                    theta = lamred, fake_logN[jj], fake_bD[kk], fake_Cf[ii]
                        
                    # Not sure this is legit.  Am smoothing the fake line, then
                    # multiplying it by the smoothed stellar continuum.  But would need
                    # access to unsmoothed stellar continuum to do better.
                    add_mod = model_NaI.model_NaI(theta, clean_velres, clean_wave)
                    newflux = clean_flux * add_mod['modflx']
                        
                    # Add noise
                    xspec = XSpectrum1D.from_tuple((restwave_unit,newflux,clean_err))

                    for nn in range(nreal):
                        xspec_noisy = xspec.add_noise(s2n=s2n)

                        clean_dict = {'wave':clean_wave, 'flux':clean_flux, \
                                      'err':clean_err, 'velres':clean_velres, \
                                      'flux_fkline':xspec_noisy.flux.value, \
                                      'Cf':fake_Cf[ii], 'logN':fake_logN[jj], \
                                      'bD':fake_bD[kk], 'v':fake_v[ll], 's2n':s2n}
                     

                        #print(data['Cf'], data['logN'], data['bD'], data['v'])
                        dictList.append(clean_dict)
    
    np.save(outfil_npy, dictList)                  

    
def fake_select(infil=None, outpdf=None, outfits=None):

    if(infil==None):

        # Collection of objects with full range of inclinations and total stellar masses
        # (This file is made on laptop)
        infil = '../LineProfileSims/NaImcmcfake_galselect.fits'

    if(outpdf==None):

        outpdf = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/Plots/fig_NaImcmcfake_selectbins_SNmin.pdf'

    if(outfits==None):

        outfits = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/NaImcmcfake_galselect_binselect_SNmin30.fits'
        
    fakefits = fits.getdata(infil)
    
    saspath = '/data/manga/sas/mangawork/'
    anlypath = 'users/u6021943/Alt-DAP/SQUARE2.0_2018oct25/SQUARE2.0-GAU-MILESHC/'
    rdxpath = 'manga/spectro/redux/MPL-7/'
    dapfits = saspath + 'users/u6021943/Alt-DAP/SQUARE2.0_2018oct25/dapall-v2_4_3-2.2.1.fits'
    
    plate = fakefits['PLATE']
    ifu = fakefits['IFU']
    #plateifu = data['PLATEIFU']
    #mngtarg1 = data['MNGTARG1']
    #mngtarg3 = data['MNGTARG3']
    #objra = data['OBJRA']
    #objdec = data['OBJDEC']
    #daptype = data['DAPTYPE']
    #bin_r_snr = data['BIN_R_SNR'][:,0]
    #z = data['NSA_Z']
    #ba = data['NSA_SERSIC_BA']
    #sersic_n = data['NSA_SERSIC_N']

    # For continuum-normalization around NaI
    blim = [5850.0,5870.0]
    rlim = [5910.0,5930.0]
    fitlim = [5880.0,5910.0]

    
    # Collect info on veldisp of each bin and EW of NaI in stellar model
    veldisp = np.empty((len(fakefits),300))
    veldisp.fill(np.nan)
    stellar_ewNaI = np.empty((len(fakefits),300))
    stellar_ewNaI.fill(np.nan)
    all_binids = np.empty((len(fakefits),300))
    all_binids.fill(np.nan)
    NaI_s2n = np.empty((len(fakefits),300))
    NaI_s2n.fill(np.nan)
                            
    
    for qq in range(len(fakefits)):

        plate_str = "{:.0f}".format(plate[qq])
        ifu_str = "{:.0f}".format(ifu[qq])
        dap_path = saspath + anlypath + plate_str+'/'+ifu_str+'/manga-'+plate_str+'-'+ifu_str
        fits_dap_map = dap_path + '-MAPS-SQUARE2.0-GAU-MILESHC.fits.gz'
        fits_dap_cube = dap_path + '-LOGCUBE-SQUARE2.0-GAU-MILESHC.fits.gz'

        ## Read in data
        hdu_maps = fits.open(fits_dap_map)
        hdu_hdr = hdu_maps['PRIMARY'].header
        cz = hdu_hdr['SCINPVEL']  # in km/s, z=v/c
        binid = hdu_maps['BINID'].data[1,:,:]
        binarea = hdu_maps['BIN_AREA'].data
        binsnr = hdu_maps['BIN_SNR'].data
        
        stellar_vfield = hdu_maps['STELLAR_VEL'].data
        stellar_vfield_mask = hdu_maps['STELLAR_VEL_MASK'].data
        stellar_vfield_ivar = hdu_maps['STELLAR_VEL_IVAR'].data

        stellar_sigma = hdu_maps['STELLAR_SIGMA'].data
        stellar_sigma_mask = hdu_maps['STELLAR_SIGMA_MASK'].data
        stellar_sigmacorr = hdu_maps['STELLAR_SIGMACORR'].data

        sigma_star_corr = np.sqrt(stellar_sigma**2.0 - stellar_sigmacorr**2.0)
        
        # Calculate EWs -- from NaImcmc_write_fitsinfo
        ewNaI = ew_NaI_allspax.ew(fits_dap_map, fits_dap_cube, blim, rlim, FIT_FLG=0)

        ewNaI_mod_map = np.zeros(binid.shape)
        ewNaI_s2n_map = np.zeros(binid.shape)
        
        for pp in range(len(ewNaI['binid'])):

            ewind = np.where(binid==ewNaI['binid'][pp])
            ewNaI_mod_map[ewind] = ewNaI['modew'][pp]
            ewNaI_s2n_map[ewind] = ewNaI['s2n'][pp]
            #print(ewNaI['binid'][pp], binid[ewind])

        # Reshape!
        binidb = np.reshape(binid, [binid.shape[0] * binid.shape[1]])
        sigma_star_corrb = np.reshape(sigma_star_corr, [binid.shape[0] * binid.shape[1]])
        ewNaI_mod_mapb = np.reshape(ewNaI_mod_map, [binid.shape[0] * binid.shape[1]])
        ewNaI_s2n_mapb = np.reshape(ewNaI_s2n_map, [binid.shape[0] * binid.shape[1]])
        
        uniq_el, uniq_ind = np.unique(binidb, return_index=True)
        uniq_el = uniq_el[1:]
        uniq_ind = uniq_ind[1:]

        ubinidb = binidb[uniq_ind]
        usigma_star_corrb = sigma_star_corrb[uniq_ind]
        uewNaI_mod_mapb = ewNaI_mod_mapb[uniq_ind]
        uewNaI_s2n_mapb = ewNaI_s2n_mapb[uniq_ind]

        all_binids[qq,0:len(ubinidb)] = ubinidb
        veldisp[qq,0:len(usigma_star_corrb)] = usigma_star_corrb
        stellar_ewNaI[qq,0:len(uewNaI_mod_mapb)] = uewNaI_mod_mapb
        NaI_s2n[qq,0:len(uewNaI_s2n_mapb)] = uewNaI_s2n_mapb

        #if(qq==4):
        #    pdb.set_trace()
        # Save this?

    #pdb.set_trace()
    # Plot! -- the above looks ok!  continue here!
    pl.figure(figsize=(8.0,7.0))
    pl.rcParams.update({'font.size': 14})

    pltnote = ['$10^{\circ} < i < 20^{\circ}$', \
               '$20^{\circ} < i < 30^{\circ}$', \
               '$30^{\circ} < i < 40^{\circ}$', \
               '$40^{\circ} < i < 50^{\circ}$', \
               '$50^{\circ} < i < 60^{\circ}$', \
               '$60^{\circ} < i < 70^{\circ}$', '$i > 70^{\circ}$']
    incl_lim_lo = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    incl_lim_hi = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 90.0]

    # Select vel disps and EWs appropriate for each inclination bin
    goal_veldisp = np.arange(60.0,300.0,60.0)
    span_veldisp = 180.0
    goal_ewNaI = np.arange(1.0,5.0,1.0)
    span_ewNaI = 2.5
    sel_limit = 0.2

    select_plate = []
    select_ifu = []
    select_binid = []
    select_inclination = []
    select_logMstar = []
    select_logSFR = []
    select_veldisp = []
    select_ewNaI = []
    select_s2nNaI = []
    
    color_choice = ['red', 'orange', 'cyan']
    
    for kk in range(len(incl_lim_lo)):

        ax = pl.subplot(3,3,kk+1)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.tick_params(which='major', axis='both', width=1.5, length=5, \
                       top='on', right='on', direction='in')
        ax.tick_params(which='minor', axis='both', width=1.5, length=3, \
                       top='on', right='on', direction='in')
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        #ax.yaxis.set_minor_locator(pl.MultipleLocator(0.1))
        #ax.xaxis.set_minor_locator(pl.MultipleLocator(0.1))
        ax.set_xlim(0,300)
        ax.set_ylim(-5,6.5)
        
        whincl= np.where((fakefits['INCLINATION'] < incl_lim_hi[kk]) & \
                         (fakefits['INCLINATION'] > incl_lim_lo[kk]))
        whincl = whincl[0]

        
        for ll in range(len(whincl)):

            ind = whincl[ll]
            
            print(fakefits['PLATE'][ind], fakefits['IFU'][ind], fakefits['STELLAR_MASS'][ind])

            if(fakefits['STELLAR_MASS'][ind] < 9.5):
                pltclr = 'cyan'
            elif((fakefits['STELLAR_MASS'][ind] > 9.5) & (fakefits['STELLAR_MASS'][ind] < 10.5)):
                pltclr = 'orange'
            else:
                pltclr = 'red'

            whsn = (NaI_s2n[ind,:] > 30.0) #& (NaI_s2n[ind,:] < 35.0)
            
            plate_arr = np.empty(len(stellar_ewNaI[ind,whsn]))
            plate_arr.fill(fakefits['PLATE'][ind])
            ifu_arr = np.empty(len(stellar_ewNaI[ind,whsn]))
            ifu_arr.fill(fakefits['IFU'][ind])
            inclination_arr = np.empty(len(stellar_ewNaI[ind,whsn]))
            inclination_arr.fill(fakefits['INCLINATION'][ind])
            logMstar_arr = np.empty(len(stellar_ewNaI[ind,whsn]))
            logMstar_arr.fill(fakefits['STELLAR_MASS'][ind])
            logSFR_arr = np.empty(len(stellar_ewNaI[ind,whsn]))
            logSFR_arr.fill(fakefits['LOGSFR'][ind])
        
            # Collect all info for this inclination bin
            if(ll==0):
                
                collect_plate = plate_arr
                collect_ifu = ifu_arr
                collect_inclination = inclination_arr
                collect_logMstar = logMstar_arr
                collect_logSFR = logSFR_arr
                collect_binid = all_binids[ind,whsn]
                collect_ew = stellar_ewNaI[ind,whsn]
                collect_veldisp = veldisp[ind,whsn]
                collect_NaI_s2n = NaI_s2n[ind,whsn]
                
            else:

                collect_plate = np.concatenate([collect_plate,plate_arr])
                collect_ifu = np.concatenate([collect_ifu,ifu_arr])
                collect_inclination = np.concatenate([collect_inclination,inclination_arr])
                collect_logMstar = np.concatenate([collect_logMstar,logMstar_arr])
                collect_logSFR = np.concatenate([collect_logSFR,logSFR_arr])
                collect_binid = np.concatenate([collect_binid,all_binids[ind,whsn]])
                collect_ew = np.concatenate([collect_ew,stellar_ewNaI[ind,whsn]])
                collect_veldisp = np.concatenate([collect_veldisp,veldisp[ind,whsn]])
                collect_NaI_s2n = np.concatenate([collect_NaI_s2n,NaI_s2n[ind,whsn]])
                #pdb.set_trace()
            
            #ax.plot(veldisp[ind,:], stellar_ewNaI[ind,:], marker='o', linestyle='none', \
            #        markersize=1.5, color=pltclr)

            #whsn = (NaI_s2n[ind,:] > 30.0) #& (NaI_s2n[ind,:] < 35.0)
            ax.plot(veldisp[ind,whsn], stellar_ewNaI[ind,whsn], marker='o', linestyle='none', \
                    markersize=1.5, color=pltclr)
            #pdb.set_trace()
            
            #print(np.nanmin(veldisp[ind,:]), np.nanmax(veldisp[ind,:]))
            #print(np.nanmin(stellar_ewNaI[ind,:]), np.nanmax(stellar_ewNaI[ind,:]))
            #print("Medians: ", np.nanmedian(collect_veldisp), np.nanmedian(collect_ew))
            #clrct = clrct+1

            # TRY SELECTING TWO BINS FROM EACH GALAXY
            # SELECT BY HAND; just overplot bin numbers
            # just make sure the selections complement each other within each inclination bin

            # Actually do selection
            for ii in range(len(goal_veldisp)):
                for jj in range(len(goal_ewNaI)):

                    dist = np.sqrt(((collect_ew-goal_ewNaI[jj])/span_ewNaI)**2 + \
                                   ((collect_veldisp-goal_veldisp[ii])/span_veldisp)**2)
                    sel = np.nanargmin(dist)
                    print(goal_ewNaI[jj], goal_veldisp[ii], dist[sel], collect_ew[sel], collect_veldisp[sel])

                    if(dist[sel] < sel_limit):
                        select_plate.append(collect_plate[sel])
                        select_ifu.append(collect_ifu[sel])
                        select_binid.append(collect_binid[sel])
                        select_inclination.append(collect_inclination[sel])
                        select_logMstar.append(collect_logMstar[sel])
                        select_logSFR.append(collect_logSFR[sel])
                        select_veldisp.append(collect_veldisp[sel])
                        select_ewNaI.append(collect_ew[sel])
                        select_s2nNaI.append(collect_NaI_s2n[sel])

                        ax.plot(collect_veldisp[sel], collect_ew[sel], marker='o', linestyle='none', \
                                markersize=3.0, color='blue')
               
   
        pl.text(0.5,1.035,pltnote[kk],ha='center',va='center',transform = ax.transAxes,fontsize=14)
        print("Medians: ", np.nanmedian(collect_veldisp), np.nanmedian(collect_ew))

    pl.tight_layout()
    pl.savefig(outpdf, format='pdf')
    
    select_plate = np.array(select_plate)
    select_ifu = np.array(select_ifu)
    select_binid = np.array(select_binid)
    select_inclination = np.array(select_inclination)
    select_logMstar = np.array(select_logMstar)
    select_logSFR = np.array(select_logSFR)
    select_veldisp = np.array(select_veldisp)
    select_ewNaI = np.array(select_ewNaI)
    select_s2nNaI = np.array(select_s2nNaI)
   
    ## OUTPUT TABLE WITH THESE PLATE-IFUs, binids, etc
    hdr = fits.Header()
    primary_hdu = fits.PrimaryHDU(header=hdr)
    col1 = fits.Column(name='PLATE', format=rec_to_fits_type(select_plate), array=select_plate)
    col2 = fits.Column(name='IFU', format=rec_to_fits_type(select_ifu), array=select_ifu)
    col3 = fits.Column(name='BINID', format=rec_to_fits_type(select_binid), array=select_binid)
    col4 = fits.Column(name='STELLAR_MASS', format=rec_to_fits_type(select_logMstar), array=select_logMstar)
    col5 = fits.Column(name='INCLINATION', format=rec_to_fits_type(select_inclination), array=select_inclination)
    col6 = fits.Column(name='LOGSFR', format=rec_to_fits_type(select_logSFR), array=select_logSFR)
    col7 = fits.Column(name='VELDISP', format=rec_to_fits_type(select_veldisp), array=select_veldisp)
    col8 = fits.Column(name='EWNAI', format=rec_to_fits_type(select_ewNaI), array=select_ewNaI)
    col9 = fits.Column(name='S2N', format=rec_to_fits_type(select_s2nNaI), array=select_s2nNaI)

    
    cols = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8, col9])
    fakeproc_hdu = fits.BinTableHDU.from_columns(cols)
    hdu1 = fits.HDUList([primary_hdu, fakeproc_hdu])
    hdu1.writeto(outfits, overwrite=True)
    

  
    pdb.set_trace()




def run_fake_setup(infits=None, setup_type=None, s2n=None):

    if(infits==None):

        #infits = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/NaImcmcfake_outflowbins.fits'
        infits = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/NaImcmcfake_galselect_binselect_SNmin30.fits'
        
    if(s2n==None):
        print("Need to set desired S/N level")
        pdb.set_trace()
        
    fakebins = fits.getdata(infits)
    #pdb.set_trace()

    if(setup_type=='NoiseOnly'):

        for qq in range(len(fakebins)):
        #for qq in range(50, len(fakebins)):
            #qq=95
            print("iter = ", qq)
            if(s2n>0.0):

                if((int(fakebins['PLATE'][qq])==8999) & (int(fakebins['IFU'][qq])==6102) & (int(fakebins['BINID'][qq])==56)):
                   print("Skipping 8999-6102-56")
                elif((int(fakebins['PLATE'][qq])==8999) & (int(fakebins['IFU'][qq])==6102) & (int(fakebins['BINID'][qq])==36)):
                   print("Skipping 8999-6102-36")
                elif((int(fakebins['PLATE'][qq])==9194) & (int(fakebins['IFU'][qq])==9101) & (int(fakebins['BINID'][qq])==115)):
                   print("Skipping 9194-9101-115")
                else:
                   fake_setup_noiseonly(plate=int(fakebins['PLATE'][qq]), ifu=int(fakebins['IFU'][qq]), \
                                     binid_for_fit=int(fakebins['BINID'][qq]), s2n=s2n)
            else:
                # Use actual spectral S/N
                thiss2n = fakebins['NAIS2N'][qq]
                fake_setup_noiseonly(plate=int(fakebins['PLATE'][qq]), ifu=int(fakebins['IFU'][qq]), \
                                     binid_for_fit=int(fakebins['BINID'][qq]), s2n=thiss2n)
            #pdb.set_trace()

    elif(setup_type=='AddAbs'):
        
        for qq in range(len(fakebins)):
            fake_setup_addabs(plate=int(fakebins['PLATE'][qq]), ifu=int(fakebins['IFU'][qq]), \
                                 binid_for_fit=int(fakebins['BINID'][qq]), s2n=s2n)
            pdb.set_trace()
    
    
def main():

    script = sys.argv[0]
    flg = int(sys.argv[1])
    
    if (flg==0):

        fake_select(infil=None, outpdf=None)
    
    if (flg==1):

        run_fake_setup(infits=None, setup_type='NoiseOnly', s2n=50.0)
        
        #NaImcmcfake_setup(7443, 6102, 26)
        #fake_setup(plate=7443, ifu=6102, binid_for_fit=26)
        #fake_setup(plate=8440, ifu=6104, binid_for_fit=5)
        #fake_setup(plate=8440, ifu=6104, binid_for_fit=30)
        #fake_setup(plate=8440, ifu=6104, binid_for_fit=35)
        #fake_setup(plate=9506, ifu=3701, binid_for_fit=8)
        #fake_setup(plate=8982, ifu=6104, binid_for_fit=26)
        #fake_setup(plate=8982, ifu=6104, binid_for_fit=53)
        #fake_setup(plate=7968, ifu=9101, binid_for_fit=1)
        #fake_setup(plate=7968, ifu=9101, binid_for_fit=9)
        #fake_setup(plate=7968, ifu=9101, binid_for_fit=121)
        #fake_setup(plate=8549, ifu=12705, binid_for_fit=8)
        #fake_setup(plate=8549, ifu=12705, binid_for_fit=45)

main()
