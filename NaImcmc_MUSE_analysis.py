from __future__ import print_function
import numpy as np
import sys
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn
from linetools.spectra.xspectrum1d import XSpectrum1D
import model_NaI
import model_fitter
import continuum_normalize_NaI
import time
from IPython import embed

def setup_script():

    path = '/Users/erickaguirre/mangadap/examples/'
    outdir = '/Users/erickaguirre/Desktop/NaImcmcIFU/muse'

    bin_dir = 'output2.0_'
    gal_name = 'NGC4030_NOISM_err_corr'
    gal_sub_dir = '/SQUARE2.0-MILESHC-MASTARHC2-NOISM/1/1/'
    outfil = outdir + gal_name + '_script2.0_err_corr'

    # log maps file path
    log_maps_fil = path + bin_dir + gal_name + gal_sub_dir + 'manga-1-1-MAPS-SQUARE2.0-MILESHC-MASTARHC2-NOISM.fits.gz'

    # Read in table file
    #table_hdu = fits.open(table_fil)[1].data
    #_, idxConvertShortToLong = np.unique(np.abs(table.BIN_ID),return_inverse=True)

    # For continuum-normalization around NaI
    # wavelength fitting range inside of NaI region
    fitlim = [5880.0, 5910.0]
    # speed of light in km/s
    c = 2.998e5
    # NGC 4030 redshift's
    redshift = 0.00489
    # NGC 1042 redshift's
    #redshift = 0.00460

    # log maps file
    hdu_map = fits.open(log_maps_fil)
    # bin ID has multiple layers of the same bin id map so use first one
    binid_map = hdu_map['BINID'].data[0]

    # Read in PPXF map
    #ppxfHDU = fits.open(ppxfmap_fil)[1].data
    #ppxf = np.array( [ppxfHDU.V, ppxfHDU.SIGMA, ppxfHDU.H3, ppxfHDU.H4, ppxfHDU.LAMBDA_R] ).T
    #median_V_stellar = np.nanmedian( ppxf[:,0] )
    #ppxf[:,0] = ppxf[:,0] - median_V_stellar
    #binid = np.array( [ppxfHDU.BIN_ID] ).T
    #binid = binid.flatten()

    # Need LSF in km/s
    # This gives LSF in Ang
    # CAUTION: this does not convert air wavelengths
    # to vacuum, or accounts for velocity offset of each bin
    # update: converted LSF wavelength from a air to a vacuum
    LSFdir = '/Users/erickaguirre/Desktop/SDSU_Research/Getting_used_to_MaNGA_DAP'
    LSFfil = LSFdir + 'LSF-Config_MUSE_WFM'
    configLSF = np.genfromtxt(LSFfil, comments='#')
    configLSF_wv_air = configLSF[:, 0]
    configLSF_res = configLSF[:, 1]

    # convert to vacuum since LSF is in air
    xspec = XSpectrum1D.from_tuple((configLSF_wv_air, 0.0 * configLSF_wv_air))
    xspec.meta['airvac'] = 'air'
    xspec.airtovac()
    configLSF_wv_vac = xspec.wavelength.value
    # convert LSF wavelength to the restframe using galaxy's redshift
    configLSF_restwv = configLSF_wv_vac/ (1.0+redshift)
    whLSF = np.where((configLSF_restwv > fitlim[0]) & (configLSF_restwv < fitlim[1]))
    median_LSFAng = np.median(configLSF_res[whLSF[0]])
    median_LSFvel = c * median_LSFAng / np.median(configLSF_wv_vac[whLSF[0]])

    LSFvel_str = "{:.2f}".format(median_LSFvel)
    redshift_str = "{:.6f}".format(redshift)
    
    f = open(outfil, "w")
    f.write("#!/bin/sh\n")

    # number of total bins from bin ID map
    nbins = np.max(binid_map)
    binsperrun = 250
    # Number of separate "runs"
    nruns = int(nbins / binsperrun)

    for nn in range(nruns+1):

        startbinid = nn*binsperrun
        endbinid = (nn+1)*binsperrun

        if(endbinid > nbins):
            endbinid = nbins

        jobname = 'NaImcmc' + '_bin_' + str(startbinid) + '_' + str(endbinid) + '_run' + str(nn)
        # print(startbinid, endbinid)

        f.write('screen -mdS '+jobname+' sh -c "python NaImcmc_MUSE_analysis.py 1 '\
                +gal_name+' '+redshift_str+' '+LSFvel_str+' '+ str(nn) +' '+\
                str(startbinid)+' '+str(endbinid)+'"\n')
        
    f.close()
    
    # Set up script that lists
    # input root, redshift, LSFvel, startbinid, endbinid
   # pdb.set_trace()

def run_mcmc(galname, redshift, LSFvel, binid_run, startbinid, endbinid):
    start_time1 = time.time()
    path = '/Users/erickaguirre/Desktop/DAP_outputs/'
    bin_dir = 'output0.6_'
    gal_sub_dir = '/SQUARE0.6-MILESHC-MASTARHC2-NOISM/1/1/'
    outdir = '/Users/erickaguirre/Desktop/NaI_MCMC_output/NGC4030_0.6_err_corr/'
    outfits = outdir+galname+'-binid-'+str(startbinid)+'-'+str(endbinid)+ \
          '-samples-'+ 'run'+str(binid_run) +'.fits'
    # outpdf = outdir+galname+'-binid-'+str(startbinid)+'-'+\
    #     str(endbinid)+'-line_triangle.pdf'


    # For continuum-normalization around NaI
    # wavelength continuum fitting range outside of NaI region
    blim = [5850.0, 5870.0]
    rlim = [5910.0, 5930.0]
    # wavelength fitting range inside of NaI region
    fitlim = [5880.0, 5910.0]
    # speed of light in km/s
    c = 2.998e5

    # log cube model and data file path
    log_cube_fil = path + bin_dir + galname + gal_sub_dir + 'manga-1-1-LOGCUBE-SQUARE0.6-MILESHC-MASTARHC2-NOISM.fits.gz'
    # log maps file path
    log_maps_fil = path + bin_dir + galname + gal_sub_dir + 'manga-1-1-MAPS-SQUARE0.6-MILESHC-MASTARHC2-NOISM.fits.gz'

    # log maps file
    hdu_map = fits.open(log_maps_fil)
    # bin ID has multiple layers of the same bin id map so use first one
    binid_map = hdu_map['BINID'].data[0]

    # stellar velocity map
    ppxf_v_map = hdu_map['STELLAR_VEL'].data

    # Read in binned spectra
    # extract cube data  spectrum
    hdu_cube = fits.open(log_cube_fil)
    spec = hdu_cube['FLUX'].data
    # extract error spectrum from cube
    ivar = hdu_cube['IVAR'].data
    espec = np.sqrt(1 / ivar)
    # extract cube best-fit model spectrum
    mod = hdu_cube['MODEL'].data
    # observed wavelength
    obswave = hdu_cube['WAVE'].data

    sv_samples = []
    sv_binnumber = []
    sv_percentiles = []
    sv_velocities = []

    # Set up array with all relevant binids
    fitbins = np.arange(startbinid,endbinid+1)

    for qq in fitbins:
        start_time2 = time.time()
        # bin ID index
        ind = binid_map == qq

        # single bin velocity
        binvel = ppxf_v_map[ind][0]
        # single flux, error and model spectrum corresponding to that bin
        flux_bin = np.ma.array(spec[:, ind][:, 0])
        err_bin = np.ma.array(espec[:, ind][:, 0])
        mod_bin = np.ma.array(mod[:, ind][:, 0])

        # Determine bin redshift: cz in km/s = tstellar_kin[*,0]
        #bin_z = (cz + stellar_vfield[qq]) / sol
        #cosmo_z = cz / sol
        bin_z = redshift + ((1 + redshift) * (binvel / c))
        restwave = obswave / (1.0 + bin_z)

        #ndata = continuum_normalize_NaI.norm(restwave, mod_bin, err_bin, blim, rlim, FIT_FLG=0)
        #ndata = continuum_normalize_NaI.norm(restwave, mod_bin, err_bin, blim, rlim, FIT_FLG=0)
        #ndata = continuum_normalize_NaI.norm(restwave, mod_bin, err_corr_bin, blim, rlim, FIT_FLG=0)
        # gas flux (total flux / continuum)
        gas_ndata = continuum_normalize_NaI.smod_norm(restwave,flux_bin,err_bin,mod_bin,blim,rlim)
        print("""Beginning fit for bin {0} """.format(qq))

        # Cut out NaI
        select = np.where((gas_ndata['nwave'] > fitlim[0]) & (gas_ndata['nwave'] < fitlim[1]))
        restwave_NaI = gas_ndata['nwave'][select].astype('float64')
        flux_NaI = gas_ndata['nflux'][select].astype('float64')
        err_NaI = gas_ndata['nerr'][select].astype('float64')
        sres_NaI = LSFvel

        data = {'wave':restwave_NaI, 'flux':flux_NaI, 'err':err_NaI, 'velres':sres_NaI}
        #pdb.set_trace()

        # check for bad data being masked
        if (data['flux'].mask.all() == True) | (data['err'].mask.all() == True):
            sv_binnumber.append(binid_map[ind][0])
            sv_samples.append(np.zeros((100,1100,4)))
            sv_percentiles.append(np.zeros((4,3)))
            sv_velocities.append(-999)
            continue
        
        # Guess good model parameters
        lamred = 5897.5581
        logN = 14.5
        bD = 20.0
        Cf = 0.5
        theta_guess = lamred, logN, bD, Cf
        guess_mod = model_NaI.model_NaI(theta_guess, data['velres'], data['wave'])
        datfit = model_fitter.model_fitter(data, theta_guess)
        # Run the MCMC
        datfit.mcmc()
        #transinfo = model_NaI.transitions()

        # get gas velocity from model lambda and rest lambda
        lamred_mcmc, logN_mcmc, bD_mcmc, Cf_mcmc = datfit.theta_percentiles
        lamrest = 5897.5581
        velocity = ( (lamred_mcmc[0]/lamrest) - 1 ) * c

        sv_binnumber.append(binid_map[ind][0])
        sv_samples.append(datfit.samples)
        sv_percentiles.append(datfit.theta_percentiles)
        sv_velocities.append(velocity)
        end_time2 = time.time()
        print('Time elapsed for this bin {:.2f} minutes'.format( (end_time2 - start_time2)/60 ) )
        
    t = Table([sv_binnumber, sv_samples, sv_percentiles, sv_velocities],\
              names=('bin', 'samples', 'percentiles', 'velocities'))
    fits.writeto(outfits, np.array(t), overwrite=True)
    end_time1 = time.time()
    print('Total time elapsed {:.2f} hours'.format((end_time1 - start_time1) / 3600))
        
def main():

    flg = int(sys.argv[1])

    if (flg==0):
        setup_script()

    if (flg==1):

        #pdb.set_trace()
        gal = sys.argv[2]
        redshift = float(sys.argv[3])
        LSFvel = float(sys.argv[4])
        binid_run = int(sys.argv[5])
        startbin = int(sys.argv[6])
        endbin = int(sys.argv[7])

        run_mcmc(galname=gal, redshift=redshift, LSFvel=LSFvel,binid_run=binid_run,
                 startbinid=startbin, endbinid=endbin)
    
main()
