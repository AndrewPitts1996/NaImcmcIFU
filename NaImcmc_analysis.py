from __future__ import print_function

import math
import numpy as np
import os
import fnmatch
import scipy.optimize as op
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
import NaImcmc_read_fits
import NaImcmc_fitwrapper
import ew_NaI_allspax
import pdb
import sys

def select_fitsample(outfil, MORPH_FLG=None, FIT_FLG=None, INCLUDE_ALL_FLG=None):

    if(FIT_FLG==0):
        wrkpath = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/plan_files/SQUARE2.0/2020jul20/FluxFit/'
        samplepath = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/MCMC/SQUARE2.0/2020jul20/FluxFit/'

    elif(FIT_FLG==1):
        # Fit stellar model rather than data
        wrkpath = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/plan_files/SQUARE2.0/2019feb04/SModFit/'
        samplepath = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/MCMC/SQUARE2.0/2019feb04/SModFit/'

    elif(FIT_FLG==2):
        # Fit after dividing out stellar model
        wrkpath = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/plan_files/SQUARE2.0/2019feb04/FluxNoSModFit/'
        samplepath = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/MCMC/SQUARE2.0/2019feb04/FluxNoSModFit/'

    # INCLUDE_ALL_FLG=1: means list will include all galaxies (even those with existing fits)
    # INCLUDE_ALL_FLG=0: means list should not include galaxies that have already been fit
        
    #saspath = '/data/home/krubin/sas/mangawork/manga/spectro/'
    #anlypath = 'analysis/MPL-6/VOR10-GAU-MILESHC/'
    #rdxpath = 'redux/MPL-6/'

    saspath = '/data/manga/sas/mangawork/'
    anlypath = 'users/u6021943/Alt-DAP/SQUARE2.0_2020jan09/SQUARE2.0-MILESHC-MASTARHC/'
    rdxpath = 'manga/spectro/redux/MPL-9/'
    dapfits = saspath + 'users/u6021943/Alt-DAP/SQUARE2.0_2020jan09/dapall-v2_7_1-2.4.1.fits'

    if(outfil==None):
        outfil = wrkpath+'/NaImcmc_select_fitsample_GZSpirals_LSF.plan'

      
    hdu = fits.open(dapfits)
    data = hdu[1].data

    #pdb.set_trace()
    plate = data['PLATE']
    ifu = data['IFUDESIGN']
    plateifu = data['PLATEIFU']
    mngtarg1 = data['MNGTARG1']
    mngtarg3 = data['MNGTARG3']
    objra = data['OBJRA']
    objdec = data['OBJDEC']
    daptype = data['DAPTYPE']
    bin_r_snr = data['BIN_R_SNR'][:,0]
    z = data['NSA_Z']
    ba = data['NSA_SERSIC_BA']
    sersic_n = data['NSA_SERSIC_N']
    ellipticity = 1.0 - ba
    incl = np.arccos(1.0 - ellipticity) * 180.0 / math.pi

    sol = 2.998e5
    dapcoords = SkyCoord(ra=objra*u.degree, dec=objdec*u.degree)

    mpl9 = (mngtarg1!=0) & (daptype=='VOR10-MILESHC-MASTARHC')
    
    if(MORPH_FLG=='HC'):
        morphfil = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/Catalogs/Huertas-Company2011/catalog_withheader.dat'
        # Settings for matching to morphological catalog
        match_morph_rad = 1.0  ## arcsec
        match_morph_vel = 200.0  ## km/s
     
        morph_hc = ascii.read(morphfil)
        morph_RAdeg = morph_hc['RAdeg']
        morph_DEdeg = morph_hc['DEdeg']
        morph_z = morph_hc['z']
        morphcoords = SkyCoord(ra=morph_RAdeg*u.degree, dec=morph_DEdeg*u.degree)

        idx, d2d, d3d = dapcoords.match_to_catalog_sky(morphcoords)
        dv = sol * np.abs(z - morph_z[idx]) / (1.0+z)
        morph_match = (d2d.arcsec < match_morph_rad) & (dv < match_morph_vel) & (dv > 0.0)

        # For those with H-C morphologies, select late-types
        wh_hcS = (mngtarg1!=0) & (daptype=='VOR10-GAU-MILESHC') & (morph_match==True) & ((morph_hc['pSab'][idx]>0.50) | (morph_hc['pScd'][idx]>0.50))
        wh_no_hcS = (mngtarg1!=0) & (daptype=='VOR10-GAU-MILESHC') & (morph_match==False) & (sersic_n <= 2.0)
        late_type = wh_hcS | wh_no_hcS

        pdb.set_trace()

    elif(MORPH_FLG=='GZ'):
        morphfil = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/Catalogs/GalaxyZoo2/gz2_hart16.fits.gz'

        # Settings for matching to morphological catalog
        match_morph_rad = 1.5  ## arcsec
        #match_morph_vel = 200.0  ## km/s
        
        # Use overall classes as explained in Willett+13
        # Select anything beginning with 'S', but excluding:
        # '(d)' - disturbed - or '(m)' - merger - or '(i)' - irregular - or
        # '(l)' - lens - or '(o)' - other

        gz = fits.getdata(morphfil)
        gz2_class = gz['gz2_class']
        gz_useflg = np.zeros(len(gz2_class))

        for ii in range(len(gz_useflg)):

            if(gz2_class[ii][0]=='S'):
                oddstr = gz2_class[ii][-3:]
                if((oddstr=='(d)') | (oddstr=='(m)') | (oddstr=='(i)') |
                   (oddstr=='(l)') | (oddstr=='(o)')):
                    gz_useflg[ii] = 2
                else:
                    gz_useflg[ii] = 1

        # Now match to DAP catalog
        morph_RAdeg = gz['ra']
        morph_DEdeg = gz['dec']
        #morph_z = morph_hc['z']
        morphcoords = SkyCoord(ra=morph_RAdeg*u.degree, dec=morph_DEdeg*u.degree)

        idx, d2d, d3d = dapcoords.match_to_catalog_sky(morphcoords)
        #dv = sol * np.abs(z - morph_z[idx]) / (1.0+z)
        morph_match = (d2d.arcsec < match_morph_rad) #& (dv < match_morph_vel) & (dv > 0.0)

        late_type_gz = (mngtarg1!=0) & (daptype=='VOR10-MILESHC-MASTARHC') & (morph_match==True) & \
            (gz_useflg[idx]==1.0)

        late_type_nogz = (mngtarg1!=0) & (daptype=='VOR10-MILESHC-MASTARHC') & (morph_match==False) & (sersic_n <= 2.0)
        late_type = late_type_gz | late_type_nogz
        
    #pdb.set_trace()
    # Go for face-on galaxies first, primary + secondary sample
    #ind = np.where((mngtarg1!=0) & (daptype=='VOR10-GAU-MILESHC') & (sersic_n < 4.0) & (incl < 30.0))
    #ind = np.where((plate==7443) & (ifu==6102) & (daptype=='VOR10-GAU-MILESHC'))

    if((FIT_FLG==0) | (FIT_FLG==1)):
        #ind = np.where((mngtarg1!=0) & (daptype=='VOR10-GAU-MILESHC') &
        #               (sersic_n < 2.0)) #& (incl >= 60.0) &
                       #(plateifu!='8466-12704') & (plateifu!='7968-9101') &
                       #(plateifu!='9493-6101'))
        ind = np.where(late_type)
    elif(FIT_FLG==2):
        #ind = np.where((mngtarg1!=0) & (daptype=='VOR10-GAU-MILESHC') &
        #               (sersic_n < 2.0) & (bin_r_snr > 20.0))
        ind = np.where(late_type)
        
    ind = ind[0]
    # exclude 8466-12704
    # pdb.set_trace()

    # Check for existing samples files
    pattern = '*samples.fits'
    findall_samples = []
    for root, dirs, files in os.walk(samplepath):
        for filename in fnmatch.filter(files,pattern):
            findall_samples.append(filename)

    print("Found this many fits so far!:", len(findall_samples))

    #pdb.set_trace()
    findall_plateifu = []
    for nn in range(len(findall_samples)):

        nnsamples = findall_samples[nn]
        nnfil = nnsamples.split('-')
        found_plate = int(nnfil[0])
        found_ifu = int(nnfil[1])
        found_plateifu = "{:.0f}".format(found_plate)+"-"+"{:.0f}".format(found_ifu)
        findall_plateifu.append(found_plateifu)

    findall_plateifu = np.array(findall_plateifu)
    
    # Produce plan file
    drp = []#np.chararray(len(ind), itemsize=100)
    dap = []#np.chararray(len(ind), itemsize=100)
    ct_nodap = 0
    
    for qq in range(len(ind)):

        # Has this plateifu already been fit?
        fnd = np.where(findall_plateifu==plateifu[ind[qq]])
        fnd = fnd[0]

        # Does the required DAP LOGCUBE file exist?
        dap_filname = saspath+anlypath+str(plate[ind[qq]])+'/'+str(ifu[ind[qq]])+'/manga-'+str(plate[ind[qq]])+'-'+str(ifu[ind[qq]])+'-LOGCUBE-SQUARE2.0-MILESHC-MASTARHC.fits.gz'
        dap_exists = os.path.exists(dap_filname)

        #pdb.set_trace()
        
        if((INCLUDE_ALL_FLG==0) & (len(fnd)>0)):

            print("Skipping plate-ifu ", findall_plateifu[fnd])
            #pdb.set_trace()

        else:

            if(dap_exists==False):
                print("Could not find DAP file", dap_filname)
                ct_nodap = ct_nodap+1

            else:        
                drp.append(rdxpath+str(plate[ind[qq]])+'/stack/manga-'+str(plate[ind[qq]])+'-'+str(ifu[ind[qq]]))
                dap.append(anlypath+str(plate[ind[qq]])+'/'+str(ifu[ind[qq]])+'/manga-'+str(plate[ind[qq]])+'-'+str(ifu[ind[qq]]))

    table = Table([drp, dap], names=['drp','dap'])
    print("Writing new plan file: "+outfil)
    print("Missing DAP files: ", ct_nodap)
    ascii.write(table, outfil, overwrite=True)
    
    #pdb.set_trace()


def run_ewcheck(infil, outfil):

    wrkpath = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/plan_files/SQUARE2.0/'
    saspath = '/data/home/krubin/sas/mangawork/'

    # For continuum-normalization around NaI
    blim = [5850.0,5870.0]
    rlim = [5910.0,5930.0]

    fitlim = [5880.0,5910.0]

    
    if(infil==None):
        infil = wrkpath+'NaImcmc_select_fitsample_incl30_60.plan'

    if(outfil==None):
        outfil = wrkpath+'NaImcmc_select_fitsample_ewcut_incl30_60.plan'

        
    table = ascii.read(infil)
    drp = table['drp']
    dap = table['dap']
    keep_ind = []

    for qq in range(len(drp)):
    

        fits_drp = saspath + drp[qq] + '-LOGCUBE.fits.gz'
        fits_dap_cube = saspath + dap[qq] + '-LOGCUBE-VOR10-GAU-MILESHC.fits.gz'
        fits_dap_map = saspath + dap[qq] + '-MAPS-VOR10-GAU-MILESHC.fits.gz'

        print("Checking EW for ", dap[qq])
        ewNaI = ew_NaI_allspax.ew(fits_dap_map, fits_dap_cube, blim, rlim)

        # Returns EWs for all unique "good" bins
        obsew = np.array(ewNaI['obsew'])
        modew = np.array(ewNaI['modew'])

        ngdbins = len(obsew)
        whexcess = np.where(obsew > modew)

        if (len(whexcess[0]) > 0.05 * np.float(ngdbins)):

            keep_ind.append(qq)
            
        #pl.scatter(obsew, modew, color='r')
        #pl.axis([5850, 5930, 0, 2])
        #pl.plot(np.arange(5), np.arange(5), color='k')
        #pl.show()
        
        #hdu_maps = fits.open(fits_dap_map)
        #hdu_cube = fits.open(fits_dap_cube)
        #hdu_drp = fits.open(fits_drp)
        #hdu_hdr = hdu_cube['PRIMARY'].header
        #cz = hdu_hdr['SCINPVEL']  # in km/s, z = v/c

    #pdb.set_trace()
    table = Table([drp[keep_ind], dap[keep_ind]], names=['drp','dap'])
    print("Writing new plan file with EW cut: "+outfil)
    ascii.write(table, outfil, overwrite=True)


def setup_jobs(infil, outfilroot, scriptfil, FIT_FLG=None):

    if(FIT_FLG==0):
        wrkpath = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/plan_files/SQUARE2.0/2020jul20/FluxFit/'
        script_flg = '3'
        
    elif(FIT_FLG==1):
        wrkpath = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/plan_files/SQUARE2.0/2019feb04/SModFit/'
        script_flg = '6'

    elif(FIT_FLG==2):
        wrkpath = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/plan_files/SQUARE2.0/2019feb04/FluxNoSModFit/'
        script_flg = '9'
        
    if(infil==None):
        infil = wrkpath + 'NaImcmc_select_fitsample_GZSpirals_LSF.plan'
    if(outfilroot==None):
        outfilroot = wrkpath + 'NaImcmc_select_fitsample_GZSpirals_LSF'
    if(scriptfil==None):
        scriptfil = wrkpath + 'NaImcmc_runscript_GZSpirals_LSF'
        
    ncores = 20
    nlabeloff = 0
    table = ascii.read(infil)
    drp = table['drp']
    dap = table['dap']

    ngal = len(drp)

    nperfil = int(ngal/ncores)

    for nn in range(ncores+1):

        fil = outfilroot+str(nn+nlabeloff)+'.plan'

        ind = np.arange(nn*nperfil, (nn+1)*nperfil)

        if(max(ind)>=ngal):

            minind = min(ind)
            maxind = ngal-1
            ind = np.arange(minind, maxind)
        table = Table([drp[ind], dap[ind]], names=['drp','dap'])
        ascii.write(table, fil, overwrite=True)

    # Need to write
    # python NaImcmc_analysis.py 2**3 outrootfils
    f = open(scriptfil, "w")
    f.write("#!/bin/sh\n")
    

    for nn in range(ncores+1):
        fil = outfilroot+str(nn+nlabeloff)+'.plan'

        jobname = 'NaImcmc'+str(nn+nlabeloff)

        #f.write('screen -mdS '+jobname+' sh -c "python NaImcmc_analysis.py 3 '+fil+'"\n')
        f.write('screen -mdS '+jobname+' sh -c "python NaImcmc_analysis.py '+script_flg+' '+fil+'"\n')
        
    f.close()
    #pdb.set_trace()
    
def run_mcmc(infil, outdir, FIT_FLG=None):

    if(FIT_FLG==0):
        wrkpath = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/plan_files/SQUARE2.0/2020jul20/FluxFit/'
        outpath = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/MCMC/SQUARE2.0/2020jul20/FluxFit/'

    elif(FIT_FLG==1):
        wrkpath = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/plan_files/SQUARE2.0/2019feb04/SModFit/'
        outpath = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/MCMC/SQUARE2.0/2019feb04/SModFit/'

    elif(FIT_FLG==2):
        wrkpath = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/plan_files/SQUARE2.0/2019feb04/FluxNoSModFit/'
        outpath = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/MCMC/SQUARE2.0/2019feb04/FluxNoSModFit/'
        
        
    #saspath = '/data/home/krubin/sas/mangawork/'
    saspath = '/data/manga/sas/mangawork/'
    

    # For continuum-normalization around NaI
    blim = [5850.0,5870.0]
    rlim = [5910.0,5930.0]

    fitlim = [5880.0,5910.0]

    
    if(infil==None):
        infil = wrkpath+'NaImcmc_select_fitsample_ewcut.plan'

    if(outdir==None):
        outdir = outpath

    table = ascii.read(infil)
    drp = table['drp']
    dap = table['dap']

    print("Running MCMC for plan file ", infil)
    for qq in range(len(drp)):

        fits_drp = saspath + drp[qq] + '-LOGCUBE.fits.gz'
        fits_dap_cube = saspath + dap[qq] + '-LOGCUBE-SQUARE2.0-MILESHC-MASTARHC.fits.gz'
        fits_dap_map = saspath + dap[qq] + '-MAPS-SQUARE2.0-MILESHC-MASTARHC.fits.gz'
        
        # use this and subsequent variables after
        cube = NaImcmc_read_fits.NaImcmc_read_fits(fits_dap_map, fits_dap_cube, fits_drp)

        wave = cube['wave']
        prespecres = cube['prespecres']
        flux = cube['fluxb']
        ivar = cube['ivarb']
        smod = cube['modelb']
        predisp = cube['predispb']
        cz = cube['cz']
        stellar_vfield = cube['stellar_vfieldb']
        binid = cube['binidb']

        dum1, dum2, plate, ifu = drp[qq].split('-')
        plateifu = plate+'-'+ifu

        print("Running fitwrapper for "+plateifu)

        if(FIT_FLG==0):
            fitwrapper = NaImcmc_fitwrapper.NaImcmc_fitwrapper(plateifu, wave, prespecres, flux, ivar, smod, predisp, cz, stellar_vfield, binid,
                                                               blim, rlim, fitlim, outdir=outdir, FIT_FLG=0)
        elif(FIT_FLG==1):
            fitwrapper = NaImcmc_fitwrapper.NaImcmc_fitwrapper(plateifu, wave, prespecres, flux, ivar, smod, predisp, cz, stellar_vfield, binid,
                                                               blim, rlim, fitlim, outdir=outdir, FIT_FLG=1)
        elif(FIT_FLG==2):
            fitwrapper = NaImcmc_fitwrapper.NaImcmc_fitwrapper(plateifu, wave, prespecres, flux, ivar, smod, predisp, cz, stellar_vfield, binid,
                                                               blim, rlim, fitlim, outdir=outdir, FIT_FLG=2)
        
    
#def main(flg):
def main():

    
    script = sys.argv[0]
    flg = int(sys.argv[1])

    #pdb.set_trace()
    if (flg==0):
        select_fitsample(outfil=None, MORPH_FLG='GZ', FIT_FLG=0, INCLUDE_ALL_FLG=1)

    if (flg==1):
        # This option pretty much out of commission
        run_ewcheck(infil=None, outfil=None, FIT_FLG=0)

    if (flg==2):
        setup_jobs(infil=None, outfilroot=None, scriptfil=None, FIT_FLG=0)
        
    if (flg==3):
        infil = sys.argv[2]
        outdir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/MCMC/SQUARE2.0/2020jul20/FluxFit/'
        run_mcmc(infil=infil, outdir=outdir, FIT_FLG=0)

    if (flg==4):
        # Fit stellar model
        select_fitsample(outfil=None, FIT_FLG=1)

    if (flg==5):
        # Fit stellar model
        setup_jobs(infil=None, outfilroot=None, scriptfil=None, FIT_FLG=1)

    if(flg==6):
        # Fit stellar model
        infil = sys.argv[2]
        run_mcmc(infil=infil, outdir=None, FIT_FLG=1)

    if(flg==7):
        # Divide out stellar model before fitting
        select_fitsample(outfil=None, FIT_FLG=2)

    if(flg==8):
        # Divide out stellar model before fitting
        setup_jobs(infil=None, outfilroot=None, scriptfil=None, FIT_FLG=2)

    if(flg==9):
        # Divide out stellar model before fitting
        infil = sys.argv[2]
        run_mcmc(infil=infil, outdir=None, FIT_FLG=2)


            
        
# Command line execution
#if __name__ == '__main__':

#    flg_analy = 0
    #flg_analy += 2**0  ## Run selection
    #flg_analy += 2**1  ## Run EW culling
#    flg_analy += 2**2
    #flg_analy += 2**3  ## Run MCMC
    
#main(flg_analy)
main()
