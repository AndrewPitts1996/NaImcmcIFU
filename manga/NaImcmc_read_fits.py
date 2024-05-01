from __future__ import print_function

import math
import numpy as np
from astropy.io import fits
from astropy.table import Table
import warnings
from mangadap.util.fileio import channel_dictionary
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
pl.rcParams['font.family'] = 'stixgeneral'
pl.rcParams['mathtext.fontset'] = 'stix'
from matplotlib.ticker import MaxNLocator
from mangadap.dapfits import DAPCubeBitMask
from mangadap.util.fitsutil import DAPFitsUtil
from mangadap.util.drpfits import DRPFits
from mangadap.proc.reductionassessments import ReductionAssessment
from mangadap.proc.spectralstack import SpectralStackPar, SpectralStack
from mangadap.proc import spatialbinning
from mangadap.proc.spatiallybinnedspectra import SpatiallyBinnedSpectra, SpatiallyBinnedSpectraDef
import pdb
from IPython import embed

def NaImcmc_read_fits(fits_dap_map, fits_dap_cube, fits_drp):

    
    hdu_maps = fits.open(fits_dap_map)
    hdu_cube = fits.open(fits_dap_cube)

    hdu_drp = fits.open(fits_drp)
    predisp = hdu_drp['PREDISP'].data    # Broadened pre-pixel dispersion solution (1sigma LSF in Angstroms)
    predisp_dapcube = np.full(predisp.shape[::-1], -1.0, dtype=np.float)
    prespecres = hdu_drp['PRESPECRES'].data
    nwave = predisp.shape[0]
    wave = hdu_cube['WAVE'].data

    binid = hdu_maps['BINID'].data[1,:,:]
    bin_indx = binid.ravel()
    unique_bins, reconstruct = np.unique(bin_indx, return_inverse=True)

    # Get the valid bins
    indx = bin_indx > -1
    nbins = np.max(unique_bins)+1
    
    predisp_array = np.full((nbins,nwave),-1.0, dtype=np.float) 

    for ii in range(nbins):

        realbin = (unique_bins==ii)
        if(np.any(realbin)==True):
        
            find = (binid==ii)  
            mean_predisp_chunk = np.mean(predisp[:,find], axis=1)
            predisp_array[ii,:] = mean_predisp_chunk
        #else:
            #print("Skipping ii = ", ii)

    predisp_dapcube.reshape(-1,nwave)[indx,:] = predisp_array[unique_bins[reconstruct[indx]],:]
    # Above, unique_bins used to be unique_indx
    predisp_dapcubeT = np.transpose(predisp_dapcube, (2,0,1))


    #### TESTING #####
    
    #embed()
    #exit()

    #binidb = np.reshape(binid, [binid.shape[0] * binid.shape[1]])
    #predisp_dapcubeb = np.reshape(predisp_dapcubeT, [predisp_dapcubeT.shape[0], predisp_dapcubeT.shape[1] * predisp_dapcubeT.shape[2]])
    #predispb = np.reshape(predisp, [predisp.shape[0], predisp.shape[1] * predisp.shape[2]])
    
    #pltct = 0
    #nperpage=20
    #outfil = '../../Figures/specres_test.pdf'
    #with PdfPages(outfil) as pdf:
    # 
    #    for qq in range(nbins):
    #
    #        if((qq % nperpage)==0):
    # 
    #            pl.figure(figsize=(13.0, 11.0))
    #            pltct =0
    #            
    #        ax = pl.subplot(5,4,pltct+1)
    # 
    #        wh = np.where(binidb==qq)
    #        wh = wh[0]
    #        print(wh)
    # 
    #        for pp in range(len(wh)):
    #            ax.plot(wave, predispb[:,wh[pp]], alpha=0.5, color='gray')
    #
    #        for pp in range(len(wh)):
    #            ax.plot(wave, predisp_dapcubeb[:,wh[pp]], alpha=0.5, color='red', lw=0.2)
    # 
    #        ax.text(0.05, 0.05, "{:.1f}".format(qq), ha='left', va='center', transform=ax.transAxes)
    #        if((pltct==nperpage-1) | (qq==(nbins-1))):
    # 
    #            pl.tight_layout()
    #            pdf.savefig()
    #        pltct = pltct + 1
    
    #embed()
    #exit()
  
        
    #sres = hdu_drp['SPECRES'].data        # Original, median LSF for all fibers
        
    hdu_hdr = hdu_cube['PRIMARY'].header
    cz = hdu_hdr['SCINPVEL']  # in km/s, z = v/c
    emlc = channel_dictionary(hdu_maps, 'EMLINE_GVEL')


    # Gaussian profile velocity (21 channels for different emission lines; km/s relative to NSA redshift)
    mask_ext = hdu_maps['EMLINE_GVEL'].header['QUALDATA']
    emission_vfield = np.ma.MaskedArray(hdu_maps['EMLINE_GVEL'].data[emlc['Ha-6564'],:,:],
                                        mask=hdu_maps[mask_ext].data[emlc['Ha-6564'],:,:] > 0)
    emission_vfield_ivar = np.ma.MaskedArray(hdu_maps['EMLINE_GVEL_IVAR'].data[emlc['Ha-6564'],:,:],
                                            mask=hdu_maps[mask_ext].data[emlc['Ha-6564'],:,:] > 0)


    mask_ext = hdu_maps['STELLAR_VEL'].header['QUALDATA']
    stellar_vfield = np.ma.MaskedArray(hdu_maps['STELLAR_VEL'].data,
                                    mask=hdu_maps[mask_ext].data > 0)
    stellar_vfield_ivar = np.ma.MaskedArray(hdu_maps['STELLAR_VEL_IVAR'].data,
                                            mask=hdu_maps[mask_ext].data > 0)

    bm = DAPCubeBitMask()
    flux = np.ma.MaskedArray(hdu_cube['FLUX'].data,
                            mask=bm.flagged(hdu_cube['MASK'].data,
                            [ 'IGNORED', 'FLUXINVALID', 'IVARINVALID', 'ARTIFACT' ]))
    ivar = np.ma.MaskedArray(hdu_cube['IVAR'].data,
                            mask=bm.flagged(hdu_cube['MASK'].data,
                            [ 'IGNORED', 'FLUXINVALID', 'IVARINVALID', 'ARTIFACT' ]))

    model = np.ma.MaskedArray(hdu_cube['MODEL'].data,
                            mask=bm.flagged(hdu_cube['MASK'].data, 'FITFAILED'))

  
    fluxb = np.reshape(flux, [flux.shape[0], flux.shape[1] * flux.shape[2]])
    ivarb = np.reshape(ivar, [ivar.shape[0], ivar.shape[1] * ivar.shape[2]])
    modelb = np.reshape(model, [model.shape[0], model.shape[1] * model.shape[2]])
    predispb = np.reshape(predisp_dapcubeT, [predisp_dapcubeT.shape[0], predisp_dapcubeT.shape[1] * predisp_dapcubeT.shape[2]])
    stellar_vfieldb = np.reshape(stellar_vfield, [stellar_vfield.shape[0] * stellar_vfield.shape[1]])
    stellar_vfield_ivarb = np.reshape(stellar_vfield_ivar, [stellar_vfield_ivar.shape[0] * stellar_vfield_ivar.shape[1]])
    emission_vfieldb = np.reshape(emission_vfield, [emission_vfield.shape[0] * emission_vfield.shape[1]])
    emission_vfield_ivarb = np.reshape(emission_vfield_ivar, [emission_vfield_ivar.shape[0] * emission_vfield_ivar.shape[1]])
    binidb = np.reshape(binid, [binid.shape[0] * binid.shape[1]])
        
    return {'cz':cz, 'stellar_vfieldb':stellar_vfieldb, 'stellar_vfield_ivarb':stellar_vfield_ivarb,
            'emission_vfieldb':emission_vfieldb, 'emission_vfield_ivarb':emission_vfield_ivarb, 'binidb':binidb,
            'wave':wave, 'prespecres':prespecres, 'fluxb':fluxb, 'ivarb':ivarb, 'modelb':modelb, 'predispb':predispb}
    
    

#### TESTING #####
#def main():
#
#    saspath = '/data/manga/sas/mangawork/'
#    drp = 'manga/spectro/redux/MPL-9/7443/stack/manga-7443-12702'
#    dap = 'users/u6021943/Alt-DAP/SQUARE2.0_2020jan09/SQUARE2.0-MILESHC-MASTARHC/7443/12702/manga-7443-12702'
#
#    fits_drp = saspath + drp + '-LOGCUBE.fits.gz'
#    fits_dap_cube = saspath + dap + '-LOGCUBE-SQUARE2.0-MILESHC-MASTARHC.fits.gz'
#    fits_dap_map = saspath + dap + '-MAPS-SQUARE2.0-MILESHC-MASTARHC.fits.gz'
#    plate = 7443
#    ifu = 12702
#
#    cube = NaImcmc_read_fits(fits_dap_map, fits_dap_cube, fits_drp)
#
#main()
