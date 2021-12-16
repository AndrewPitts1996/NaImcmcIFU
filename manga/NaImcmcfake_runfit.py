import numpy as np
import scipy.special as sp
import math
import sys
import os
import fnmatch
from linetools.spectra.xspectrum1d import XSpectrum1D
from linetools.spectra import convolve as lsc
from astropy import units as u
from astropy.units import Quantity
from astropy.table import Table, Column, MaskedColumn
from astropy.io import ascii

import json
import pdb

import matplotlib.pyplot as pl
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
pl.rcParams['font.family'] = 'stixgeneral'
pl.rcParams['mathtext.fontset'] = 'stix'

import model_NaI
import model_fitter

def readplan_fakefit(infil, s2n=None, setup_type=None):

    table = ascii.read(infil)
    plate_all = table['plate']
    ifu_all = table['ifu']
    binid_all = table['binid']

    if(s2n<0.0):
        s2n_all = table['s2n']

    for nn in range(len(plate_all)):

        plate = plate_all[nn]
        ifu = ifu_all[nn]
        binid = binid_all[nn]

        if(s2n<0.0):
            thiss2n = s2n_all[nn]
        else:
            thiss2n = s2n

        run_fakefit(plate=plate, ifu=ifu, binid_for_fit=binid, s2n=thiss2n, setup_type=setup_type)
        

def run_fakefit(plate=None, ifu=None, binid_for_fit=None, s2n=None, setup_type=None):

 
    if((plate==None) | (ifu==None) | (binid_for_fit==None)):
        print("Need to select bin properly!")
        pdb.set_trace()
        
    if(s2n==None):
        print("Need to request a S/N level")
        pdb.set_trace()

    if(setup_type=='NoiseOnly'):
        setup_type_root = 'noiseonly'
    elif(setup_type=='AddNoise'):
        setup_type_root = 'addnoise'
    else:
        print('Setup type not defined!')
        
    #if(s2n>0.0):
    s2n_str = '{0:.0f}'.format(s2n)
    #else:
    #    s2n_str = 'Variable'
        
    #s2n_str = '{0:.0f}'.format(s2n)
    outdir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/FakeSets/'+setup_type+'/Representative_logN14p4_16p0Test'
    infil_npy = outdir + '/'+str(plate)+'-'+str(ifu)+'-'+str(binid_for_fit)+'-'+setup_type_root+'-fakeset-SN'+s2n_str+'.npy'
    outfil_npy = outdir + '/'+str(plate)+'-'+str(ifu)+'-'+str(binid_for_fit)+'-'+setup_type_root+'-fakeset-SN'+s2n_str+'-logN14p4_16p0-wsamples.npy'
    outfil_linetime = outdir + '/'+str(plate)+'-'+str(ifu)+'-'+str(binid_for_fit)+'-'+setup_type_root+'-fakeset-SN'+s2n_str+'-logN14p4_16p0-linetime.pdf'

    dictList = np.load(infil_npy, allow_pickle=True)
    dictsampList = []

    print("Beginning fits for ", infil_npy)
    #pdb.set_trace()
    for qq in range(len(dictList)):

        print("Fitting realization number ", qq)

        data = dictList[qq]
        dictforfit = {'wave':data['wave'], 'flux':data['flux_fkline'], 'err':data['err_fkline'], 'velres':data['velres']}

        # Guess good model parameters
        logN = 14.5
        #bD = 200.0
        bD = 100.0
        Cf = 0.5
        lamred = 5897.5581
        theta_guess = lamred, logN, bD, Cf

        print("Running fit for model with Cf = ", data['Cf'], ", logN = ", data['logN'], ", v = ", data['v'])
        datfit = model_fitter.model_fitter(dictforfit, theta_guess, linetimefil=outfil_linetime)
        # Run the MCMC
        datfit.mcmc()
        # Read in MCMC percentiles
        lamred_mcmc, logN_mcmc, bD_mcmc, Cf_mcmc = datfit.theta_percentiles

        data['samples'] = datfit.samples
        data['theta_percentiles'] = datfit.theta_percentiles

        dictsampList.append(data)

    np.save(outfil_npy, dictsampList)

def makeplan_fakefit(s2n=None, setup_type=None):

    if(s2n>0.0):
        s2n_str = '{0:.0f}'.format(s2n)
        s2n_searchstr = s2n_str
    else:
        s2n_str = 'Variable'
        s2n_searchstr = '*'
    plan_dir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/plan_files/'
    outfilroot = plan_dir + 'plan-logN14p4_16p0-'+setup_type+'-SN'+s2n_str
    scriptfil = plan_dir + 'NaImcmcmfake_runscript-logN14p4_16p0-'+setup_type+'-SN'+s2n_str

    if(setup_type=='NoiseOnly'):
        setup_type_root = 'noiseonly'
    elif(setup_type=='AddNoise'):
        setup_type_root = 'addnoise'
    else:
        print('Setup type not defined!')
    
    # Search for .npy files to fit
    fakesetsdir = '/data/home/krubin/Projects/MaNGA/NaImcmc/Analysis/LineProfileSims/FakeSets/'+setup_type+'/Representative_logN14p4_16p0Test/'

    pattern = '*'+setup_type_root+'-fakeset-SN'+s2n_searchstr+'.npy'
    #fitpattern = '-'+setup_type_root+'-fakeset-SN'+s2n_str+'-wsamples.npy'

    findall_fakesets = []
    for root, dirs, files in os.walk(fakesetsdir):
        for filename in fnmatch.filter(files,pattern):
            findall_fakesets.append(filename)
    print("Found this many fake datasets to fit:", len(findall_fakesets))


    findall_unfit = []
    plate_unfit = []
    ifu_unfit = []
    binid_unfit = []
    s2n_unfit = []
    for nn in range(len(findall_fakesets)):

        nnfakeset = findall_fakesets[nn]
        nnfil = nnfakeset.split('-')
        found_plate = int(nnfil[0])
        found_ifu = int(nnfil[1])
        found_binid = int(nnfil[2])
        found_plateifubin = "{:.0f}".format(found_plate)+"-"+"{:.0f}".format(found_ifu)+"-"+"{:.0f}".format(found_binid)
        found_s2n_init = nnfil[5]
        tmp = found_s2n_init.split('.')
        found_s2n = int(tmp[0][2:])
        #pdb.set_trace()
        fitpattern = '-'+setup_type_root+'-fakeset-SN'+"{:.0f}".format(found_s2n)+'-logN14p4_16p0-wsamples.npy'
        outfitsfil = found_plateifubin+fitpattern

        fit_exists = os.path.exists(fakesetsdir+outfitsfil)
        #pdb.set_trace()
        if(fit_exists):
            print("Skipping plate-ifu-binid ", found_plateifubin)

        else:
            findall_unfit.append(findall_fakesets[nn])
            plate_unfit.append(found_plate)
            ifu_unfit.append(found_ifu)
            binid_unfit.append(found_binid)
            s2n_unfit.append(found_s2n)

    
    plate_unfit = np.array(plate_unfit)
    ifu_unfit = np.array(ifu_unfit)
    binid_unfit = np.array(binid_unfit)
    s2n_unfit = np.array(s2n_unfit)
    
    ngal = len(findall_unfit)
    #pdb.set_trace()
    ncores = 5
    nlabeloff = 0
    nperfil = int(ngal / ncores)

    for nn in range(ncores+1):
        
        fil = outfilroot+'-'+str(nn+nlabeloff)+'.plan'
        ind = np.arange(nn*nperfil, (nn+1)*nperfil)

        if(max(ind)>=ngal):
            minind = min(ind)
            maxind = ngal-1
            ind = np.arange(minind, maxind)
        table = Table([plate_unfit[ind], ifu_unfit[ind], binid_unfit[ind], s2n_unfit[ind]], \
                      names=['plate', 'ifu', 'binid', 's2n'])
        ascii.write(table, fil, overwrite=True)
        #pdb.set_trace()
        
    f = open(scriptfil, "w")
    f.write("#!/bin/sh\n")

    for nn in range(ncores+1):

        fil = outfilroot+'-'+str(nn+nlabeloff)+'.plan'
        jobname = 'NaImcmcfake'+str(nn+nlabeloff)
        f.write('screen -mdS '+jobname+' sh -c "python NaImcmcfake_runfit.py 1 '+fil+'"\n')
    f.close()

    pdb.set_trace()
            
    #run_fakefit(plate=7443, ifu=6102, binid_for_fit=63)
    #run_fakefit(plate=8440, ifu=6104, binid_for_fit=5)
    #fake_setup(plate=8440, ifu=6104, binid_for_fit=30)
    #fake_setup(plate=8440, ifu=6104, binid_for_fit=35)
    #run_fakefit(plate=9506, ifu=3701, binid_for_fit=8)
    #run_fakefit(plate=8982, ifu=6104, binid_for_fit=26)
    #fake_setup(plate=8982, ifu=6104, binid_for_fit=53)
    #run_fakefit(plate=7968, ifu=9101, binid_for_fit=1)
    #run_fakefit(plate=7968, ifu=9101, binid_for_fit=9)
    #fake_setup(plate=7968, ifu=9101, binid_for_fit=121)
    #fake_setup(plate=8549, ifu=12705, binid_for_fit=8)
    #fake_setup(plate=8549, ifu=12705, binid_for_fit=45)

def main():

    script = sys.argv[0]
    flg = int(sys.argv[1])

    s2n = 50.0
    #s2n = -1.0
    setup_type='NoiseOnly'
    
    if(flg==0):
        makeplan_fakefit(s2n=s2n,setup_type=setup_type)
    
    if(flg==1):
        infil = sys.argv[2]
        readplan_fakefit(infil=infil,s2n=s2n,setup_type=setup_type)

main()
