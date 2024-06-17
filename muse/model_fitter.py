#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copied from DFM's script to fit a line

from __future__ import print_function

import emcee
import corner
import lnlikelihood
import model_NaI
import math
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
import pdb
from IPython import embed

class model_fitter:

    #def __init__(self,data,guesses):
    def __init__(self,data,guesses,linetimefil=None):

        lamred_guess, logN_guess, bD_guess, Cf_guess = guesses
        self.wave = data['wave']
        self.flux = data['flux']
        self.err = data['err']
        self.velres = data['velres']

        # MCMC setup
        self.sampndim = 4
        self.sampnwalk = 100
        self.nsteps = 1100
        self.burnin = 1000
        
        ## The following is the setup for the 2020jul20 run
        ## -- this is now canceled
        #self.nsteps = 200
        #self.burnin = 150
        self.theta_guess = [lamred_guess, logN_guess, bD_guess, Cf_guess]

        if(linetimefil==None):
            self.linetimefil = None
        else:
            self.linetimefil = linetimefil

        
    def maxlikelihood(self):

        """

        Calculate the maximum likelihood model

        """

        chi2 = lambda *args: -2 * lnlikelihood.lnlike(*args)       
        result = op.minimize(chi2, self.theta_guess, args=(self.wave, self.flux, self.err, self.velres))

        self.theta_ml = result["x"]



    def mcmc(self):

        """
        
        Set up the sampler.
        Then run the chain and make time plots for inspection

        """

        ndim = self.sampndim
        nwalkers = self.sampnwalk
        #startpoint = self.theta_ml
        startpoint = self.theta_guess
        pos = [startpoint + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlikelihood.lnprob,
                                        args=(self.wave, self.flux, self.err, self.velres))

        # Clear and run the production chain.
        print("Running MCMC...")
        sampler.run_mcmc(pos, self.nsteps, rstate0=np.random.get_state())
        print("Done.")

        pl.clf()

        if(self.linetimefil != None):
            
            fig, axes = pl.subplots(ndim, 1, sharex=True, figsize=(8, 9))
            label_list = ['$\lambda_{red}$', 'log N', '$b_D$', '$C_f$']

            for ind in range(0,ndim):
                axes[ind].plot(sampler.chain[:, :, ind].T, color="k", alpha=0.4)
                axes[ind].yaxis.set_major_locator(MaxNLocator(5))
                axes[ind].axhline(startpoint[ind], color="#888888", lw=2)
                axes[ind].set_ylabel(label_list[ind])


                fig.tight_layout(h_pad=0.0)
                pl.show()
                #fig.savefig(self.linetimefil, format='pdf')

        # get the samples
        burnin = self.burnin
        samples_burnin = sampler.chain[:, burnin:, :].reshape((-1, ndim))
        # return sample shape is (nwalkers,nsteps,ndim)
        self.samples = np.transpose(sampler.get_chain(),axes=(1,0,2))

        # Compute the quantiles.
        theta_mcmc = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                              zip(*np.percentile(samples_burnin, [16, 50, 84], axis=0))))
        
    
        self.theta_percentiles = theta_mcmc
        print("""MCMC result:""")
        #pdb.set_trace()
        for ind in range(0,ndim):
            print(""" par {0} = {1[0]} +{1[1]} -{1[2]}""".format(ind, theta_mcmc[ind]))
