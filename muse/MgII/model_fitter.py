from __future__ import print_function

import emcee
#import corner
import lnlikelihood
import MgII_modeling
import math
import numpy as np
import numpy.ma as ma
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
import pdb
import corner

class modeling:

    def __init__(self, data, guesses, specres, corner=None, linetimefil=None, savespec=None, printresult=False, showplot=False, skipnorm=False):
        
        ## if spectrum has masked data (flux==0), mask it
        if any(data['flux']==0):
            flux = ma.masked_equal(data['flux'],0)
            error = ma.array(data['err'], mask=flux.mask)
            wavelength = ma.array(data['wavelength'], mask=flux.mask)
        else:
            flux = data['flux']
            error = data['err']
            wavelength = data['wavelength']

        ## put spectrum into restframe
        wavelength = wavelength / (1+data['z'])
        
        if skipnorm:
            wave = wavelength
            err = error
        else:
            ## normalize and slice spectrum around absorption
            wave, flux, err = MgII_modeling.continuum_normalize(flux, wavelength, error)

        ## initialize object values
        lamred_guess, logN_guess, bD_guess, Cf_guess = guesses
        self.wave = wave
        self.flux = flux
        self.err = err
        self.z = data['z']
        self.fwhm = MgII_modeling.get_fwhm(data,specres)
        
        #self.plot = plot

        # MCMC setup
        self.sampndim = 4
        self.sampnwalk = 100
        #self.nsteps = 500
        #self.burnin = 400
        self.nsteps = 600
        self.burnin = 500

        #self.nsteps = 200
        #self.burnin = 150
        self.theta_guess = [lamred_guess, logN_guess, bD_guess, Cf_guess]
        
        ## for plotting the spectrum with model
        self.savespec = savespec
        ## for plotting the mcmc result
        self.printresult = printresult
        ## output the plot
        self.showplot = showplot
        ## for saving corner plots
        self.corner = corner
        
        ## for plotting sampler time chain
        if(linetimefil==None):
            self.linetimefil = None
        else:
            self.linetimefil = linetimefil

        
    ## max likelihood as 1/2chisq; P = exp(-0.5chisq)
    def maxlikelihood(self):

        """

        Calculate the maximum likelihood model

        """

        sol = 2.998e5    # km/s
        transinfo = MgII_modeling.transitions()

        vlim = 400.0     # km/s
        lamlim1 = -1.0 * (vlim * transinfo['lamred0'] / sol) + transinfo['lamred0']
        lamlim2 = (vlim * transinfo['lamred0'] / sol) + transinfo['lamred0']

        logNlim1 = 10.0
        logNlim2 = 18.0

        bDlim1 = 10.0
        bDlim2 = 200.0

        Cflim1 = 0.0
        Cflim2 = 1.0
        
        chi2 = lambda *args: -2 * lnlikelihood.lnlike(*args)       
        result = op.minimize(chi2, self.theta_guess, args=(self.wave, self.flux, self.err, self.fwhm), 
                             bounds=((lamlim1, lamlim2),(logNlim1,logNlim2),(bDlim1,bDlim2),(Cflim1,Cflim2)))

        self.theta_ml = result.x


    def mcmc(self):

        """
        
        Set up the sampler.
        Then run the chain and make time plots for inspection

        """

        self.maxlikelihood()
        ndim = self.sampndim
        nwalkers = self.sampnwalk
        pos = [self.theta_ml + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlikelihood.lnprob,
                                        args=(self.wave, self.flux, self.err, self.fwhm))

        # Clear and run the production chain.        
        sampler.run_mcmc(pos, self.nsteps, rstate0=np.random.get_state())#, progress=True)


        if(self.linetimefil != None):
            
            fig, axes = pl.subplots(ndim, 1, sharex=True, figsize=(5,6))
            label_list = ['$\lambda_{red}$', 'log N', '$b_D$', '$C_f$']

            for ind in range(0,ndim):
                axes[ind].plot(sampler.chain[:, :, ind].T, color="k", alpha=0.4)
                axes[ind].yaxis.set_major_locator(MaxNLocator(5))
                axes[ind].axhline(self.theta_ml[ind], color="#888888", lw=2)
                axes[ind].set_ylabel(label_list[ind])
                


            fig.tight_layout(h_pad=0.0)
            fig.savefig(self.linetimefil, dpi=200)
            pl.close(fig)


        burnin = self.burnin
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
        self.samples = samples

        # Compute the quantiles.
        theta_mcmc = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0))))
        
    
        self.theta_percentiles = theta_mcmc
        
        
        lambdas = samples[:,0]
        self.prob = len(lambdas[lambdas>2803.53])/len(lambdas)
        
        thetas = (theta_mcmc[0][0], theta_mcmc[1][0], theta_mcmc[2][0], theta_mcmc[3][0])
        #self.ews = MgII_modeling.compute_EWs(self.wave, self.flux, self.err, thetas, self.fwhm)
        
        if self.corner is not None:
            
            labels = [r"$\lambda_{red}$",r"log $N$",r"$b_D$",r"$C_f$"]
            fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.50, 0.84], show_titles=True, 
                               title_fmt = '.3f', title_kwargs={"fontsize":12}, fill_contours=True)
            fig.savefig(self.corner,dpi=500,bbox_inches='tight')
            pl.close(fig)
            
            
            
        if self.printresult:
            print("""MCMC result:""")
            #pdb.set_trace()
            for ind in range(0,ndim):
                print(""" par {0} = {1[0]} +{1[1]} -{1[2]}""".format(ind, theta_mcmc[ind]))
            
            
        
    
    ## plots spectrum with model for optimal parameters
    def plot_model(self,inlinedpi=100):
        theta = self.theta_percentiles[0][0], self.theta_percentiles[1][0], self.theta_percentiles[2][0], self.theta_percentiles[3][0]
        modwav,modflx = MgII_modeling.model_MgII(theta,self.fwhm,self.wave)
        
        fig, ax = pl.subplots(figsize=(5,5),dpi=inlinedpi)
        ax.plot(self.wave, self.flux, drawstyle = 'steps-mid', color='k')
        ax.plot(modwav, modflx, color='lime', label='Model')
        ax.plot(modwav, modflx, color='gray', lw=2.75, zorder=1)
        #pl.legend(frameon=False)
        #pl.tick_params(top=True,right=True,which='both',direction='in')
        #pl.minorticks_on()
        ax.set_xlim(2770,2820)
        ax.set_ylim(self.flux.min()-0.5, self.flux.max()+0.5)
        ax.set_xlabel('Wavelength $\mathrm{\AA}$')
        ax.set_ylabel('Normalized Flux')
        ax.text(0.05,0.9, r'$\lambda_r$ = {:.1f}, log$N$ = {:.1f}, $b_D$ = {:.1f}, $C_f$ = {:.1f}'.format(theta[0],theta[1],theta[2],theta[3]), fontsize='small',transform=ax.transAxes)
        if self.savespec is not None:
            fig.savefig(self.savespec,dpi=350)
            
        if self.showplot:
            pl.show()
        
        else:
            pl.close()
        