#!/usr/bin/env python
# coding: utf-8

from kcwi_jnb import cube
import numpy as np
from astropy import wcs
from astropy.io import fits
from kcwi_jnb import transform
from matplotlib import pyplot as plt
from bokeh.plotting import figure, show
from bokeh.layouts import row
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, CustomJS, Slider,Div,SingleIntervalTicker,ColorBar
from bokeh.models.mappers import LinearColorMapper, LogColorMapper
from bokeh.models import Step, Label,Text
from astropy.table import Table
from astropy import units as u
import time
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
from bokeh.io import curdoc
from bokeh import events
from bokeh.models import Span,Square,HoverTool,Range1d
import colorcet as cc

import math
import os
import sys
import fnmatch
import scipy.optimize as op
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn,unique
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from linetools.spectra.xspectrum1d import XSpectrum1D
import model_NaI
import model_fitter
import continuum_normalize_NaI
#import NaImcmc_read_fits
#import NaImcmc_fitwrapper
#import ew_NaI_allspax
import pdb
import corner 
import glob
import os

import extinction
from bokeh.plotting import figure, output_file, save
import bebi103
import emcee

from linetools.spectralline import AbsLine
from linetools.isgm.abscomponent import AbsComponent
from astropy import units as u
# import
try:
    import seaborn as sns; sns.set_style("white")
except:
    pass

import astropy.units as u
from linetools.spectralline import AbsLine
from linetools.isgm import utils as ltiu
from linetools.analysis import absline as laa
from linetools.spectra import io as lsio

from linetools.isgm.abscomponent import AbsComponent

import imp
lt_path = imp.find_module('linetools')[1]

from astropy.table import Table
import time 

# MUSE cube directory
main_dir = '/Users/erickaguirre/mangadap/examples/'
main_dir2 = '/Users/erickaguirre/Desktop/SDSU_Research/Getting_used_to_MaNGA_DAP/'
main_dir3 = '/Users/erickaguirre/Desktop/DAP_outputs/'

# # NGC 1042
# NGC1042_output_dir = 'output0.6_NGC1042/'
# NGC1042_cube_dir = 'SQUARE0.6-MILESHC-MASTARSSP/1/2/'
#
# NGC1042_cube0_6_file = main_dir3+ NGC1042_output_dir+ NGC1042_cube_dir+'manga-1-2-LOGCUBE-SQUARE0.6-MILESHC-MASTARSSP.fits'
# NGC1042_cube0_6 = fits.open(NGC1042_cube0_6_file)


# # NGC 2089
# NGC0289_output_dir = 'output0.6_NGC0289/'
# NGC0289_cube_dir = 'SQUARE0.6-MILESHC-MASTARSSP/6/1/'
#
# NGC0289_cube0_6_file = main_dir3+ NGC0289_output_dir+ NGC0289_cube_dir+'manga-6-1-LOGCUBE-SQUARE0.6-MILESHC-MASTARSSP.fits'
# NGC0289_cube0_6 = fits.open(NGC0289_cube0_6_file)

# NGC 4030 
NGC4030_output_dir = 'output2.0_NGC4030/'
NGC4030_cube_dir = 'SQUARE2.0-MILESHC-MASTARSSP/1/1/'
# log cube
NGC4030_cube0_6_file = main_dir3+ NGC4030_output_dir+ NGC4030_cube_dir+'manga-1-1-LOGCUBE-SQUARE2.0-MILESHC-MASTARSSP.fits'
NGC4030_cube0_6 = fits.open(NGC4030_cube0_6_file)
# log maps
NGC4030_map_file = main_dir3+ NGC4030_output_dir+ NGC4030_cube_dir+'manga-1-1-MAPS-SQUARE2.0-MILESHC-MASTARSSP.fits.gz'
NGC4030_map = fits.open(NGC4030_map_file)

# obtain stellar velocity, stellar dispersion and bin id data from the galaxy
stellar_vel =  NGC4030_map['STELLAR_VEL'].data
stellar_sigma = NGC4030_map['STELLAR_SIGMA'].data
binid_map = NGC4030_map['BINID'].data[0]

# stellar velocity map
ppxf_v_map = NGC4030_map['STELLAR_VEL'].data

# create parameter range
sc_bins = np.array([265, 406, 792])  # might increase
logNs = np.arange(12.0, 15, 1)  # will be (12.0,17.0,0.5)
b_ds = np.array([50, 100])  # np.arange(5,250,50)
Cfs = np.array([0.15, 0.3])  # might increase these two parameters as well
vels_gas = np.array([-100, -50, 50, 100])  # will be np.arange(-200,200,5)? will be a lot of bins to fit though
spx_snrs = np.array([57, 165])  # different SNR
realizations = np.arange(0, 5, 1)
tot_bins = logNs.size * b_ds.size * vels_gas.size * Cfs.size * sc_bins.size * spx_snrs.size * realizations.size

# create an astropy table holding all the parameters
fake_binids = []
sc_binids = []
column_densities = []
doppler_bs = []
covering_fracs = []
velocities = []
spaxel_snrs = []
realizations_all = []

i = 0
for N in logNs:
    for b_d in b_ds:
        for C_f in Cfs:
            for v in vels_gas:
                for sc_bin in sc_bins:
                    for spx_snr in spx_snrs:
                        for realization in realizations:
                            fake_binids.append(i)
                            column_densities.append(N)
                            doppler_bs.append(b_d)
                            covering_fracs.append(C_f)
                            velocities.append(v)
                            sc_binids.append(sc_bin)
                            spaxel_snrs.append(spx_snr)
                            realizations_all.append(realization)
                            i += 1

param_table = Table([fake_binids, column_densities, doppler_bs, covering_fracs,
                     velocities, sc_binids, spaxel_snrs, realizations_all],
                    names=('fake_bin_id', 'logN', 'b_D', 'C_f', 'gas_velocity',
                           'sc_bin_id', 'spaxel_snr', 'realizations'))
param_table = param_table[param_table['fake_bin_id'] < 1361]

# emission line header dictionary
emline = {}
for k, v in NGC4030_map['EMLINE_GFLUX'].header.items():
    if k[0] == 'C':
        try:
            i = int(k[1:])-1
        except ValueError:
            continue
        emline[v] = i

# H-alpha emission line flux from MUSE cube (1E-17 erg/s/cm^2/spaxel)
# 23rd index = 24 channel
#halpha_map = NGC4030_map['EMLINE_GFLUX'].data[emline['Ha-6564'],:,:]
# H-beta emission line flux from MUSE cube
# obtain flux array from MUSE cube

# Read in binned spectra
# extract cube data  spectrum
hdu_cube = fits.open(NGC4030_cube0_6_file)
flux = hdu_cube['FLUX'].data
# extract cube best-fit model spectrum
model = hdu_cube['MODEL'].data
# extract error spectrum from cube
ivar = hdu_cube['IVAR'].data
error = np.sqrt(1 / ivar)
# observed wavelength
wave = hdu_cube['WAVE'].data

# --------------------------------------------------------------
# obtain gas velocities for spectral bin

# For continuum-normalization around NaI
# wavelength continuum fitting range outside of NaI region
blim = [5850.0,5870.0]
rlim = [5910.0,5930.0]
# wavelength fitting range inside of NaI region
fitlim = [5880.0,5910.0]
# speed of light in km/s
c = 2.998e5
# Na I doublet vacuum absorption wavelengths 
D1 = 5891.582 # in angstroms
D2 = 5897.558 # in angstroms

# sorting function
def sort_key(file_path):
    # individually splits the directory and filename from the path given
    dir_path, filename = os.path.split(file_path)
    # grab only run number from filename
    run = filename[10:16]
    # return only integer run number to sort
    num = int(''.join(i for i in run if i.isdigit()) )
    return num

# directory path
path = '/Users/erickaguirre/Desktop/NaI_MCMC_output/NGC4030_2.0_test_fake_prior4/'
# fits file path
fits_paths = sorted(glob.glob(path+'NGC4030*'), key = sort_key)

# combine all the results from each fits file 
# done in MUSE analysis for one galaxy and create an
# astropy table with all the values for each bin id
binid_all = []
samples_all = []
percentiles_all = []
gas_vel_all = []
vel_sigma_all = []

# rest wavelenth for Na I D2 absorption line
lamrest = 5897.558  # in angstroms

for fits_file in fits_paths:
    file = fits.open(fits_file)
    data_arr = file[1].data

    for i in range(len(data_arr['fake_bin'])):
        binid_all.append(data_arr['fake_bin'][i])
        samples_all.append(data_arr['samples'][i])
        percentiles_all.append(data_arr['percentiles'][i])
        gas_vel_all.append(data_arr['velocities'][i])
        # get uncertainty in gas velocity
        lambda_sigma = np.mean(data_arr['percentiles'][i][0][1:])
        vel_sigma_all.append(lambda_sigma * (c / lamrest))

table_all = Table([binid_all, samples_all, percentiles_all, gas_vel_all, vel_sigma_all],
                  names=('bin', 'samples', 'percentiles', 'velocities', 'vel_sigma'))

# get rid of duplicate bins in the table
table_uniq = unique(table_all, keys=['bin'], keep='last')

# create a map that places the gas velocitiy values 
# to their respective bin ID pixel coordinate
binid_flat = np.arange(0,1444)
gas_vel_flat = np.zeros(len(binid_flat))

for i in range(len(table_uniq['bin'])):
    gas_vel_flat[i] = table_uniq['velocities'][i]

abs_binid_map = np.array(binid_flat).reshape(38,38)
gas_vel_map = np.array(gas_vel_flat).reshape(38,38)
    
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# plot DAP flux against best-fit NaImcmc model

# Need LSF in km/s
# This gives LSF in Ang
# CAUTION: this does not convert air wavelengths
# to vacuum, or account for velocity offset of each bin
redshift = 0.00489 
LSFdir = '/Users/erickaguirre/Desktop/SDSU_Research/Getting_used_to_MaNGA_DAP/'
LSFfil = LSFdir + 'LSF-Config_MUSE_WFM'
configLSF = np.genfromtxt(LSFfil, comments='#')
configLSF_wv_air = configLSF[:,0]
configLSF_res = configLSF[:,1]

# convert to vacuum since LSF is in air
xspec = XSpectrum1D.from_tuple((configLSF_wv_air, 0.0*configLSF_wv_air))
xspec.meta['airvac'] = 'air'
xspec.airtovac()
configLSF_wv_vac = xspec.wavelength.value

configLSF_restwv = configLSF_wv_vac / (1.0+redshift)
whLSF = np.where((configLSF_restwv > fitlim[0]) & (configLSF_restwv < fitlim[1]))

median_LSFAng = np.median(configLSF_res[whLSF[0]])
median_LSFvel = c * median_LSFAng / np.median(configLSF_wv_vac[whLSF[0]])
LSFvel = median_LSFvel


# create fake absorber from parameter input table
def mk_absorber(param_table_row,LSFang):
    # model absorber astropy table
    tab = Table()
    tab['ion_name'] = ['NaI', 'NaI']
    tab['Z'] = [11, 11]
    tab['ion'] = [1, 1]
    tab['Ej'] = [0., 0.] / u.cm
    tab['z_comp'] = [0., 0.]
    tab['logN'] = [param_table_row['logN'], param_table_row['logN']]
    tab['sig_logN'] = [0.1, 0.1]
    tab['flag_logN'] = [1, 1]
    tab['RA'] = 0 * u.deg * np.ones(len(tab))
    tab['DEC'] = 0 * u.deg * np.ones(len(tab))
    tab['vmin'] = -300 * u.km / u.s * np.ones(len(tab))
    tab['vmax'] = 300 * u.km / u.s * np.ones(len(tab))
    tab['b'] = [param_table_row['b_D'], param_table_row['b_D']] * u.km / u.s
    tab['sig_b'] = [5, 5] * u.km / u.s
    tab['vel'] = [param_table_row['gas_velocity'], param_table_row['gas_velocity']] * u.km / u.s
    tab['sig_vel'] = [0.5, 0.5] * u.km / u.s

    complist = ltiu.complist_from_table(tab)

    # add abslines
    wvlim = [5890, 5990] * u.AA
    for comp in complist:
        comp.add_abslines_from_linelist(llist='ISM', wvlim=wvlim, min_Wr=0.01 * u.AA, chk_sep=False)

    from linetools.analysis import voigt as lav
    wv_array = np.arange(5840, 5920, .01) * u.AA
    model_abs = lav.voigt_from_components(wv_array, complist, fwhm= LSFang / .01)
    return model_abs



def get_fake_fluxes(param_row, restwave, flux_bin, err_bin, 
                    mod_bin, blim, rlim, median_LSFAng,LSFvel,table_uniq):
    
    # speed of light in km/s
    c = 2.998e5
    # Na I doublet vacuum absorption wavelengths 
    D1 = 5891.582 # in angstroms
    D2 = 5897.558 # in angstroms
    redshift = 0.00489 

    # normalize the flux from the stellar continuum
    ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, blim, rlim, FIT_FLG=1, smod=mod_bin)
    # have the continuum normalized flux have a slightly larger range than when actually fitting it
    fitlim_ext = [5870.0, 5920.0]

    # Cut out NaI
    select = np.where((ndata['nwave'] > fitlim_ext[0]) & (ndata['nwave'] < fitlim_ext[1]))
    restwave_NaI = ndata['nwave'][select].astype('float64')
    flux_NaI = ndata['nflux'][select].astype('float64')
    err_NaI = ndata['nerr'][select].astype('float64')
    sres_NaI = LSFvel

    # create fake absorber using input model parameters
    model_abs = mk_absorber(param_row,median_LSFAng)
    
    model_abs_wav = model_abs.wavelength.value
    model_abs_flux = model_abs.flux.value
    v = param_row['gas_velocity'][0] # km/s
    # shift in angstroms from velocity input
    shift_lamb = D2 * (v/c)
    fake_new_wv = model_abs_wav + shift_lamb
    
    xspec = XSpectrum1D.from_tuple((fake_new_wv*u.AA,model_abs_flux))

    # rebin fake absorber wavelength using stellar continuum wavelength centered around Na I region
    rebin_xspec = xspec.rebin(restwave_NaI * u.AA)
    rebin_wav = rebin_xspec.wavelength
    rebin_flux = rebin_xspec.flux
    opt_depths = -np.log(rebin_flux)

    # create fake absorber flux with input covering fraction and optical depths
    fake_abs_flux = np.ma.array(1 - param_row['C_f'][0] +
                                (param_row['C_f'][0] * np.exp(-1.0 * opt_depths))).astype('float64')

    # combine fake absorber flux with flux from the stellar continuum
    fake_obs_flux = flux_NaI * fake_abs_flux

    # add Gaussian noise to the data
    fake_obs_xspec = XSpectrum1D.from_tuple((restwave_NaI * u.AA, fake_obs_flux))
    fake_noise_xspec = fake_obs_xspec.add_noise(s2n=param_row['spaxel_snr'][0])

    fake_wav = np.ma.array(fake_noise_xspec.wavelength.value)
    fake_flux = np.ma.array(fake_noise_xspec.flux.value)
    fake_err = np.ma.array(fake_noise_xspec.sig.value)
    
    binid_indx = table_uniq['bin'] == param_row['fake_bin_id']
    fit_bin = model_NaI.model_NaI(table_uniq['percentiles'][binid_indx][0][:, 0], LSFvel, restwave_NaI)
    fit_bin['velwv']=c*((fit_bin['modwv']/D2)-1)
    
    # convert wavelength to velocity
    sc_vel = c * ( (restwave_NaI/D2) - 1)
    fk_abs_vel = c * ( (rebin_wav.value/D2)- 1 )
    fk_obs_vel = c * ( (fake_wav/D2)- 1 )

    # return model absorber, stellar continuum, combined fake observed flux, model, absorption fit
    return sc_vel, flux_NaI, fk_abs_vel,fake_abs_flux, fk_obs_vel, fake_flux,fit_bin
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# corner plot for each spaxel
def corner(binid,table_uniq):
    binid_indx = np.where(table_uniq['bin']== binid)
    # emcee parameters
    ndim = 4
    burnin = 1000
    sampler = table_uniq['samples'][binid_indx][0]
    samples = sampler[:, burnin:, :].reshape((-1, ndim))

    lambda_red = samples[:,0]
    N = samples[:,1]
    b_d = samples[:,2]
    C_f = samples[:,3]
    return lambda_red, N, b_d, C_f
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# text labels for plot 6
def get_text_labels(table_uniq,binid):
    bin_indx = table_uniq['bin'] == binid
    
    gas_vel = table_uniq['velocities'][bin_indx][0]
    gas_vel_sig = table_uniq['vel_sigma'][bin_indx][0]
    gas_vel_text = 'NaI V.: {:.1f} +/- {:.1f} km/s'.format(gas_vel, gas_vel_sig)

    percentiles_lamred = table_uniq['percentiles'][bin_indx][0][0]
    percentiles_logN = table_uniq['percentiles'][bin_indx][0][1]
    percentiles_bd = table_uniq['percentiles'][bin_indx][0][2]
    percentiles_Cf = table_uniq['percentiles'][bin_indx][0][3]

    lamred_50th = percentiles_lamred[0]
    logN_50th = percentiles_logN[0]
    bd_50th = percentiles_bd[0]
    Cf_50th = percentiles_Cf[0]

    lamred_sigma = np.mean([percentiles_lamred[1], percentiles_lamred[2]])
    logN_sigma = np.mean([percentiles_logN[1], percentiles_logN[2]])
    bd_sigma = np.mean([percentiles_bd[1], percentiles_bd[2]])
    Cf_sigma = np.mean([percentiles_Cf[1], percentiles_Cf[2]])

    text_lamred = 'lamred: {:.2f} +/- {:.1e} angstr'.format(lamred_50th, lamred_sigma)
    text_logN = 'logN: {:.2f} +/- {:.1e} cm-2'.format(logN_50th, logN_sigma)
    text_bd = 'b_d: {:.2f} +/- {:.1e} km/s'.format(bd_50th, bd_sigma)
    text_Cf = 'C_f: {:.2f} +/- {:.1e}'.format(Cf_50th, Cf_sigma)

    return gas_vel_text, text_lamred, text_logN, text_bd, text_Cf
# -----------------------------------------------------------------

# get Na I absorption lines w/ respect to galaxy's redshift
z = 0.00489 # NGC 4030 redshift's
# z = 0.00489 # NGC 14042 redshift's


# get data around Na I region and the best-fit model to it
binid = abs_binid_map[20,20]
param_row = param_table[param_table['fake_bin_id'] == binid]
sc_binid = binid_map == param_row['sc_bin_id'][0]

# single bin velocity
binvel = ppxf_v_map[sc_binid][0]
# single flux, error and model spectrum corresponding to that bin
flux_bin = np.ma.array(flux[:, sc_binid][:, 0])
err_bin = np.ma.array(error[:, sc_binid][:, 0])
mod_bin = np.ma.array(model[:, sc_binid][:, 0])

# Determine bin redshift from stellar velocities
bin_z = z + ((1 + z) * (binvel / c))
restwave = wave / (1.0 + bin_z)

sc_vel,sc_flux,fk_abs_vel,fk_abs_flux, fk_obs_vel, fk_obs_flux, fit_bin = get_fake_fluxes(param_row, restwave, flux_bin,
                                                                                           err_bin, mod_bin, blim, rlim, 
                                                                                            median_LSFAng,LSFvel,table_uniq)
fit_vel = c*((fit_bin['modwv']/D2)-1)
fit_flux = fit_bin['modflx']
source6 = ColumnDataSource(data = dict(sc_vel=sc_vel,sc_flux=sc_flux.data,fk_abs_vel=fk_abs_vel,
                                       fk_abs_flux=fk_abs_flux.data.value,
                                       fk_obs_vel=fk_obs_vel.data,fk_obs_flux=fk_obs_flux.data,
                                       fit_vel=fit_vel,fit_flux=fit_flux))

# get corner plot given a binid
params = [r'lambda_red', 'N', 'b_d','C_f']
lambda_red, N, b_d, C_f = corner(binid,table_uniq)
source7 = ColumnDataSource(data = dict(lambda_red = lambda_red, N = N, \
                                       b_d = b_d, C_f = C_f))
df = source7.to_df()
g = bebi103.viz.corner(df,parameters=params,
                        xtick_label_orientation=np.pi / 4,show_contours=True,
                      frame_width = 100,frame_height = 105)

# # plot samples run for each parameter
bin_indx = np.where(table_uniq['bin']== binid)
samples = table_uniq['samples'][bin_indx][0]

# Guess good model parameters
lamred = 5897.5581
logN = 14.5
bD = 20.0
Cf = 0.5

lam_line = Span(location=lamred,dimension='width', line_color='grey', line_width=2)
N_line = Span(location=logN,dimension='width', line_color='grey', line_width=2)
bd_line = Span(location=bD,dimension='width', line_color='grey', line_width=2)
Cf_line = Span(location=Cf,dimension='width', line_color='grey', line_width=2)
iterations = np.arange(samples.shape[1])

# create tools and tooltips for each plot
tools1 = "pan,wheel_zoom,box_zoom,reset"

tooltips5 = [("(x,y)", "($x{0}, $y{0})"), ("Gas Velocity (km/s)", "@image{0.00}")]
tooltips6 = [("Velocity", "@fk_obs_vel{000.0}"), ("Fake Flux", "@fk_obs_flux{0.00}"), ("Fit Flux", "@fit_flux{0.00}")]

# create figures for each plot
p5 = figure(title='Gas Velocity Map',tools=tools1, plot_width=535, plot_height=480,toolbar_location="below")
p6 = figure(title='DAP Na I Region w/ MCMC Best-Fit Model',tools=tools1,plot_width=600,plot_height=480,
            toolbar_location="left")
p6.legend.label_text_font_size = "10pt"
p6.x_range = Range1d(-1000, 1000)
p6.y_range = Range1d(0.55, 1.05)

p8_1 = figure(title="Samples logN prior4 range (12.0-16.5)", y_axis_label='lambda_red')
p8_2 = figure(y_axis_label='N' )
p8_3 = figure(y_axis_label='b_d')
p8_4 = figure(y_axis_label='C_f')

p5.x_range.range_padding = p5.y_range.range_padding = 0

# plot 5 MUSE gas velocity map plot
# low=0.2*np.min(gas_vel_map), high=0.2*np.max(gas_vel_map)
color_mapper5 = LinearColorMapper(palette=cc.coolwarm,low=-125,high=125)
color_bar5 = ColorBar(color_mapper=color_mapper5, label_standoff=12)
gas_velocity = p5.image(image=[np.transpose(gas_vel_map)], x=0, y=0, dw=gas_vel_map.shape[0],dh=gas_vel_map.shape[1],
         color_mapper=color_mapper5)
p5.add_layout(color_bar5,'right')

# add hover tool
p5.add_tools(HoverTool(tooltips = tooltips5, renderers = [gas_velocity]))

# create gas velocity image box highlight in plot 5
# box 1: move the box along with the Mouse event
# box 2: move the box along with the Tap event
gasvel_box1 = Square(x = 20 ,y = 20 ,size = 10, fill_alpha = 0.5)
gasvel_box2 = Square(x = 20 ,y = 20 ,size = 10, fill_alpha = 0.5)
p5.add_glyph(gasvel_box1)
p5.add_glyph(gasvel_box2)

source6 = ColumnDataSource(data = dict(sc_vel=sc_vel,sc_flux=sc_flux,fk_abs_vel=fk_abs_vel,fk_abs_flux=fk_abs_flux,
                                       fk_obs_vel=fk_obs_vel,fk_obs_flux=fk_obs_flux,fit_vel=fit_vel,fit_flux=fit_flux))

# plot 6 DAP Na I spectrum plot w/ best-fit absorption model
fk_abs = p6.line('fk_abs_vel', 'fk_abs_flux', source=source6,legend_label = 'Fake Absorber',color='steelblue')
sc_model = p6.line('sc_vel', 'sc_flux', source=source6,legend_label = 'S.C. Model',color='orange')
fk_obs = p6.step('fk_obs_vel', 'fk_obs_flux', source=source6,legend_label = 'Fake Obs. Flux',mode="center",color='green')
fit = p6.line('fit_vel', 'fit_flux', source=source6,legend_label='Absorption Fit',color='black')

# have hover tool include wavelength, flux and model values using vline mode
p6.add_tools(HoverTool(tooltips = tooltips6, renderers =[fk_obs] , mode='vline'))
p6.legend.click_policy="hide"
p6.yaxis.axis_label = "Flux"
# add Na I lines to the spectrum plot
# Na_D1 = Span(location=D1,
#                               dimension='height', line_color='grey',
#                               line_dash='dashed', line_width=2)
Na_D2 = Span(location=0,
                            dimension='height', line_color='grey',
                            line_dash='dashed', line_width=2)
# p6.add_layout(Na_D1)
p6.add_layout(Na_D2)
# Add an annotation which includes bin ID and spaxel coordinate
binid_text = 'Bin ID: ' + str(binid)
x,y = 20,20
binid_label = Label(x=5,y=46, x_units='screen', y_units='screen', text = binid_text)
spaxel_text_p6 = 'Spaxel: (' +str(x) +','+str(y)+')'
spaxel_label_p6 = Label(x=5,y=31, x_units='screen', y_units='screen', text = spaxel_text_p6)
# text labels
gas_vel_text,text_lamred,text_logN,text_bd,text_Cf = get_text_labels(table_uniq,binid)

inp_title = Label(x=350,y=180,x_units='screen', y_units='screen',text='Input Parameters')
inp_gasvel_txt = 'gas vel.: '+str(param_row['gas_velocity'][0])+' km/s'
inp_gasvel_label = Label(x=350,y=167,x_units='screen', y_units='screen',text=inp_gasvel_txt)
inp_logN_txt = 'logN: '+str(param_row['logN'][0])+' cm-2'
inp_logN_label = Label(x=350,y=148,x_units='screen', y_units='screen',text=inp_logN_txt)
inp_bD_txt = 'bD: '+str(param_row['b_D'][0])+' km/s'
inp_bD_label = Label(x=350,y=131,x_units='screen', y_units='screen',text=inp_bD_txt)
inp_Cf_txt = 'C_f: '+str(param_row['C_f'][0])
inp_Cf_label = Label(x=350,y=114,x_units='screen', y_units='screen',text=inp_Cf_txt)
inp_snr_txt = 'Spaxel SNR: '+str(param_row['spaxel_snr'][0])
inp_snr_label = Label(x=350,y=97,x_units='screen', y_units='screen',text=inp_snr_txt)
inp_sc_txt = 'S.C Bin ID: '+str(param_row['sc_bin_id'][0])
inp_sc_label = Label(x=350,y=80,x_units='screen', y_units='screen',text=inp_sc_txt)

outp_title = Label(x=300,y=46,x_units='screen', y_units='screen',text='Output Parameters')
gas_vel_label_p6 = Label(x=5,y=16, x_units='screen', y_units='screen', text = gas_vel_text)
lamred_label_p6 = Label(x=5,y=1, x_units='screen', y_units='screen', text = text_lamred)
# left side of the plot
logN_label_p6 = Label(x=300,y=31, x_units='screen', y_units='screen', text = text_logN)
bd_label_p6 = Label(x=300,y=16, x_units='screen', y_units='screen', text = text_bd)
Cf_label_p6 = Label(x=300,y=1, x_units='screen', y_units='screen', text = text_Cf)


p6.add_layout(inp_title)
p6.add_layout(inp_gasvel_label)
p6.add_layout(inp_logN_label)
p6.add_layout(inp_bD_label)
p6.add_layout(inp_Cf_label)
p6.add_layout(inp_snr_label)
p6.add_layout(inp_sc_label)

p6.add_layout(outp_title)
p6.add_layout(binid_label)
p6.add_layout(spaxel_label_p6)
p6.add_layout(gas_vel_label_p6)
p6.add_layout(lamred_label_p6)
p6.add_layout(logN_label_p6)
p6.add_layout(bd_label_p6)
p6.add_layout(Cf_label_p6)

# samples grid plot for each parameter
for i in range(samples.shape[0]):
    p8_1.line(iterations,samples[i,:,0],line_alpha=0.4,line_color='black')
    p8_1.add_layout(lam_line)
    p8_2.line(iterations,samples[i,:,1],line_alpha=0.4,line_color='black')
    p8_2.add_layout(N_line)
    p8_3.line(iterations,samples[i,:,2],line_alpha=0.4,line_color='black')
    p8_3.add_layout(bd_line)
    p8_4.line(iterations,samples[i,:,3],line_alpha=0.4,line_color='black')
    p8_4.add_layout(Cf_line)

grid8 = gridplot([[p8_1],[p8_2],[p8_3],[p8_4]],plot_width=535, plot_height=120)

# -----------------------------------------------------------------
def update_data(attr):
    # x and y values when tapping the mouse over a pixel
    x = round(attr.x)
    y = round(attr.y)
    
    # get data around Na I region and the best-fit model to it
    binid = abs_binid_map[int(x),int(y)]
    param_row = param_table[param_table['fake_bin_id'] == binid]
    sc_binid = binid_map == param_row['sc_bin_id'][0]
    
    # single bin velocity
    binvel = ppxf_v_map[sc_binid][0]
    # single flux, error and model spectrum corresponding to that bin
    flux_bin = np.ma.array(flux[:, sc_binid][:, 0])
    err_bin = np.ma.array(error[:, sc_binid][:, 0])
    mod_bin = np.ma.array(model[:, sc_binid][:, 0])

    # Determine bin redshift from stellar velocities
    bin_z = z + ((1 + z) * (binvel / c))
    restwave = wave / (1.0 + bin_z)
    

    sc_vel,sc_flux,fk_abs_vel,fk_abs_flux, fk_obs_vel, fk_obs_flux, fit_bin = get_fake_fluxes(param_row, restwave, flux_bin,
                                                                                               err_bin, mod_bin, blim, rlim, 
                                                                                                median_LSFAng,LSFvel,table_uniq)
    
    fit_vel = c*((fit_bin['modwv']/D2)-1)
    fit_flux = fit_bin['modflx']
    
    source6.data = dict(sc_vel=sc_vel,sc_flux=sc_flux.data,fk_abs_vel=fk_abs_vel,fk_abs_flux=fk_abs_flux.data.value,
                                       fk_obs_vel=fk_obs_vel.data,fk_obs_flux=fk_obs_flux.data,
                                       fit_vel=fit_vel,fit_flux=fit_flux)
    
    # update bin ID and spaxel annotation in plot 6
    binid_text = 'Bin ID: ' + str(binid) 
    spaxel_text = 'Spaxel: (' +str(int(x)) +','+str(int(y))+')'
    # text labels
    gas_vel_text, text_lamred, text_logN, text_bd, text_Cf = get_text_labels(table_uniq,binid)
    
    inp_gasvel_txt = 'gas vel.: '+str(param_row['gas_velocity'][0])+' km/s'
    inp_logN_txt = 'logN: '+str(param_row['logN'][0])+' cm-2'
    inp_bD_txt = 'bD: '+str(param_row['b_D'][0])+' km/s'
    inp_Cf_txt = 'C_f: '+str(param_row['C_f'][0])
    inp_snr_txt = 'Spaxel SNR: '+str(param_row['spaxel_snr'][0])
    inp_sc_txt = 'S.C Bin ID: '+str(param_row['sc_bin_id'][0])

    inp_gasvel_label.update(text = inp_gasvel_txt)
    inp_logN_label.update(text = inp_logN_txt)
    inp_bD_label.update(text = inp_bD_txt)
    inp_Cf_label.update(text = inp_Cf_txt)
    inp_snr_label.update(text = inp_snr_txt)
    inp_sc_label.update(text = inp_sc_txt)
                            
    binid_label.update(text = binid_text)
    spaxel_label_p6.update(text = spaxel_text)
    gas_vel_label_p6.update(text = gas_vel_text)
    lamred_label_p6.update(text = text_lamred)
    logN_label_p6.update(text = text_logN)
    bd_label_p6.update(text = text_bd)
    Cf_label_p6.update(text = text_Cf)

    # update corner plot with new spaxel bin ID
    lambda_red, N, b_d, C_f = corner(binid,table_uniq)
    source7.data = dict(lambda_red = lambda_red, N = N, b_d = b_d, C_f = C_f)
    df = source7.to_df()
    g = bebi103.viz.corner(df,parameters=params,
                        xtick_label_orientation=np.pi / 4,show_contours=True,
                      frame_width = 100,frame_height = 105)
    
    layout_row2.children[0] = g
    
    # update samples run plot
    samples = table_uniq['samples'][bin_indx][0]
    iterations = np.arange(samples.shape[1])
    
    p8_1_update = figure(title="Samples logN prior4 range (12.0-16.5)", y_axis_label='lambda_red')
    p8_2_update = figure(y_axis_label='N' )
    p8_3_update = figure(y_axis_label='b_d')
    p8_4_update = figure(y_axis_label='C_f')
    
    # samples iteration grid plot for each parameter
    for i in range(samples.shape[0]):
        p8_1_update.line(iterations,samples[i,:,0],line_alpha=0.4,line_color='black')
        p8_1_update.add_layout(lam_line)
        p8_2_update.line(iterations,samples[i,:,1],line_alpha=0.4,line_color='black')
        p8_2_update.add_layout(N_line)
        p8_3_update.line(iterations,samples[i,:,2],line_alpha=0.4,line_color='black')
        p8_3_update.add_layout(bd_line)
        p8_4_update.line(iterations,samples[i,:,3],line_alpha=0.4,line_color='black')
        p8_4_update.add_layout(Cf_line)
    
    grid8_update = gridplot([[p8_1_update],[p8_2_update],[p8_3_update],[p8_4_update]],plot_width=535, plot_height=120)
    layout_row2.children[1] = grid8_update
    
    
    # update gas velocity box highlight in plot 5
    gasvel_box1.update(x=x,y=y)
    
# -----------------------------------------------------------------
    
def update_box(attr):
    # x and y values when moving the mouse over a pixel
    x = attr.x
    y = attr.y
    
    if x > gas_vel_map.shape[0]:
        # update gas velocity image box highlight in plot 5
        gasvel_box2.update(x=gas_vel_map.shape[0] ,y=y)

        
    elif x < 0:
        gasvel_box2.update(x=0 ,y=y)

    elif y > gas_vel_map.shape[1]:
        gasvel_box2.update(x=x ,y=gas_vel_map.shape[1])

    elif y < 0:
        gasvel_box2.update(x=x ,y=0)

    else:
        gasvel_box2.update(x=x ,y=y)

# update each event on each plot    
p5.on_event(events.Tap,update_data)
p5.on_event(events.MouseMove,update_box)

layout_row1 = row(children=[p5,p6])
curdoc().add_root(layout_row1)

layout_row2 = row(children=[g,grid8])
curdoc().add_root(layout_row2)
#layout_row3 =row(children=[p2,p1,p3])
#curdoc().add_root(layout_row3)
# curdoc().add_root(row(children=[p5,p6]))
# curdoc().add_root(row(children=[p1,p2]))
# curdoc().add_root(row(children=[p3,p4]))