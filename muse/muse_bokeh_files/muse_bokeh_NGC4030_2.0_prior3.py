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

# test cube
test_cube_file = main_dir2+ 'NGC0000.fits'

# obtain stellar velocity, stellar dispersion and bin id data from the galaxy
stellar_vel =  NGC4030_map['STELLAR_VEL'].data
stellar_sigma = NGC4030_map['STELLAR_SIGMA'].data
binid_map = NGC4030_map['BINID'].data[0]

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
halpha_map = NGC4030_map['EMLINE_GFLUX'].data[emline['Ha-6564'],:,:]
# H-beta emission line flux from MUSE cube
hbeta_map = NGC4030_map['EMLINE_GFLUX'].data[emline['Hb-4862'],:,:]
# dust attenuation map
ebv_map = 0.935 * np.log( (halpha_map/hbeta_map) /2.86 )
# extinction in magnitudes using Calzettie et al. 2000 extinction curve
k_lambda = extinction.calzetti00(np.array([6564.614]),a_v = 1, r_v = 3.1, unit='aa')
# corrected h-alpha map 
#halpha_map_corrected = halpha_map / 10**(-0.4*k_lambda*ebv_map)
halpha_map_corrected = halpha_map / 10**(0.4*k_lambda*ebv_map)
# replace NaN values w/ finite values
ebv_map[np.logical_not(np.isfinite(ebv_map))] = -999
# use good values in original h-alpha map to replace NaN values
# in corrected h-alpha map due to nan values from ebv map
nan_indx = np.where(np.isfinite(halpha_map_corrected) == False)
halpha_map_corrected[nan_indx] = halpha_map[nan_indx]

# recessional velocity of NGC 4030 (SIMBAD value) (km/s)
v_recess = 1376 * (u.kilometer / u.second)
# Hubble constant (km/s/Mpc)
H_0 = 73 * (u.kilometer / (u.second * 1e6 * u.parsec) )
# NGC4030 distance
r = (v_recess / H_0).to(u.centimeter)
# h-alpha luminosity map (ergs/s)
L_halpha = 4*np.pi * r**2 * halpha_map_corrected * ( (1e-17 * u.erg) / (u.second * u.centimeter**2) )
# SFR map from h-alpha luminosity in units of M_sun / year
logSFR_map = np.log10(L_halpha.value) - 41.27
logSFR_map[[np.logical_not(np.isfinite(logSFR_map))]] = -999
halpha_map_corrected[np.logical_not(np.isfinite(halpha_map_corrected))] = -999

# obtain image data averaged around Na I region from MUSE cube
dc = cube.DataCube(inp=NGC4030_cube0_6_file)
n_band = transform.narrowband(dc, 5880, 5940, mode='median')
img_data = n_band.data

# obtain flux array from MUSE cube
flux = NGC4030_cube0_6['FLUX'].data
model = NGC4030_cube0_6['MODEL'].data
ivar = NGC4030_cube0_6['IVAR'].data
error = np.sqrt(1/ivar)
wave = NGC4030_cube0_6['WAVE'].data

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

# sorting function (after changes I will make (04/20/22) )
def sort_key(file_path):
    # individually splits the directory and filename from the path given
    dir_path, filename = os.path.split(file_path)
    # grab only run number from filename
    run = filename[-8:-4]
    # return only integer run number to sort
    num = int(''.join(i for i in run if i.isdigit()) )
    return num

# directory path
path = '/Users/erickaguirre/Desktop/NaI_MCMC_output/NGC4030_2.0_test_prior3/'
#path = '/Users/erickaguirre/Desktop/NaI_MCMC_output/modfits/NGC4030_2.0_test_modfit_prior3/'
# fits file path
fits_paths = sorted(glob.glob(path+'NGC4030*'), key = sort_key)

# combine all the results from each fits file 
# done in MUSE analysis for one galaxy and create an
# astropy table with all the values for each bin id
binid_all = []
samples_all = []
percentiles_all = []
gas_vel_all = []

for fits_file in fits_paths:
    file = fits.open(fits_file)
    data_arr = file[1].data
                      
    for i in range(len(data_arr['bin'])):
        binid_all.append(data_arr['bin'][i])
        samples_all.append(data_arr['samples'][i])
        percentiles_all.append(data_arr['percentiles'][i])
        gas_vel_all.append(data_arr['velocities'][i])

table_all = Table([binid_all, samples_all,percentiles_all, gas_vel_all], \
          names=('bin', 'samples','percentiles','velocities'))

# get rid of duplicate bins in the table
table_uniq = unique(table_all, keys=['bin'], keep='last')

# create a map that places the gas velocitiy values 
# to their respective bin ID pixel coordinate
gas_vel_map = np.zeros(binid_map.shape)

for binid in table_uniq['bin']:
    index = np.where(binid_map == binid)
    binid_indx = np.where(table_uniq['bin']== binid)
    gas_vel_map[index] = table_uniq['velocities'][binid_indx][0]
    
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# plot DAP flux against best-fit NaImcmc model

# Need LSF in km/s
# This gives LSF in Ang
# CAUTION: this does not convert air wavelengths
# to vacuum, or account for velocity offset of each bin
redshift = 0.00489 # NGC 4030 redshift's
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

def NaImcmc(binid, redshift, binid_map, stellar_vel, flux, model,error, obswave,LSFvel):

    ind = binid_map == binid
    # single bin velocity
    binvel = stellar_vel[ind][0]
    # single flux, error and model spectrum corresponding to that bin 
    flux_bin = np.ma.array(flux[:,ind][:,0])
    err_bin = np.ma.array(error[:,ind][:,0])
    mod_bin = np.ma.array(model[:,ind][:,0])

    # Determine bin redshift:
    bin_z = redshift + ((1 + redshift) * binvel / c)
    restwave = obswave / (1.0 + bin_z)

    ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, blim, rlim, FIT_FLG=0)
    ndata_mod = continuum_normalize_NaI.norm(restwave, mod_bin, err_bin, blim, rlim, FIT_FLG=0)
    
    # Cut out NaI
    select = np.where((ndata['nwave'] > fitlim[0]) & (ndata['nwave'] < fitlim[1]))
    select_mod = np.where((ndata_mod['nwave'] > fitlim[0]) & (ndata_mod['nwave'] < fitlim[1]))
    restwave_NaI = ndata['nwave'][select].astype('float64')
    flux_NaI = ndata['nflux'][select].astype('float64')
    err_NaI = ndata['nerr'][select].astype('float64')
    sres_NaI = LSFvel
    mod_NaI = ndata_mod['nflux'][select_mod].astype('float64')

    data = {'wave':restwave_NaI, 'flux':flux_NaI, 'model':mod_NaI, 'err':err_NaI, 'velres':sres_NaI}
    binid_indx = table_uniq['bin'] == binid
    model_fit_bin = model_NaI.model_NaI(table_uniq['percentiles'][binid_indx][0][:,0], data['velres'], data['wave'])
    
    return data, model_fit_bin 
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# corner plot for each spaxel
def corner(binid):
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

# get Na I absorption lines w/ respect to galaxy's redshift
z = 0.00489 # NGC 4030 redshift's
# z = 0.00460 # NGC 14042 redshift's

# create ColumnDataSource from MUSE cube
source2 = ColumnDataSource(data = dict(wave = wave, flux = flux[:,195,195], model = model[:,195,195]))

# get data around Na I region and the best-fit model to it
binid = binid_map[195,195]
data_NaI, model_fit_bin_NaI = NaImcmc(binid, redshift, binid_map, stellar_vel, flux, model,error, wave,LSFvel)
source6 = ColumnDataSource(data = dict(data_wave = data_NaI['wave'], data_flux = data_NaI['flux'],data_smod=data_NaI['model'],
                                       model_wave = model_fit_bin_NaI['modwv'], model_flux = model_fit_bin_NaI['modflx']))

# get corner plot given a binid
params = [r'lambda_red', 'N', 'b_d','C_f']
lambda_red, N, b_d, C_f = corner(binid)
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
logN = 13.5
bD = 20.0
Cf = 0.5

lam_line = Span(location=lamred,dimension='width', line_color='grey', line_width=2)
N_line = Span(location=logN,dimension='width', line_color='grey', line_width=2)
bd_line = Span(location=bD,dimension='width', line_color='grey', line_width=2)
Cf_line = Span(location=Cf,dimension='width', line_color='grey', line_width=2)
iterations = np.arange(samples.shape[1])

# create tools and tooltips for each plot
tools1 = "pan,wheel_zoom,box_zoom,reset"

tooltips1 = [("Spaxel (x,y)", "($x{0}, $y{0})"), ("Velocity (km/s)", "@image{0.00}")]
tooltips2 = [("Wavelength", "@wave{0000.00}"), ("Flux", "@flux{0.00000}"), ("Model", "@model{0.00000}")]
tooltips3 = [("(x,y)", "($x{0}, $y{0})"), ("Stellar Dispersion", "@image{0.000}")]
tooltips4 = [("(x,y)", "($x{0}, $y{0})"), ("Flux", "@image{0.000}")]
tooltips5 = [("(x,y)", "($x{0}, $y{0})"), ("Gas Velocity (km/s)", "@image{0.00}")]
tooltips6 = [("Wavelength", "@data_wave{0000.0}"), ("Flux", "@data_flux"), ("Model", "@model_flux")]
tooltips9 = [("(x,y)", "($x{0}, $y{0})"), ("Flux", "@image{0.000}")]
tooltips10 = [("(x,y)", "($x{0}, $y{0})"), ("SFR (M_sun/yr)", "@image{0.000}")]

# create figures for each plot
p1 = figure(title='Stellar Velocity Map',tools=tools1, plot_width=535, plot_height=480,toolbar_location="right")
p2 = figure(title='DAP Spectrum',tools=tools1,x_range=(5880,5950), plot_width=600, plot_height=480,
            toolbar_location="left")
p3 = figure(title='Stellar Dispersion Map',tools=tools1,plot_width=535, plot_height=480,toolbar_location="below")
p4 = figure(title='Dust Attenuation Map',tools=tools1, plot_width=535, plot_height=480,toolbar_location="below")
p5 = figure(title='Gas Velocity Map',tools=tools1, plot_width=535, plot_height=480,toolbar_location="below")
p6 = figure(title='DAP Na I Region w/ MCMC Best-Fit Model',tools=tools1,plot_width=600,plot_height=480,
            toolbar_location="left")

p8_1 = figure(title="Samples logN prior 3 range (12.0-20.0)", y_axis_label='lambda_red')
p8_2 = figure(y_axis_label='N' )
p8_3 = figure(y_axis_label='b_d')
p8_4 = figure(y_axis_label='C_f')

p9 = figure(title='Corrected H-alpha Emission Flux Image',tools=tools1,plot_width=535,plot_height=480,
            toolbar_location="below")
p10 = figure(title='SFR H-alpha Map',tools=tools1, plot_width=535, plot_height=480,toolbar_location="below")

p1.x_range.range_padding = p1.y_range.range_padding = 0
p3.x_range.range_padding = p3.y_range.range_padding = 0
p4.x_range.range_padding = p4.y_range.range_padding = 0
p5.x_range.range_padding = p5.y_range.range_padding = 0
p9.x_range.range_padding = p9.y_range.range_padding = 0
p10.x_range.range_padding = p10.y_range.range_padding = 0

# plot 1 MUSE stellar velocity map plot
color_mapper1 = LinearColorMapper(palette=cc.coolwarm,low=-175,high=175)
color_bar1 = ColorBar(color_mapper=color_mapper1, label_standoff=12)
stellar_velocity = p1.image(image=[np.transpose(stellar_vel)], x=0, y=0, dw=img_data.shape[0],dh=img_data.shape[1], \
         color_mapper=color_mapper1)
p1.add_layout(color_bar1,'right')

# add hover tool
p1.add_tools(HoverTool(tooltips = tooltips1, renderers = [stellar_velocity]))

# create stellar velocity image box highlight in plot 1
# box 1: move the box along with the Mouse event
# box 2: move the box along with the Tap event
stellvel_box1 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
stellvel_box2 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
p1.add_glyph(stellvel_box1)
p1.add_glyph(stellvel_box2)

# plot 2 MUSE spaxel spectrum plot w/ best-fit model
flux_spec = p2.step('wave', 'flux', source=source2,legend_label = 'Data') #,mode="center")
model_spec = p2.line('wave', 'model', source=source2,color='orange',legend_label='Model')

# have hover tool include wavelength, flux and model values using vline mode
p2.add_tools(HoverTool(tooltips = tooltips2, renderers =[model_spec] , mode='vline'))

# format y-range for better visualization using a crude estimation of
# the spectrum's continuum 
continuum = np.mean(source2.data['flux'])
p2.xaxis.axis_label = 'Wavelength (angstroms)'#r"$$\lambda  (\AA)$$"
p2.yaxis.axis_label = "Flux"
p2.y_range = Range1d(start = 0.6 * continuum , end = 1.4 * continuum)
# hide either data or model spectrum whene clicked on
p2.legend.click_policy="hide"
# Add an annotation which includes spaxel coordinate
x,y = 195,195 
spaxel_text_p2 = 'Spaxel: (' +str(x) +','+str(y)+')'
spaxel_label_p2 = Label(x=25,y=28, x_units='screen', y_units='screen', text = spaxel_text_p2)
p2.add_layout(spaxel_label_p2)

# plot 6 DAP Na I spectrum plot w/ best-fit absorption model
NaI_spec = p6.step('data_wave', 'data_flux', source=source6,legend_label = 'DAP Data',mode="center")
NaI_smodel = p6.line('data_wave', 'data_smod', source=source6,legend_label = 'DAP S. Model',color='orange')
NaI_model_spec = p6.line('model_wave', 'model_flux', source=source6,color='green',legend_label='Abs. Model')
# have hover tool include wavelength, flux and model values using vline mode
p6.add_tools(HoverTool(tooltips = tooltips6, renderers =[NaI_model_spec] , mode='vline'))
p6.legend.click_policy="hide"
p6.yaxis.axis_label = "Flux"
# add Na I lines to the spectrum plot
Na_D1 = Span(location=D1,
                              dimension='height', line_color='grey',
                              line_dash='dashed', line_width=2)
Na_D2 = Span(location=D2,
                            dimension='height', line_color='grey',
                            line_dash='dashed', line_width=2)
p6.add_layout(Na_D1)
p6.add_layout(Na_D2)
# Add an annotation which includes bin ID and spaxel coordinate
binid_text = 'Bin ID: ' + str(binid)
x,y = 195,195
binid_label = Label(x=21,y=95, x_units='screen', y_units='screen', text = binid_text)
spaxel_text_p6 = 'Spaxel: (' +str(x) +','+str(y)+')'
spaxel_label_p6 = Label(x=21,y=80, x_units='screen', y_units='screen', text = spaxel_text_p6)
gas_vel = table_uniq['velocities'][bin_indx][0]
gas_vel_text = 'Na I Vel. (km/s): {:.2f}'.format(gas_vel)
gas_vel_label_p6 = Label(x=21,y=65, x_units='screen', y_units='screen', text = gas_vel_text)

percentiles = table_uniq['percentiles'][bin_indx][0][0]
percent_50 = percentiles[0]
percent_84 = percentiles[1]
percent_16 = percentiles[2]
percentile_text1 = 'Lambda ($$\AA$$):'
percentile_text2 = '50th %: {:.3f}'.format(percent_50)
percentile_text3 = '84th %: +{:.3f}'.format(percent_84)
percentile_text4 = '16th %: -{:.3f}'.format(percent_16)

percentile_label1_p6 = Label(x=21,y=50, x_units='screen', y_units='screen', text = percentile_text1)
percentile_label2_p6 = Label(x=21,y=35, x_units='screen', y_units='screen', text = percentile_text2)
percentile_label3_p6 = Label(x=21,y=20, x_units='screen', y_units='screen', text = percentile_text3)
percentile_label4_p6 = Label(x=21,y=5, x_units='screen', y_units='screen', text = percentile_text4)
p6.add_layout(binid_label)
p6.add_layout(spaxel_label_p6)
p6.add_layout(gas_vel_label_p6)
p6.add_layout(percentile_label1_p6)
p6.add_layout(percentile_label2_p6)
p6.add_layout(percentile_label3_p6)
p6.add_layout(percentile_label4_p6)

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

# plot 3 MUSE stellar dispersion plot
# must give a vector of image data for image parameter
# Plasma265 (purple to orange)
# Blues8 (blue to white)
# commented color_mapper3 below is for 0.6 square binning
# color_mapper3 = LinearColorMapper(palette='Viridis256',low=np.percentile(stellar_sigma,20),high=0.12*np.max(stellar_sigma))
color_mapper3 = LinearColorMapper(palette='Viridis256',low=55,high=115)
color_bar3 = ColorBar(color_mapper=color_mapper3, label_standoff=12)                                  
stellar_dispersion = p3.image(image=[np.transpose(stellar_sigma)], x=0, y=0, dw=stellar_sigma.shape[0],dh=stellar_sigma.shape[1],
                              color_mapper=color_mapper3)
p3.add_layout(color_bar3,'right')

# add hover tool
p3.add_tools(HoverTool(tooltips = tooltips3, renderers = [stellar_dispersion]))

# create stellar dispersion plot image box highlight 
# box 1: move the box along with the Mouse event
# box 2: move the box along with the Tap event
stell_disp_box1 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
stell_disp_box2 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
p3.add_glyph(stell_disp_box1)
p3.add_glyph(stell_disp_box2)

# plot 4 MUSE cube NaI narrow band image
# must give a vector of image data for image parameter
# Plasma265 (purple to orange)
# Blues8 (blue to white)
color_mapper4 = LogColorMapper(palette='Inferno256',low=np.percentile(img_data,10),high=np.max(img_data))
color_bar4 = ColorBar(color_mapper=color_mapper4, label_standoff=12)   
narrow_band = p4.image(image=[np.transpose(img_data)], x=0, y=0, dw=img_data.shape[0],dh=img_data.shape[1], 
        color_mapper=color_mapper4)
p4.add_layout(color_bar4,'right')

# add hover tool 
p4.add_tools(HoverTool(tooltips = tooltips4, renderers = [narrow_band]))

# create narrow-band image box highlight 
# box 1: move the box along with the Mouse event
# box 2: move the box along with the Tap event
img_box1 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
img_box2 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
p4.add_glyph(img_box1)
p4.add_glyph(img_box2)

# create narrow-band image box highlight 
# box 1: move the box along with the Mouse event
# box 2: move the box along with the Tap event
img_box1 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
img_box2 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
p4.add_glyph(img_box1)
p4.add_glyph(img_box2)

# plot 5 MUSE gas velocity map plot
# low=0.2*np.min(gas_vel_map), high=0.2*np.max(gas_vel_map)
color_mapper5 = LinearColorMapper(palette=cc.coolwarm,low=-100,high=100)
color_bar5 = ColorBar(color_mapper=color_mapper5, label_standoff=12)
gas_velocity = p5.image(image=[np.transpose(gas_vel_map)], x=0, y=0, dw=img_data.shape[0],dh=img_data.shape[1],
         color_mapper=color_mapper5)
p5.add_layout(color_bar5,'right')

# add hover tool
p5.add_tools(HoverTool(tooltips = tooltips5, renderers = [gas_velocity]))

# create gas velocity image box highlight in plot 5
# box 1: move the box along with the Mouse event
# box 2: move the box along with the Tap event
gasvel_box1 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
gasvel_box2 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
p5.add_glyph(gasvel_box1)
p5.add_glyph(gasvel_box2)

# plot 10 MUSE SFR H-alpha map plot
# low=0.2*np.min(gas_vel_map), high=0.2*np.max(gas_vel_map)
color_mapper10 = LinearColorMapper(palette='Inferno256',low=-6,high=-4.5)
color_bar10 = ColorBar(color_mapper=color_mapper10, label_standoff=12)
#color_bar10 = ColorBar(color_mapper=color_mapper10, label_standoff=12,title='Solar Mass/year',title_text_align = 'center')
SFR_halpha = p10.image(image=[np.transpose(logSFR_map)], x=0, y=0, dw=logSFR_map.shape[0],dh=logSFR_map.shape[1],
         color_mapper=color_mapper10)
p10.add_layout(color_bar10,'right')

# add hover tool
p10.add_tools(HoverTool(tooltips = tooltips10, renderers = [SFR_halpha]))

# create SFR image box highlight in plot 10
# box 1: move the box along with the Mouse event
# box 2: move the box along with the Tap event
SFR_box1 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
SFR_box2 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
p10.add_glyph(SFR_box1)
p10.add_glyph(SFR_box2)

# plot 9 MUSE cube H-alpha emission line flux image
# must give a vector of image data for image parameter
# Plasma265 (purple to orange)
# Blues8 (blue to white)
color_mapper9 = LogColorMapper(palette='Inferno256',low=0.1,high=1.5*np.max(halpha_map_corrected))
# color_bar9 = ColorBar(color_mapper=color_mapper9, label_standoff=12,title='Flux (1E-17 erg/s/cm^2/spaxel)',
#                       title_text_align = 'center')   
color_bar9 = ColorBar(color_mapper=color_mapper9, label_standoff=12)
halpha_image = p9.image(image=[np.transpose(halpha_map_corrected)], x=0, y=0,
                        dw=halpha_map_corrected.shape[0],dh=halpha_map_corrected.shape[1], 
                        color_mapper=color_mapper9)
p9.add_layout(color_bar9,'right')

# add hover tool 
p9.add_tools(HoverTool(tooltips = tooltips9, renderers = [halpha_image]))

# create H-alpha image box highlight in plot 9
halpha_box1 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
halpha_box2 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
p9.add_glyph(halpha_box1)

# -----------------------------------------------------------------
def update_data(attr):
    # x and y values when tapping the mouse over a pixel
    x = round(attr.x)
    y = round(attr.y)
    # update the source data to show the spectrum corresponding to that pixel
    source2.data = dict(wave = wave, flux = flux[:,int(x),int(y)], model = model[:,int(x),int(y)])
    
    # get data around Na I region and the best-fit model to it
    binid = binid_map[int(x),int(y)]
    bin_indx = np.where(table_uniq['bin']== binid)
    data_NaI, model_fit_bin_NaI = NaImcmc(binid, redshift, binid_map, stellar_vel, flux, model,error, wave,LSFvel)
    
    source6.data = dict(data_wave = data_NaI['wave'], data_flux = data_NaI['flux'], data_smod = data_NaI['model'], \
                                model_wave = model_fit_bin_NaI['modwv'], model_flux = model_fit_bin_NaI['modflx'])
    
    # update bin ID and spaxel annotation in plot 6
    binid_text = 'Bin ID: ' + str(binid) 
    spaxel_text = 'Spaxel: (' +str(int(x)) +','+str(int(y))+')'
    gas_vel = table_uniq['velocities'][bin_indx][0]
    gas_vel_text = 'Gas V. (km/s): {:.2f}'.format(gas_vel)
  
    percentiles = table_uniq['percentiles'][bin_indx][0][0]
    percent_50 = percentiles[0]
    percent_84 = percentiles[1]
    percent_16 = percentiles[2]
    percentile_text1 = 'Lambda (A):'
    percentile_text2 = '50th %: {:.3f}'.format(percent_50)
    percentile_text3 = '84th %: +{:.3f}'.format(percent_84)
    percentile_text4 = '16th %: -{:.3f}'.format(percent_16)
    
    binid_label.update(text = binid_text)
    spaxel_label_p2.update(text = spaxel_text)
    spaxel_label_p6.update(text = spaxel_text)
    gas_vel_label_p6.update(text = gas_vel_text)
    percentile_label1_p6.update(text = percentile_text1)
    percentile_label2_p6.update(text = percentile_text2)
    percentile_label3_p6.update(text = percentile_text3)
    percentile_label4_p6.update(text = percentile_text4)
    
    # update corner plot with new spaxel bin ID
    lambda_red, N, b_d, C_f = corner(binid)
    source7.data = dict(lambda_red = lambda_red, N = N, b_d = b_d, C_f = C_f)
    df = source7.to_df()
    g = bebi103.viz.corner(df,parameters=params,
                        xtick_label_orientation=np.pi / 4,show_contours=True,
                      frame_width = 100,frame_height = 105)
    
    layout_row2.children[1] = g
    
    # update samples run plot
    samples = table_uniq['samples'][bin_indx][0]
    iterations = np.arange(samples.shape[1])
    
    p8_1_update = figure(title="Samples logN prior3 range (12.0-20.0)", y_axis_label='lambda_red')
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
    layout_row2.children[2] = grid8_update
    
    # re-format y-range from new MUSE spectrum
    continuum = np.mean(source2.data['flux'])    
    p2.y_range.update(start = 0.6*continuum , end = 1.4*continuum)
    
    # update stellar velocity box highlight in plot 1
    stellvel_box1.update(x=x,y=y)
    
    # update stellar dispersion box highlight in plot 3
    stell_disp_box1.update(x=x,y=y)
    
    # update narrow-band box highlight in plot 4
    img_box1.update(x=x,y=y)
    
    # update gas velocity box highlight in plot 5
    gasvel_box1.update(x=x,y=y)
    
    # update halpha box highlight in plot 9
    halpha_box1.update(x=x,y=y)
    
    # update SFR box highlight in plot 10
    SFR_box1.update(x=x,y=y)
    
# -----------------------------------------------------------------
    
def update_box(attr):
    # x and y values when moving the mouse over a pixel
    x = attr.x
    y = attr.y
    
    if x > stellar_vel.shape[0]:
        # update stellar velocity image box highlight in plot 1
        stellvel_box2.update(x=stellar_vel.shape[0] ,y=y)
        # update stellar dispersion box highlight in plot 3
        stell_disp_box2.update(x=stellar_vel.shape[0],y=y)
        # update narrow-band image box highlight in plot 4
        img_box2.update(x=img_data.shape[0],y=y)
        # update gas velocity image box highlight in plot 5
        gasvel_box2.update(x=stellar_vel.shape[0] ,y=y)
        # update gas velocity image box highlight in plot 9
        halpha_box2.update(x=stellar_vel.shape[0] ,y=y)
        # update gas velocity image box highlight in plot 10
        SFR_box2.update(x=stellar_vel.shape[0] ,y=y)
        
    elif x < 0:
        stellvel_box2.update(x=0 ,y=y)
        stell_disp_box2.update(x=0,y=y)
        img_box2.update(x=0,y=y)
        gasvel_box2.update(x=0 ,y=y)
        halpha_box2.update(x=0 ,y=y)
        SFR_box2.update(x=0 ,y=y)
        
    elif y > stellar_vel.shape[1]:
        stellvel_box2.update(x=x ,y=stellar_vel.shape[1])
        stell_disp_box2.update(x=x,y=stellar_vel.shape[1])
        img_box2.update(x=x,y=img_data.shape[1])
        gasvel_box2.update(x=x ,y=stellar_vel.shape[1])
        halpha_box2.update(x=x ,y=stellar_vel.shape[1])
        SFR_box2.update(x=x ,y=stellar_vel.shape[1])
        
    elif y < 0:
        stellvel_box2.update(x=x ,y=0)
        stell_disp_box2.update(x=x,y=0)
        img_box2.update(x=x,y=0)
        gasvel_box2.update(x=x ,y=0)
        halpha_box2.update(x=x ,y=0)
        SFR_box2.update(x=x ,y=0)
    
    else:
        stellvel_box2.update(x=x ,y=y)
        stell_disp_box2.update(x=x,y=y)
        img_box2.update(x=x,y=y)
        gasvel_box2.update(x=x ,y=y)
        halpha_box2.update(x=x ,y=y)
        SFR_box2.update(x=x ,y=y)
    

# update each event on each plot    
p1.on_event(events.Tap,update_data)
p1.on_event(events.MouseMove,update_box)

p3.on_event(events.Tap,update_data)
p3.on_event(events.MouseMove,update_box)

p4.on_event(events.Tap,update_data)
p4.on_event(events.MouseMove,update_box)

p5.on_event(events.Tap,update_data)
p5.on_event(events.MouseMove,update_box)

p9.on_event(events.Tap,update_data)
p9.on_event(events.MouseMove,update_box)

p10.on_event(events.Tap,update_data)
p10.on_event(events.MouseMove,update_box)

layout_row1 = row(children=[p5,p1,p3])
curdoc().add_root(layout_row1)
layout_row2 = row(children=[p6,g,grid8])
curdoc().add_root(layout_row2)
# layout_row3 =row(children=[p2]) #,p1,p3])
# curdoc().add_root(layout_row3)
# curdoc().add_root(row(children=[p5,p6]))
# curdoc().add_root(row(children=[p1,p2]))
# curdoc().add_root(row(children=[p3,p4]))