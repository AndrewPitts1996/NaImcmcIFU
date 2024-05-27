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
from bokeh.io import export_png

from astropy.table import Table
from astropy import units as u
import time
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
from bokeh.io import curdoc
from bokeh import events
from bokeh.models import Span,HoverTool,Range1d,Scatter

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
from IPython import embed

import extinction
from bokeh.plotting import figure, output_file, save
from bokeh.palettes import Greys256,Turbo256
import bebi103
import emcee

# MUSE cube directory
main_dir = '/Users/erickaguirre/mangadap/examples/'
main_dir2 = '/Users/erickaguirre/Desktop/SDSU_Research/Getting_used_to_MaNGA_DAP/'
main_dir3 = '/Users/erickaguirre/Desktop/DAP_outputs/'

# NGC 4030 
NGC4030_output_dir = 'output2.0_NGC4030_NOISM/'
NGC4030_cube_dir = 'SQUARE2.0-MILESHC-MASTARHC2-NOISM/1/1/'
# log cube
NGC4030_cube0_6_file = main_dir3+ NGC4030_output_dir+ NGC4030_cube_dir+\
                       'manga-1-1-LOGCUBE-SQUARE2.0-MILESHC-MASTARHC2-NOISM.fits'
NGC4030_cube0_6 = fits.open(NGC4030_cube0_6_file)

# log maps
NGC4030_map_file = main_dir3+ NGC4030_output_dir+ NGC4030_cube_dir+\
                   'manga-1-1-MAPS-SQUARE2.0-MILESHC-MASTARHC2-NOISM.fits'
NGC4030_map = fits.open(NGC4030_map_file)

# obtain stellar velocity, stellar dispersion and bin id data from the galaxy
stellar_vel =  NGC4030_map['STELLAR_VEL'].data
stellar_err = np.sqrt(1/NGC4030_map['STELLAR_VEL_IVAR'].data)
binid_map = NGC4030_map['BINID'].data[0]

# # emission line header dictionary
# emline = {}
# for k, v in NGC4030_map['EMLINE_GFLUX'].header.items():
#     if k[0] == 'C':
#         try:
#             i = int(k[1:])-1
#         except ValueError:
#             continue
#         emline[v] = i
#
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
# -----------------------------------------------------------------
def NaImcmc(binid, redshift, binid_map, stellar_vel, flux, model, error, obswave, LSFvel):
    ind = binid_map == binid
    # single bin velocity
    binvel = stellar_vel[ind][0]
    # single flux, error and model spectrum corresponding to that bin
    flux_bin = np.ma.array(flux[:, ind][:, 0])
    err_bin = np.ma.array(error[:, ind][:, 0])
    mod_bin = np.ma.array(model[:, ind][:, 0])

    # Determine bin redshift:
    bin_z = redshift + ((1 + redshift) * binvel / c)
    restwave = obswave / (1.0 + bin_z)

    # observed flux (gas+continuum model)
    tot_ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, blim, rlim, FIT_FLG=0)
    # continuum model
    mod_ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, blim, rlim,FIT_FLG=1,smod=mod_bin)
    # gas flux (total flux / continuum)
    gas_ndata = continuum_normalize_NaI.smod_norm(restwave,tot_ndata['nflux'],tot_ndata['nerr'],
                                                  mod_ndata['nflux'],blim,rlim)
    # Cut out NaI
    select_obs = np.where((tot_ndata['nwave'] > fitlim[0]) & (tot_ndata['nwave'] < fitlim[1]))
    select_mod = np.where((mod_ndata['nwave'] > fitlim[0]) & (mod_ndata['nwave'] < fitlim[1]))
    select_gas = np.where((gas_ndata['nwave'] > fitlim[0]) & (gas_ndata['nwave'] < fitlim[1]))

    restwave_NaI = tot_ndata['nwave'][select_obs].astype('float64')
    obsflux_NaI = tot_ndata['nflux'][select_obs].astype('float64')
    err_NaI = tot_ndata['nerr'][select_obs].astype('float64')
    sres_NaI = LSFvel
    mod_NaI = mod_ndata['nflux'][select_mod].astype('float64')
    gas_NaI = gas_ndata['nflux'][select_gas].astype('float64')

    data = {'wave': restwave_NaI, 'tot_flux': obsflux_NaI, 'model': mod_NaI, 'gas_flux': gas_NaI,
            'err': err_NaI, 'velres': sres_NaI}

    # binid_indx = table_uniq['bin'] == binid
    # model_fit_bin = model_NaI.model_NaI(table_uniq['percentiles'][binid_indx][0][:, 0], data['velres'], data['wave'])

    return data #, model_fit_bin
# -----------------------------------------------------------------

# create NaI D2 equivalent width map from the entire cube
# -----------------------------------------------------------------
def get_ew(binid_map, redshift, stellar_vel, flux, model, error, obswave):
    # select wavelength range
    wv_lim = [5800, 6000]
    flux_filt = flux[(obswave > wv_lim[0]) & (obswave < wv_lim[1]), :]
    err_filt = error[(obswave > wv_lim[0]) & (obswave < wv_lim[1]), :]
    mod_filt = model[(obswave > wv_lim[0]) & (obswave < wv_lim[1]), :]
    wave_filt = obswave[(obswave > wv_lim[0]) & (obswave < wv_lim[1])]

    ew_map = np.zeros(binid_map.shape)

    for i in range(np.max(binid_map)):
        ind = binid_map == i
        # single bin velocity
        binvel = stellar_vel[ind][0]
        # single flux, error and model spectrum corresponding to that bin
        flux_bin = np.ma.array(flux_filt[:, ind][:, 0])
        err_bin = np.ma.array(err_filt[:, ind][:, 0])
        mod_bin = np.ma.array(mod_filt[:, ind][:, 0])

        # Determine bin redshift:
        bin_z = redshift + ((1 + redshift) * binvel / c)
        restwave = wave_filt / (1.0 + bin_z)

        # Focus on D2 component(5891 angstrom)
        fitlim = [5887, 5894]
        D2_flux = flux_bin[(restwave > fitlim[0]) & (restwave < fitlim[1])]
        D2_model = mod_bin[(restwave > fitlim[0]) & (restwave < fitlim[1])]

        delta_lam = restwave[1] - restwave[0]
        ew = np.sum((1 - (D2_flux / D2_model)) * delta_lam)
        ew_map[ind] = ew

    return ew_map
# -----------------------------------------------------------------

# get Na I absorption lines w/ respect to galaxy's redshift
z = 0.00489 # NGC 4030 redshift's
# z = 0.00489 # NGC 14042 redshift's
print('so far so good!')
# create ColumnDataSource from MUSE cube
#source2 = ColumnDataSource(data = dict(wave = wave, flux = flux[:,195,195], model = model[:,195,195]))

# get data around Na I region and the best-fit model to it
binid = binid_map[195,195]
data_NaI = NaImcmc(binid, redshift, binid_map, stellar_vel, flux, model,error, wave,LSFvel)
source6 = ColumnDataSource(data = dict(wave=data_NaI['wave'], tot_flux=data_NaI['tot_flux'],err_flux=data_NaI['err'],
                                       model_flux=data_NaI['model'],gas_flux = data_NaI['gas_flux']))

# get Na I D2 equivalent width map
ew_D2_map = get_ew(binid_map, redshift, stellar_vel, flux, model, error, wave)

# create tools and tooltips for each plot
tools1 = "pan,wheel_zoom,box_zoom,reset"

tooltips1 = [("Spaxel (x,y)", "($x{0}, $y{0})"), ("Velocity (km/s)", "@image{0.00}")]
tooltips2 = [("Wavelength", "@wave{0000.00}"), ("Flux", "@flux{0.00000}"), ("Model", "@model{0.00000}")]
tooltips3 = [("(x,y)", "($x{0}, $y{0})"), ("Stellar Dispersion", "@image{0.000}")]
tooltips4 = [("(x,y)", "($x{0}, $y{0})"), ("Flux", "@image{0.000}")]
tooltips5 = [("(x,y)", "($x{0}, $y{0})"), ("EW", "@image{0.00}")]
tooltips6 = [("Wavelength", "@data_wave{0000.0}"),("Flux", "@tot_flux"),("Model","@model_flux"),("Gas", "@gas_flux")]
tooltips9 = [("(x,y)", "($x{0}, $y{0})"), ("Flux", "@image{0.000}")]
tooltips10 = [("(x,y)", "($x{0}, $y{0})"), ("SFR (M_sun/yr)", "@image{0.000}")]

# create figures for each plot
p1 = figure(title='Stellar Velocity Map',tools=tools1, width=535, height=480,toolbar_location="right")
# p2 = figure(title='DAP Spectrum',tools=tools1,x_range=(5880,5950), plot_width=600, plot_height=480,
#             toolbar_location="left")
p5 = figure(title='Na I EW Map',tools=tools1, width=535, height=480,toolbar_location="below")
p6 = figure(title='DAP Na I Region w/ Model',tools=tools1,width=600,height=480,
             toolbar_location="left")

p1.x_range.range_padding = p1.y_range.range_padding = 0
p5.x_range.range_padding = p5.y_range.range_padding = 0


# plot 1 MUSE stellar velocity map plot
color_mapper1 = LinearColorMapper(palette=cc.coolwarm,low=-175,high=175)
color_bar1 = ColorBar(color_mapper=color_mapper1, label_standoff=12)
stellar_velocity = p1.image(image=[np.transpose(stellar_vel)], x=0, y=0, dw=stellar_vel.shape[0],dh=stellar_vel.shape[1], \
         color_mapper=color_mapper1)
p1.add_layout(color_bar1,'right')
# add hover tool
p1.add_tools(HoverTool(tooltips = tooltips1, renderers = [stellar_velocity]))

# create stellar velocity image box highlight in plot 1
# box 1: move the box along with the Mouse event
# box 2: move the box along with the Tap event
stellvel_box1 = Scatter(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5,marker="square")
stellvel_box2 = Scatter(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5,marker="square")
p1.add_glyph(stellvel_box1)
p1.add_glyph(stellvel_box2)

# plot 6 DAP Na I spectrum plot w/ best-fit absorption model
NaI_spec = p6.step('wave', 'tot_flux', source=source6,legend_label = 'Total Flux',mode="center")
NaI_smodel = p6.line('wave', 'model_flux', source=source6,legend_label = 'DAP Model',color='orange')
NaI_gas = p6.line('wave', 'gas_flux', source=source6,color='green',legend_label='Gas Flux')
# have hover tool include wavelength, flux and model values using vline mode
p6.add_tools(HoverTool(tooltips = tooltips6, renderers =[NaI_spec,NaI_smodel,NaI_gas] , mode='vline'))
p6.legend.click_policy="hide"
p6.yaxis.axis_label = "Normalized Flux"
p6.xaxis.axis_label = "Wavelength (angstroms)"

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
binid_label = Label(x=5,y=46, x_units='screen', y_units='screen', text = binid_text)
spaxel_text_p6 = 'Spaxel: (' +str(x) +','+str(y)+')'
spaxel_label_p6 = Label(x=5,y=31, x_units='screen', y_units='screen', text = spaxel_text_p6)

p6.add_layout(binid_label)
p6.add_layout(spaxel_label_p6)
# text labels
# gas_vel_text,text_lamred,text_logN,text_bd,text_Cf = get_text_labels(table_uniq,bin_indx)
#
# gas_vel_label_p6 = Label(x=5,y=16, x_units='screen', y_units='screen', text = gas_vel_text)
# lamred_label_p6 = Label(x=5,y=1, x_units='screen', y_units='screen', text = text_lamred)
# # left side of the plot
# logN_label_p6 = Label(x=300,y=31, x_units='screen', y_units='screen', text = text_logN)
# bd_label_p6 = Label(x=300,y=16, x_units='screen', y_units='screen', text = text_bd)
# Cf_label_p6 = Label(x=300,y=1, x_units='screen', y_units='screen', text = text_Cf)

# plot 5 MUSE gas velocity map plot
# low=0.2*np.min(gas_vel_map), high=0.2*np.max(gas_vel_map)
# color_mapper5 = LinearColorMapper(palette= list(reversed(Greys256)),low=0,
#                                   high=0.5)
color_mapper5 = LinearColorMapper(palette=list(reversed(Turbo256)),low=-0.6,high=0.6)
color_bar5 = ColorBar(color_mapper=color_mapper5, label_standoff=12,title='EW (angstroms)',
                      major_label_text_align = 'right',
                      title_standoff = 4,title_text_align='right',title_text_font_size = '15px')

equiv_width = p5.image(image=[np.transpose(ew_D2_map)], x=0, y=0, dw=ew_D2_map.shape[0],dh=ew_D2_map.shape[1],
         color_mapper=color_mapper5)
p5.add_layout(color_bar5,'right')

# add hover tool
p5.add_tools(HoverTool(tooltips = tooltips5, renderers = [equiv_width]))

# create gas velocity image box highlight in plot 5
# box 1: move the box along with the Mouse event
# box 2: move the box along with the Tap event
gasvel_box1 = Scatter(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5,marker="square")
gasvel_box2 = Scatter(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5,marker="square")
p5.add_glyph(gasvel_box1)
p5.add_glyph(gasvel_box2)

#export_png(p5, filename="gasvel_map.png")
# -----------------------------------------------------------------
def update_data(attr):
    # x and y values when tapping the mouse over a pixel
    x = round(attr.x)
    y = round(attr.y)
    # update the source data to show the spectrum corresponding to that pixel
    #source2.data = dict(wave = wave, flux = flux[:,int(x),int(y)], model = model[:,int(x),int(y)])

    # get data around Na I region and the best-fit model to it
    binid = binid_map[int(x),int(y)]
    #bin_indx = np.where(table_uniq['bin']== binid)
    data_NaI = NaImcmc(binid, redshift, binid_map, stellar_vel, flux, model, error, wave, LSFvel)

    source6.data = dict(wave=data_NaI['wave'], tot_flux=data_NaI['tot_flux'], err_flux=data_NaI['err'],
                        model_flux=data_NaI['model'], gas_flux=data_NaI['gas_flux'])

    # update bin ID and spaxel annotation in plot 6
    binid_text = 'Bin ID: ' + str(binid)
    spaxel_text = 'Spaxel: (' +str(int(x)) +','+str(int(y))+')'
    # text labels
    #gas_vel_text, text_lamred, text_logN, text_bd, text_Cf = get_text_labels(table_uniq,bin_indx)

    binid_label.update(text = binid_text)
#     spaxel_label_p2.update(text=spaxel_text)
    spaxel_label_p6.update(text = spaxel_text)
#     gas_vel_label_p6.update(text = gas_vel_text)
#     lamred_label_p6.update(text = text_lamred)
#     logN_label_p6.update(text = text_logN)
#     bd_label_p6.update(text = text_bd)
#     Cf_label_p6.update(text = text_Cf)

#     # update corner plot with new spaxel bin ID
#     lambda_red, N, b_d, C_f = corner(binid)
#     source7.data = dict(lambda_red = lambda_red, N = N, b_d = b_d, C_f = C_f)
#     df = source7.to_df()
#     g = bebi103.viz.corner(df,parameters=params,
#                         xtick_label_orientation=np.pi / 4,show_contours=True,
#                       frame_width = 100,frame_height = 105)

#     layout_row2.children[1] = g


    # re-format y-range from new MUSE spectrum
    #continuum = np.mean(source2.data['flux'])
    #p2.y_range.update(start = 0.6*continuum , end = 1.4*continuum)

    # update stellar velocity box highlight in plot 1
    stellvel_box1.update(x=x,y=y)

    # update stellar dispersion box highlight in plot 3
    #stell_disp_box1.update(x=x,y=y)

    # update narrow-band box highlight in plot 4
    #img_box1.update(x=x,y=y)

    # update gas velocity box highlight in plot 5
    gasvel_box1.update(x=x,y=y)

    # update halpha box highlight in plot 9
    #halpha_box1.update(x=x,y=y)

    # update SFR box highlight in plot 10
    #SFR_box1.update(x=x,y=y)

# -----------------------------------------------------------------

def update_box(attr):
    # x and y values when moving the mouse over a pixel
    x = attr.x
    y = attr.y

    if x > stellar_vel.shape[0]:
        # update stellar velocity image box highlight in plot 1
        stellvel_box2.update(x=stellar_vel.shape[0] ,y=y)
        # update stellar dispersion box highlight in plot 3
        #stell_disp_box2.update(x=stellar_vel.shape[0],y=y)
        # update narrow-band image box highlight in plot 4
        #img_box2.update(x=img_data.shape[0],y=y)
        # update gas velocity image box highlight in plot 5
        gasvel_box2.update(x=stellar_vel.shape[0] ,y=y)
        # update gas velocity image box highlight in plot 9
        #halpha_box2.update(x=stellar_vel.shape[0] ,y=y)
        # update gas velocity image box highlight in plot 10
        #SFR_box2.update(x=stellar_vel.shape[0] ,y=y)

    elif x < 0:
        stellvel_box2.update(x=0 ,y=y)
        #stell_disp_box2.update(x=0,y=y)
        #img_box2.update(x=0,y=y)
        gasvel_box2.update(x=0 ,y=y)
        #halpha_box2.update(x=0 ,y=y)
        #SFR_box2.update(x=0 ,y=y)

    elif y > stellar_vel.shape[1]:
        stellvel_box2.update(x=x ,y=stellar_vel.shape[1])
        #stell_disp_box2.update(x=x,y=stellar_vel.shape[1])
        #img_box2.update(x=x,y=img_data.shape[1])
        gasvel_box2.update(x=x ,y=stellar_vel.shape[1])
        #halpha_box2.update(x=x ,y=stellar_vel.shape[1])
        #SFR_box2.update(x=x ,y=stellar_vel.shape[1])

    elif y < 0:
        stellvel_box2.update(x=x ,y=0)
        #stell_disp_box2.update(x=x,y=0)
        #img_box2.update(x=x,y=0)
        gasvel_box2.update(x=x ,y=0)
        #halpha_box2.update(x=x ,y=0)
        #SFR_box2.update(x=x ,y=0)

    else:
        stellvel_box2.update(x=x ,y=y)
        #stell_disp_box2.update(x=x,y=y)
        #img_box2.update(x=x,y=y)
        gasvel_box2.update(x=x ,y=y)
        #halpha_box2.update(x=x ,y=y)
        #SFR_box2.update(x=x ,y=y)


# update each event on each plot
p1.on_event(events.Tap,update_data)
p1.on_event(events.MouseMove,update_box)

# p3.on_event(events.Tap,update_data)
# p3.on_event(events.MouseMove,update_box)
#
# p4.on_event(events.Tap,update_data)
# p4.on_event(events.MouseMove,update_box)

p5.on_event(events.Tap,update_data)
p5.on_event(events.MouseMove,update_box)

# p9.on_event(events.Tap,update_data)
# p9.on_event(events.MouseMove,update_box)
#
# p10.on_event(events.Tap,update_data)
# p10.on_event(events.MouseMove,update_box)

layout_row1 = row(children=[p1,p5])
curdoc().add_root(layout_row1)
layout_row2 = row(children=[p6])
curdoc().add_root(layout_row2)

# layout_row1 = row(children=[p5,p1,p3])
# curdoc().add_root(layout_row1)
# layout_row2 = row(children=[p6,g,grid8])
# curdoc().add_root(layout_row2)
#layout_row3 =row(children=[p2,p1,p3])
#curdoc().add_root(layout_row3)
# curdoc().add_root(row(children=[p5,p6]))
# curdoc().add_root(row(children=[p1,p2]))
# curdoc().add_root(row(children=[p3,p4]))