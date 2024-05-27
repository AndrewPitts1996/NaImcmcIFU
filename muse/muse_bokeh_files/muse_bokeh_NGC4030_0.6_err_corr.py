#!/usr/bin/env python
# coding: utf-8

import numpy as np
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, CustomJS, Slider,Div,SingleIntervalTicker,ColorBar
from bokeh.models.mappers import LinearColorMapper, LogColorMapper
from bokeh.models import Step, Label,Text

#from bokeh.layouts import gridplot
from bokeh.io import curdoc
from bokeh import events
from bokeh.models import Span,HoverTool,Range1d,Scatter
import colorcet as cc
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn,unique
from astropy import units as u
from linetools.spectra.xspectrum1d import XSpectrum1D
import model_NaI
import continuum_normalize_NaI
#import glob
#import os
import numpy.ma as ma
import extinction
from bokeh.plotting import figure, output_file, save
from bokeh.palettes import Turbo256,Viridis256,Inferno256
#import bebi103


# MUSE cube directory
main_dir = '/Users/erickaguirre/mangadap/examples/'
main_dir2 = '/Users/erickaguirre/Desktop/SDSU_Research/Getting_used_to_MaNGA_DAP/'
main_dir3 = '/Users/erickaguirre/Desktop/DAP_outputs/'

# NGC 4030
NGC4030_output_dir = 'output0.6_NGC4030_NOISM_err_corr/'
NGC4030_cube_dir = 'SQUARE0.6-MILESHC-MASTARHC2-NOISM/1/1/'
# log cube
NGC4030_cube0_6_file = main_dir3+ NGC4030_output_dir+ NGC4030_cube_dir+\
                       'manga-1-1-LOGCUBE-SQUARE0.6-MILESHC-MASTARHC2-NOISM.fits'
NGC4030_cube0_6 = fits.open(NGC4030_cube0_6_file)

# log maps
NGC4030_map_file = main_dir3+ NGC4030_output_dir+ NGC4030_cube_dir+\
                   'manga-1-1-MAPS-SQUARE0.6-MILESHC-MASTARHC2-NOISM.fits'
NGC4030_map = fits.open(NGC4030_map_file)

# obtain stellar velocity, stellar dispersion and bin id data from the galaxy
stellar_vel =  NGC4030_map['STELLAR_VEL'].data
stellar_sigma = NGC4030_map['STELLAR_SIGMA'].data
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
# # H-alpha emission line flux from MUSE cube (1E-17 erg/s/cm^2/spaxel)
# # 23rd index = 24 channel
# halpha_map = NGC4030_map['EMLINE_GFLUX'].data[emline['Ha-6564'],:,:]
# # H-beta emission line flux from MUSE cube
# hbeta_map = NGC4030_map['EMLINE_GFLUX'].data[emline['Hb-4862'],:,:]
# # dust attenuation map
# ebv_map = 0.935 * np.log( (halpha_map/hbeta_map) /2.86 )
# # extinction in magnitudes using Calzettie et al. 2000 extinction curve
# k_lambda = extinction.calzetti00(np.array([6564.614]),a_v = 1, r_v = 3.1, unit='aa')
# # corrected h-alpha map
# halpha_map_corrected = halpha_map / 10**(-0.4*k_lambda*ebv_map)
# # replace NaN values w/ finite values
# ebv_map[np.logical_not(np.isfinite(ebv_map))] = -999
# # use good values in original h-alpha map to replace NaN values
# # in corrected h-alpha map due to nan values from ebv map
# nan_indx = np.where(np.isfinite(halpha_map_corrected) == False)
# halpha_map_corrected[nan_indx] = halpha_map[nan_indx]

# # recessional velocity of NGC 4030 (SIMBAD value) (km/s)
# v_recess = 1467.2 * (u.kilometer / u.second)
# # Hubble constant (km/s/Mpc)
# H_0 = 73 * (u.kilometer / (u.second * 1e6 * u.parsec) )
# # NGC4030 distance
# r = (v_recess / H_0).to(u.centimeter)
# # h-alpha luminosity map (ergs/s)
# L_halpha = 4*np.pi * r**2 * halpha_map_corrected * ( (1e-17 * u.erg) / (u.second * u.centimeter**2) )
# # SFR map from h-alpha luminosity in units of M_sun / year
# logSFR_map = np.log10(L_halpha.value) - 41.27
# logSFR_map[[np.logical_not(np.isfinite(logSFR_map))]] = -999
# halpha_map_corrected[np.logical_not(np.isfinite(halpha_map_corrected))] = -999

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

# output MCMC data file
file = fits.open('NGC4030_0.6_err_corr_mcmc_output.fits')
table_uniq = Table(file[1].data)
gas_vel_map = file[2].data

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
halpha_map_corrected = halpha_map / 10**(-0.4*k_lambda*ebv_map)
# replace NaN values w/ finite values
ebv_map[np.logical_not(np.isfinite(ebv_map))] = -999
# use good values in original h-alpha map to replace NaN values
# in corrected h-alpha map due to nan values from ebv map
nan_indx = np.where(np.isfinite(halpha_map_corrected) == False)
halpha_map_corrected[nan_indx] = halpha_map[nan_indx]

# recessional velocity of NGC 4030 (SIMBAD value) (km/s)
v_recess = 1467.2 * (u.kilometer / u.second)
# Hubble constant (km/s/Mpc)
H_0 = 73 * (u.kilometer / (u.second * 1e6 * u.parsec) )
# NGC4030 distance
r = (v_recess / H_0).to(u.centimeter)
# h-alpha luminosity map (ergs/s)
L_halpha = 4*np.pi * r**2 * halpha_map_corrected * ( (1e-17 * u.erg) / (u.second * u.centimeter**2) )
# SFR map from h-alpha luminosity in units of M_sun / year
logSFR_map = np.log10(L_halpha.value) - 41.27
logSFR_map[np.logical_not(np.isfinite(logSFR_map))] = np.nan
halpha_map_corrected[np.logical_not(np.isfinite(halpha_map_corrected))] = np.nan
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
    gas_ndata = continuum_normalize_NaI.smod_norm(restwave, flux_bin, err_bin, mod_bin, blim, rlim)

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

    binid_indx = table_uniq['bin'] == binid
    model_fit_bin = model_NaI.model_NaI(table_uniq['percentiles'][binid_indx][0][:, 0], data['velres'], data['wave'])

    return data,model_fit_bin
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# Na I equivalent width map
def get_vel_width_map(binid_map, table_uniq):

    # find velocity width between the 84th and 16th percentile from the lambda pdf
    vel_width_map = np.zeros(binid_map.shape)
    for binid in table_uniq['bin']:
        table_bin_indx = np.where(table_uniq['bin']== binid)
        gas_vel_84th = table_uniq['vel_uncert_84th_percent'][table_bin_indx][0]
        gas_vel_16th = table_uniq['vel_uncert_16th_percent'][table_bin_indx][0]

        if (gas_vel_84th > 0) & (gas_vel_16th<0):
            vel_width = np.absolute(gas_vel_84th + abs(gas_vel_16th) )
        else:
            vel_width = np.absolute(gas_vel_84th - gas_vel_16th)

        ind = binid_map == binid
        vel_width_map[ind] = vel_width

    return vel_width_map
# -----------------------------------------------------------------

# -----------------------------------------------------------------
#  corner plot for each spaxel bin
def corner_values(binid):
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
def get_text_labels(table_uniq,bin_indx):
    gas_vel = table_uniq['velocities'][bin_indx][0]
    gas_vel_84th = table_uniq['vel_uncert_84th_percent'][bin_indx][0]
    gas_vel_16th = table_uniq['vel_uncert_16th_percent'][bin_indx][0]
    gas_vel_text2 = 'NaI Vel: {:.1f} km/s'.format(gas_vel)
    gas_vel_text1 = '84th: {:.1f} 16th: {:.1f} km/s'.format(gas_vel_84th, gas_vel_16th)

    #percentiles_lamred = table_uniq['percentiles'][bin_indx][0][0]
    percentiles_logN = table_uniq['percentiles'][bin_indx][0][1]
    percentiles_bd = table_uniq['percentiles'][bin_indx][0][2]
    percentiles_Cf = table_uniq['percentiles'][bin_indx][0][3]

    #lamred_50th = percentiles_lamred[0]
    logN_50th = percentiles_logN[0]
    bd_50th = percentiles_bd[0]
    Cf_50th = percentiles_Cf[0]

    #lamred_sigma = np.mean([percentiles_lamred[1], percentiles_lamred[2]])
    logN_sigma = np.mean([percentiles_logN[1], percentiles_logN[2]])
    bd_sigma = np.mean([percentiles_bd[1], percentiles_bd[2]])
    Cf_sigma = np.mean([percentiles_Cf[1], percentiles_Cf[2]])

    #text_lamred = 'lamred: {:.2f} +/- {:.1e} angstr'.format(lamred_50th, lamred_sigma)
    text_logN = 'logN: {:.2f} +/- {:.1e} cm-2'.format(logN_50th, logN_sigma)
    text_bd = 'b_d: {:.2f} +/- {:.1e} km/s'.format(bd_50th, bd_sigma)
    text_Cf = 'C_f: {:.2f} +/- {:.1e}'.format(Cf_50th, Cf_sigma)

    return gas_vel_text1,gas_vel_text2, text_logN, text_bd, text_Cf
# -----------------------------------------------------------------

# get Na I absorption lines w/ respect to galaxy's redshift
z = 0.00489 # NGC 4030 redshift's
# z = 0.00489 # NGC 14042 redshift's
paths = '/Users/erickaguirre/Desktop/NaI_MCMC_output/NGC4030_0.6_err_corr/'
#table_uniq, gas_vel_map = mk_table_gasmap(paths)

# ColumnDataSource for stellar velocity map
source1 = ColumnDataSource( data = dict( stell_vel=[stellar_vel],
                                       bin_id_map=[binid_map] ) )
# get Na I D2 equivalent width map
vel_width_map = get_vel_width_map(binid_map, table_uniq)
# ColumnDataSource for equivalent width map
source2 = ColumnDataSource( data = dict( vel_width_map=[vel_width_map],
                                       bin_id_map=[binid_map] ) )

source3 = ColumnDataSource( data = dict( sfr_map=[logSFR_map],
                                       bin_id_map=[binid_map] ) )
# gas_vel_map[(ew_D2_map<0) | (binid_map == -1)] = np.nan
gas_vel_mask = ma.masked_where((vel_width_map>100) | (binid_map==-1), gas_vel_map)

# ColumnDataSource for gas velocity map
source5 = ColumnDataSource( data = dict( gas_vel=[gas_vel_mask],
                                       bin_id_map=[binid_map] ) )

# get data around Na I region and the best-fit model to it
binid = binid_map[195,159]
data_NaI, model_fit_bin_NaI = NaImcmc(binid, redshift, binid_map, stellar_vel, flux, model,error, wave,LSFvel)
source6 = ColumnDataSource(data = dict(wave=data_NaI['wave'], tot_flux=data_NaI['tot_flux'],err_flux=data_NaI['err'],
                                       model_flux=data_NaI['model'],gas_flux = data_NaI['gas_flux'],
                                       gas_model_flux = model_fit_bin_NaI['modflx']))

# # get corner plot given a binid
# params = [r'lambda_red', 'N', 'b_d','C_f']
# lambda_red, N, b_d, C_f = corner_values(binid)
# source7 = ColumnDataSource(data = dict(lambda_red = lambda_red, N = N, b_d = b_d, C_f = C_f))
# df = source7.to_df()
# g = bebi103.viz.corner(df,parameters=params,
#                         xtick_label_orientation=np.pi / 4,show_contours=True,
#                       frame_width = 100,frame_height = 105)
#
# # # plot samples run for each parameter
bin_indx = np.where(table_uniq['bin']== binid)
# samples = table_uniq['samples'][bin_indx][0]
#
# # Guess good model parameters
# lamred = 5897.5581
# logN = 14.5
# bD = 20.0
# Cf = 0.5
#
# lam_line = Span(location=lamred,dimension='width', line_color='grey', line_width=2)
# N_line = Span(location=logN,dimension='width', line_color='grey', line_width=2)
# bd_line = Span(location=bD,dimension='width', line_color='grey', line_width=2)
# Cf_line = Span(location=Cf,dimension='width', line_color='grey', line_width=2)
# iterations = np.arange(samples.shape[1])

# create tools and tooltips for each plot
tools1 = "pan,wheel_zoom,box_zoom,reset"

tooltips1 = [("Spaxel (x,y)", "($x{0}, $y{0})"), ("Velocity (km/s)", "@stell_vel{0.00}"),("Bin ID","@bin_id_map")]
tooltips2 = [("Spaxel (x,y)", "($x{0}, $y{0})"), ("Vel. Width", "@vel_width_map"),("Bin ID","@bin_id_map")]
tooltips3 = [("Spaxel (x,y)", "($x{0}, $y{0})"), ("SFR (M_sun/yr)", "@sfr_map"),("Bin ID","@bin_id_map")]
tooltips4 = [("(x,y)", "($x{0}, $y{0})"), ("Stellar Dispersion", "@image{0.000}")]
tooltips5 = [("Spaxel (x,y)", "($x{0}, $y{0})"), ("Velocity (km/s)", "@gas_vel{0.00}"),("Bin ID","@bin_id_map")]
tooltips6 = [("Wavelength", "@data_wave{0000.0}"),("Flux", "@tot_flux"),("Model","@model_flux"),("Gas", "@gas_flux"),
             ("Gas Model","@gas_model_flux")]
tooltips9 = [("(x,y)", "($x{0}, $y{0})"), ("Flux", "@image{0.000}")]


# create figures for each plot
p1 = figure(title='Stellar Velocity Map',tools=tools1, width=535, height=480,toolbar_location="right")
p2 = figure(title='Velocity Sigma Width Map',tools=tools1,width=535, height=480,toolbar_location="below")
p3 = figure(title='logSFR Map',tools=tools1,width=535, height=480,toolbar_location="below")
# p4 = figure(title='Dust Attenuation Map',tools=tools1, width=535, height=480,toolbar_location="below")
p5 = figure(title='Na I Velocity Map',tools=tools1, width=535, height=480,toolbar_location="below")
p6 = figure(title='DAP Na I Region w/ MCMC Best-Fit Model',tools=tools1,width=600,height=480,
            toolbar_location="left")
p8_1 = figure(title="Samples logN prior4 range (12.0-16.5)", y_axis_label='lambda_red')
p8_1.min_border=0
p8_2 = figure(y_axis_label='N')
p8_2.min_border=0
p8_3 = figure(y_axis_label='b_d')
p8_3.min_border=0
p8_4 = figure(y_axis_label='C_f')
p8_4.min_border=0

# p9 = figure(title='Corrected H-alpha Emission Flux Image',tools=tools1,width=535,height=480,
#             toolbar_location="below")
# p10 = figure(title='SFR H-alpha Map',tools=tools1, width=535, height=480,toolbar_location="below")

# p1.x_range.range_padding = p1.y_range.range_padding = 0

# plot 1 MUSE stellar velocity map plot
color_mapper1 = LinearColorMapper(palette=cc.coolwarm,low=-175,high=175)
color_bar1 = ColorBar(color_mapper=color_mapper1, label_standoff=12)
stellar_velocity = p1.image(image='stell_vel', source=source1,x=0,y=0,
                            dw=stellar_vel.shape[0],dh=stellar_vel.shape[1],
                            color_mapper=color_mapper1)
p1.add_layout(color_bar1,'right')
# add hover tool
p1.add_tools(HoverTool(tooltips = tooltips1, renderers = [stellar_velocity]))
# create stellar velocity image box highlight in plot 1
# box 1: move the box along with the Mouse event
# box 2: move the box along with the Tap event
stellvel_box1 = Scatter(x = 195 ,y = 159 ,size = 10, fill_alpha = 0.5,marker="square")
stellvel_box2 = Scatter(x = 195 ,y = 159 ,size = 10, fill_alpha = 0.5,marker="square")
p1.add_glyph(stellvel_box1)
p1.add_glyph(stellvel_box2)

# plot 2 Velocity Width map
color_mapper2 = LinearColorMapper(palette=list(reversed(Inferno256)),high=200,low=10)
color_bar2 = ColorBar(color_mapper=color_mapper2, label_standoff=12,title='Velocity Width (km/s)',
                      major_label_text_align = 'right',
                      title_standoff = 4,title_text_align='right',title_text_font_size = '15px')
equiv_width = p2.image(image='vel_width_map', source=source2,x=0,y=0,
                            dw=stellar_vel.shape[0],dh=stellar_vel.shape[1],
                            color_mapper=color_mapper2)
p2.add_layout(color_bar2,'right')
# add hover tool
p2.add_tools(HoverTool(tooltips = tooltips2, renderers = [equiv_width]))
# create stellar velocity image box highlight in plot 1
# box 1: move the box along with the Mouse event
# box 2: move the box along with the Tap event
equiv_width_box1 = Scatter(x = 195 ,y = 159 ,size = 10, fill_alpha = 0.5,marker="square")
equiv_width_box2 = Scatter(x = 195 ,y = 159 ,size = 10, fill_alpha = 0.5,marker="square")
p2.add_glyph(equiv_width_box1)
p2.add_glyph(equiv_width_box2)

# color_mapper10 = LinearColorMapper(palette='Inferno256',low=-6,high=-4.5)
# color_bar10 = ColorBar(color_mapper=color_mapper10, label_standoff=12,name='Na I Velocity (km/s')
# #color_bar10 = ColorBar(color_mapper=color_mapper10, label_standoff=12,title='Solar Mass/year',title_text_align = 'center')
# SFR_halpha = p10.image(image=[np.transpose(logSFR_map)], x=0, y=0, dw=logSFR_map.shape[0],dh=logSFR_map.shape[1],
#          color_mapper=color_mapper10)
# plot 3 log SFR map
color_mapper3 = LinearColorMapper(palette='Viridis256',low=-6,high=-4.5)
color_bar3 = ColorBar(color_mapper=color_mapper3, label_standoff=12,title=r"SFR M \[_{\odot}\] /yr \[^{-1}\]",
                      major_label_text_align = 'right',
                      title_standoff = 4,title_text_align='right',title_text_font_size = '15px')
sfr_map = p3.image(image='sfr_map', source=source3,x=0,y=0,
                            dw=stellar_vel.shape[0],dh=stellar_vel.shape[1],
                            color_mapper=color_mapper3)
p3.add_layout(color_bar3,'right')
# add hover tool
p3.add_tools(HoverTool(tooltips = tooltips3, renderers = [sfr_map]))
# create stellar velocity image box highlight in plot 1
# box 1: move the box along with the Mouse event
# box 2: move the box along with the Tap event
SFR_box1 = Scatter(x = 195 ,y = 159 ,size = 10, fill_alpha = 0.5,marker="square")
SFR_box2 = Scatter(x = 195 ,y = 159 ,size = 10, fill_alpha = 0.5,marker="square")
p3.add_glyph(SFR_box1)
p3.add_glyph(SFR_box2)

# plot 5 MUSE gas velocity map plot
color_mapper5 = LinearColorMapper(palette=cc.coolwarm,low=-50,high=50)
#color_mapper5 = LinearColorMapper(palette='BuRd',low=-50,high=50)
color_bar5 = ColorBar(color_mapper=color_mapper5, label_standoff=12,name='Na I Velocity (km/s)')
gas_velocity = p5.image(image='gas_vel', source=source5,x=0,y=0,
                            dw=gas_vel_map.shape[0],dh=gas_vel_map.shape[1],
                            color_mapper=color_mapper1)
                        #
p5.add_layout(color_bar5,'right')
# add hover tool
p5.add_tools(HoverTool(tooltips = tooltips5, renderers = [gas_velocity]))
# create gas velocity image box highlight in plot 5
# box 1: move the box along with the Mouse event
# box 2: move the box along with the Tap event
gasvel_box1 = Scatter(x = 195 ,y = 159 ,size = 10, fill_alpha = 0.5,marker="square")
gasvel_box2 = Scatter(x = 195 ,y = 159 ,size = 10, fill_alpha = 0.5,marker="square")
p5.add_glyph(gasvel_box1)
p5.add_glyph(gasvel_box2)
#export_png(p5, filename="gasvel_map.png")

# plot 6 DAP Na I spectrum plot w/ best-fit absorption model
NaI_spec = p6.step('wave', 'tot_flux', source=source6,legend_label = 'Total Flux',mode="center")
NaI_smodel = p6.step('wave', 'model_flux', source=source6,legend_label = 'DAP Model',color='orange',mode="center")
NaI_gas = p6.step('wave', 'gas_flux', source=source6,mode='center',color='green',legend_label='Gas Flux')
NaI_gas_mod = p6.step('wave','gas_model_flux',source=source6,color='purple',legend_label='Gas Model',mode="center")
# have hover tool include wavelength, flux and model values using vline mode
p6.add_tools(HoverTool(tooltips = tooltips6, renderers =[NaI_spec] , mode='vline'))
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
# text labels
gas_vel_text1,gas_vel_text2, text_logN, text_bd, text_Cf = get_text_labels(table_uniq,bin_indx)

gas_vel_label1_p6 = Label(x=5,y=1, x_units='screen', y_units='screen', text = gas_vel_text1)
gas_vel_label2_p6 = Label(x=5,y=16, x_units='screen', y_units='screen', text = gas_vel_text2)
#lamred_label_p6 = Label(x=5,y=1, x_units='screen', y_units='screen', text = text_lamred)
# left side of the plot
logN_label_p6 = Label(x=300,y=31, x_units='screen', y_units='screen', text = text_logN)
bd_label_p6 = Label(x=300,y=16, x_units='screen', y_units='screen', text = text_bd)
Cf_label_p6 = Label(x=300,y=1, x_units='screen', y_units='screen', text = text_Cf)

p6.add_layout(binid_label)
p6.add_layout(spaxel_label_p6)
p6.add_layout(gas_vel_label1_p6)
p6.add_layout(gas_vel_label2_p6)
#p6.add_layout(lamred_label_p6)
p6.add_layout(logN_label_p6)
p6.add_layout(bd_label_p6)
p6.add_layout(Cf_label_p6)

# # samples grid plot for each parameter
# for i in range(samples.shape[0]):
#     p8_1.line(iterations,samples[i,:,0],line_alpha=0.4,line_color='black')
#     p8_1.add_layout(lam_line)
#     p8_2.line(iterations,samples[i,:,1],line_alpha=0.4,line_color='black')
#     p8_2.add_layout(N_line)
#     p8_3.line(iterations,samples[i,:,2],line_alpha=0.4,line_color='black')
#     p8_3.add_layout(bd_line)
#     p8_4.line(iterations,samples[i,:,3],line_alpha=0.4,line_color='black')
#     p8_4.add_layout(Cf_line)
#
# grid8 = gridplot([[p8_1],[p8_2],[p8_3],[p8_4]],height=120)

# plot 3 MUSE stellar dispersion plot
# must give a vector of image data for image parameter
# Plasma265 (purple to orange)
# Blues8 (blue to white)
# commented color_mapper3 below is for 0.6 square binning
# color_mapper3 = LinearColorMapper(palette='Viridis256',low=np.percentile(stellar_sigma,20),high=0.12*np.max(stellar_sigma))
# color_mapper3 = LinearColorMapper(palette='Viridis256',low=55,high=115)
# color_bar3 = ColorBar(color_mapper=color_mapper3, label_standoff=12)
# stellar_dispersion = p3.image(image=[np.transpose(stellar_sigma)], x=0, y=0, dw=stellar_sigma.shape[0],dh=stellar_sigma.shape[1],
#                               color_mapper=color_mapper3)
# p3.add_layout(color_bar3,'right')
#
# # add hover tool
# p3.add_tools(HoverTool(tooltips = tooltips3, renderers = [stellar_dispersion]))
#
# # create stellar dispersion plot image box highlight
# # box 1: move the box along with the Mouse event
# # box 2: move the box along with the Tap event
# stell_disp_box1 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
# stell_disp_box2 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
# p3.add_glyph(stell_disp_box1)
# p3.add_glyph(stell_disp_box2)

# plot 4 MUSE cube NaI narrow band image
# must give a vector of image data for image parameter
# Plasma265 (purple to orange)
# Blues8 (blue to white)
# color_mapper4 = LogColorMapper(palette='Inferno256',low=np.percentile(img_data,10),high=np.max(img_data))
# color_bar4 = ColorBar(color_mapper=color_mapper4, label_standoff=12)
# narrow_band = p4.image(image=[np.transpose(img_data)], x=0, y=0, dw=img_data.shape[0],dh=img_data.shape[1],
#         color_mapper=color_mapper4)
# p4.add_layout(color_bar4,'right')
#
# # add hover tool
# p4.add_tools(HoverTool(tooltips = tooltips4, renderers = [narrow_band]))
# # create narrow-band image box highlight
# # box 1: move the box along with the Mouse event
# # box 2: move the box along with the Tap event
# img_box1 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
# img_box2 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
# p4.add_glyph(img_box1)
# p4.add_glyph(img_box2)

# create narrow-band image box highlight 
# box 1: move the box along with the Mouse event
# box 2: move the box along with the Tap event
# img_box1 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
# img_box2 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
# p4.add_glyph(img_box1)
# p4.add_glyph(img_box2)

# plot 10 MUSE SFR H-alpha map plot
# color_mapper10 = LinearColorMapper(palette='Inferno256',low=-6,high=-4.5)
# color_bar10 = ColorBar(color_mapper=color_mapper10, label_standoff=12,name='Na I Velocity (km/s')
# #color_bar10 = ColorBar(color_mapper=color_mapper10, label_standoff=12,title='Solar Mass/year',title_text_align = 'center')
# SFR_halpha = p10.image(image=[np.transpose(logSFR_map)], x=0, y=0, dw=logSFR_map.shape[0],dh=logSFR_map.shape[1],
#          color_mapper=color_mapper10)
# p10.add_layout(color_bar10,'right')
# # add hover tool
# p10.add_tools(HoverTool(tooltips = tooltips10, renderers = [SFR_halpha]))
# # create SFR image box highlight in plot 10
# # box 1: move the box along with the Mouse event
# # box 2: move the box along with the Tap event
# SFR_box1 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
# SFR_box2 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
# p10.add_glyph(SFR_box1)
# p10.add_glyph(SFR_box2)

# plot 9 MUSE cube H-alpha emission line flux image
# must give a vector of image data for image parameter
# Plasma265 (purple to orange)
# Blues8 (blue to white)
# color_mapper9 = LogColorMapper(palette='Inferno256',low=0.1,high=1.5*np.max(halpha_map_corrected))
# # color_bar9 = ColorBar(color_mapper=color_mapper9, label_standoff=12,title='Flux (1E-17 erg/s/cm^2/spaxel)',
# #                       title_text_align = 'center')
# color_bar9 = ColorBar(color_mapper=color_mapper9, label_standoff=12)
# halpha_image = p9.image(image=[np.transpose(halpha_map_corrected)], x=0, y=0,
#                         dw=halpha_map_corrected.shape[0],dh=halpha_map_corrected.shape[1],
#                         color_mapper=color_mapper9)
# p9.add_layout(color_bar9,'right')
#
# # add hover tool
# p9.add_tools(HoverTool(tooltips = tooltips9, renderers = [halpha_image]))
#
# # create H-alpha image box highlight in plot 9
# halpha_box1 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
# halpha_box2 = Square(x = 195 ,y = 195 ,size = 10, fill_alpha = 0.5)
# p9.add_glyph(halpha_box1)

# -----------------------------------------------------------------
def update_data(attr):
    # x and y values when tapping the mouse over a pixel
    x = round(attr.x)
    y = round(attr.y)

    # get data around Na I region and the best-fit model to it
    binid = binid_map[int(y),int(x)]
    bin_indx = np.where(table_uniq['bin']== binid)
    data_NaI, model_fit_bin_NaI = NaImcmc(binid, redshift, binid_map, stellar_vel, flux, model, error, wave, LSFvel)

    # update the source data to show the spectrum corresponding to that pixel
    source6.data = dict(wave=data_NaI['wave'], tot_flux=data_NaI['tot_flux'], err_flux=data_NaI['err'],
                                         model_flux=data_NaI['model'], gas_flux=data_NaI['gas_flux'],
                                         gas_model_flux=model_fit_bin_NaI['modflx'])

    # update bin ID and spaxel annotation in plot 6
    binid_text = 'Bin ID: ' + str(binid) 
    spaxel_text = 'Spaxel: (' +str(int(x)) +','+str(int(y))+')'
    # text labels
    gas_vel_text1,gas_vel_text2, text_logN, text_bd, text_Cf = get_text_labels(table_uniq,bin_indx)
    
    binid_label.update(text = binid_text)
    #spaxel_label_p2.update(text=spaxel_text)
    spaxel_label_p6.update(text = spaxel_text)
    gas_vel_label1_p6.update(text = gas_vel_text1)
    gas_vel_label2_p6.update(text=gas_vel_text2)
    #lamred_label_p6.update(text = text_lamred)
    logN_label_p6.update(text = text_logN)
    bd_label_p6.update(text = text_bd)
    Cf_label_p6.update(text = text_Cf)

    # update corner plot with new spaxel bin ID
    # lambda_red, N, b_d, C_f = corner_values(binid)
    # source7.data = dict(lambda_red = lambda_red, N = N, b_d = b_d, C_f = C_f)
    # df = source7.to_df()
    # g = bebi103.viz.corner(df,parameters=params,
    #                     xtick_label_orientation=np.pi / 4,show_contours=True,
    #                   frame_width = 100,frame_height = 105)
    #
    # layout_row2.children[1] = g
    
    # # update samples run plot
    # samples = table_uniq['samples'][bin_indx][0]
    # iterations = np.arange(samples.shape[1])
    #
    # p8_1_update = figure(title="Samples logN prior4 range (12.0-16.5)", y_axis_label='lambda_red')
    # p8_2_update = figure(y_axis_label='N' )
    # p8_3_update = figure(y_axis_label='b_d')
    # p8_4_update = figure(y_axis_label='C_f')
    #
    # # samples iteration grid plot for each parameter
    # for i in range(samples.shape[0]):
    #     p8_1_update.line(iterations,samples[i,:,0],line_alpha=0.4,line_color='black')
    #     p8_1_update.add_layout(lam_line)
    #     p8_2_update.line(iterations,samples[i,:,1],line_alpha=0.4,line_color='black')
    #     p8_2_update.add_layout(N_line)
    #     p8_3_update.line(iterations,samples[i,:,2],line_alpha=0.4,line_color='black')
    #     p8_3_update.add_layout(bd_line)
    #     p8_4_update.line(iterations,samples[i,:,3],line_alpha=0.4,line_color='black')
    #     p8_4_update.add_layout(Cf_line)
    #
    # grid8_update = gridplot([[p8_1_update],[p8_2_update],[p8_3_update],[p8_4_update]],height=120)
    # layout_row2.children[2] = grid8_update
    
#     # re-format y-range from new MUSE spectrum
#     continuum = np.mean(source2.data['flux'])    
#     p2.y_range.update(start = 0.6*continuum , end = 1.4*continuum)
    
    # update stellar velocity box highlight in plot 1
    stellvel_box1.update(x=x,y=y)
    
    # update stellar dispersion box highlight in plot 2
    equiv_width_box1.update(x=x,y=y)
    
    # update narrow-band box highlight in plot 4
    #img_box1.update(x=x,y=y)
    
    # update gas velocity box highlight in plot 5
    gasvel_box1.update(x=x,y=y)
    
    # update halpha box highlight in plot 9
    #halpha_box1.update(x=x,y=y)
    
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
        equiv_width_box2.update(x=stellar_vel.shape[0],y=y)
        # update narrow-band image box highlight in plot 4
        #img_box2.update(x=img_data.shape[0],y=y)
        # update gas velocity image box highlight in plot 5
        gasvel_box2.update(x=stellar_vel.shape[0] ,y=y)
        # update gas velocity image box highlight in plot 9
        #halpha_box2.update(x=stellar_vel.shape[0] ,y=y)
        # update gas velocity image box highlight in plot 10
        SFR_box2.update(x=stellar_vel.shape[0] ,y=y)
        
    elif x < 0:
        stellvel_box2.update(x=0 ,y=y)
        equiv_width_box2.update(x=0,y=y)
        #img_box2.update(x=0,y=y)
        gasvel_box2.update(x=0 ,y=y)
        #halpha_box2.update(x=0 ,y=y)
        SFR_box2.update(x=0 ,y=y)
        
    elif y > stellar_vel.shape[1]:
        stellvel_box2.update(x=x ,y=stellar_vel.shape[1])
        equiv_width_box2.update(x=x,y=stellar_vel.shape[1])
        #img_box2.update(x=x,y=img_data.shape[1])
        gasvel_box2.update(x=x ,y=stellar_vel.shape[1])
        #halpha_box2.update(x=x ,y=stellar_vel.shape[1])
        SFR_box2.update(x=x ,y=stellar_vel.shape[1])
        
    elif y < 0:
        stellvel_box2.update(x=x ,y=0)
        equiv_width_box2.update(x=x,y=0)
        #img_box2.update(x=x,y=0)
        gasvel_box2.update(x=x ,y=0)
        #halpha_box2.update(x=x ,y=0)
        SFR_box2.update(x=x ,y=0)
    
    else:
        stellvel_box2.update(x=x ,y=y)
        equiv_width_box2.update(x=x,y=y)
        #img_box2.update(x=x,y=y)
        gasvel_box2.update(x=x ,y=y)
        #halpha_box2.update(x=x ,y=y)
        SFR_box2.update(x=x ,y=y)
    

# update each event on each plot    
p1.on_event(events.Tap,update_data)
p1.on_event(events.MouseMove,update_box)

p2.on_event(events.Tap,update_data)
p2.on_event(events.MouseMove,update_box)

p3.on_event(events.Tap,update_data)
p3.on_event(events.MouseMove,update_box)

p5.on_event(events.Tap,update_data)
p5.on_event(events.MouseMove,update_box)

# p9.on_event(events.Tap,update_data)
# p9.on_event(events.MouseMove,update_box)

# p10.on_event(events.Tap,update_data)
# p10.on_event(events.MouseMove,update_box)

layout_row1 = row(children=[p5,p1])
curdoc().add_root(layout_row1)
# layout_row1 = row(children=[p5,p1,p3])
# curdoc().add_root(layout_row1)
# layout_row2 = row(children=[p6,g,grid8])
layout_row2 = row(children=[p6,p3])
curdoc().add_root(layout_row2)
#layout_row3 =row(children=[p2,p1,p3])
#curdoc().add_root(layout_row3)

html_file = output_file(filename="MUSE_bokeh_0.6_err_corr.html", title="MUSE Data Visualization Explorer")
# save the results to a file
save(html_file)
print('saving html file, hopefully!')