#!/usr/bin/env python
# coding: utf-8

import numpy as np
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, CustomJS, Slider,Div,SingleIntervalTicker,ColorBar
from bokeh.models import Text
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn,unique
from bokeh.plotting import figure, show
from bokeh.models.mappers import LinearColorMapper, LogColorMapper

#from bokeh.layouts import gridplot
from bokeh.io import curdoc
from bokeh import events
from bokeh.models import Span,HoverTool,Range1d,Scatter,TapTool,Rect
from bokeh.events import Tap
import colorcet as cc
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn,unique
from linetools.spectra.xspectrum1d import XSpectrum1D
import model_NaI
import continuum_normalize_NaI
from bokeh.plotting import figure, output_file, save
from bokeh.palettes import Turbo256,Viridis256,Inferno256
from IPython import embed


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
binid_map = NGC4030_cube0_6['BINID'].data[0]

# obtain flux array from MUSE cube
flux = NGC4030_cube0_6['FLUX'].data
model = NGC4030_cube0_6['MODEL'].data
ivar = NGC4030_cube0_6['IVAR'].data
error = np.sqrt(1/ivar)
wave = NGC4030_cube0_6['WAVE'].data

# Need LSF in km/s
# This gives LSF in Ang
# CAUTION: this does not convert air wavelengths
# to vacuum, or account for velocity offset of each bin
# speed of light in km/s
c = 2.998e5
fitlim = [5875.0, 5915.0]
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

# ---------------------------------------------------------------------------------------------------------------------

def NaImcmc(table, binid, redshift, binid_map, stellar_vel,
            flux, model, error, obswave, LSFvel):

    ind = binid_map == binid
    # single bin velocity
    binvel = stellar_vel[ind][0]
    # single flux, error and model spectrum corresponding to that bin
    flux_bin = np.ma.array(flux[:, ind][:, 0])
    err_bin = np.ma.array(error[:, ind][:, 0])
    mod_bin = np.ma.array(model[:, ind][:, 0])

    # speed of light in km/s
    c = 2.998e5
    # Na I doublet vacuum absorption wavelengths
    D2 = 5891.582  # in angstroms
    D1 = 5897.558  # in angstroms
    blim = [5850.0, 5870.0]
    rlim = [5910.0, 5930.0]
    # wavelength fitting range inside of NaI region
    fitlim = [5875.0, 5915.0]

    # Determine bin redshift:
    bin_z = redshift + ((1 + redshift) * binvel / c)
    restwave = obswave / (1.0 + bin_z)
    # convert wavelength array from angstroms to velocity space
    vel_space = ((restwave / D1) - 1) * c

    # observed flux (gas+continuum model)
    tot_ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, blim, rlim, FIT_FLG=0)
    # stellar continuum model
    mod_ndata = continuum_normalize_NaI.norm(restwave, flux_bin, err_bin, blim, rlim, FIT_FLG=1, smod=mod_bin)
    # gas flux (total flux / stellar continuum)
    gas_ndata = continuum_normalize_NaI.smod_norm(restwave, flux_bin, err_bin, mod_bin, blim, rlim)

    # Cut out NaI
    select_obs = np.where((tot_ndata['nwave'] > fitlim[0]) & (tot_ndata['nwave'] < fitlim[1]))
    select_mod = np.where((mod_ndata['nwave'] > fitlim[0]) & (mod_ndata['nwave'] < fitlim[1]))
    select_gas = np.where((gas_ndata['nwave'] > fitlim[0]) & (gas_ndata['nwave'] < fitlim[1]))

    restwave_NaI = tot_ndata['nwave'][select_obs].astype('float64')
    rest_vel = vel_space[select_obs].astype('float64')
    obsflux_NaI = tot_ndata['nflux'][select_obs].astype('float64')
    err_NaI = tot_ndata['nerr'][select_obs].astype('float64')
    sres_NaI = LSFvel
    mod_NaI = mod_ndata['nflux'][select_mod].astype('float64')
    gas_NaI = gas_ndata['nflux'][select_gas].astype('float64')
    gas_NaI_err = gas_ndata['nerr'][select_gas].astype('float64')

    data = {'wave': restwave_NaI, 'rest_vel': rest_vel, 'tot_flux': obsflux_NaI, 'tot_err': err_NaI,
            'model': mod_NaI, 'gas_flux': gas_NaI, 'gas_err': gas_NaI_err, 'velres': sres_NaI,
            'S_N': tot_ndata['s2n'],
            }

    binid_indx = table['bin'] == binid
    model_fit_bin = model_NaI.model_NaI(table['percentiles'][binid_indx][0][:, 0],
                                        data['velres'], data['wave'])

    return data, model_fit_bin
# ---------------------------------------------------------------------------------------------------------------------
# grab indices from selected data points within velocity EW bins and highlight their corresponding spatial
# x and y positions onto the Na I velocity map
def tap_callback(attr,old,new):
    #indices = table_source.selected.indices
    # grab x and y positions of selected data points and center it by adding 1.5
    # (x and y positions from the table are positions at the edge of the bins instead of the center)
    x_pix = np.array(table_dict['x'].data[new] + 1.5)
    y_pix = np.array(table_dict['y'].data[new] + 1.5)

    new_box_select_source = dict(x_pix=x_pix, y_pix=y_pix)
    box_select_source.data = new_box_select_source

# ---------------------------------------------------------------------------------------------------------------------

# load the data
file = fits.open('NGC4030_0.6_err_corr_mcmc_output.fits')
table_all = Table(file[1].data)

# filter out bad values
table_good = table_all[table_all['velocities'] > -999]
# exclude velocities with high confidence interval widths
vel_widths = np.absolute(table_good['vel_uncert_84th_percent'] - table_good['vel_uncert_16th_percent'])
vel_widths_filt = vel_widths[vel_widths < 100]
table_uniq = table_good[vel_widths < 100]

# ---------------------------------------------------------------------------------------------------------------------
# output equivalent width (ew) and deltaV data file
file = fits.open('NGC4030_0.6_deltaV_ew_output.fits')
table_compare = Table(file[1].data)
gas_vel_map = file[2].data

# masks for different ew bins
ew_bin0_mask = (table_compare['ews']>0) & (table_compare['ews']<0.25)
ew_bin1_mask = (table_compare['ews']>0.25) & (table_compare['ews']<0.5)
ew_bin2_mask = (table_compare['ews']>0.5) & (table_compare['ews']<0.75)
ew_bin3_mask = (table_compare['ews']>0.75)
masks = [ew_bin0_mask,ew_bin1_mask,ew_bin2_mask,ew_bin3_mask]

# Astropy tables for different ew bins
tables = [table_compare[ew_bin_mask] for ew_bin_mask in masks]
columns = ['gas_vel','sn_bin','ews','logN','b_d','C_f']
tables_nan = [table_compare[columns] for i in range(len(tables))]

table_dict = {'bin_id': table_compare['bin_id'],'x':table_compare['x'],'y':table_compare['y']}
for i in range(len(tables)):
    for col in columns:
        tables_nan[i][col][masks[i]==False] = np.nan
        dict_col_name = col+'_ew' + str(i)
        table_dict[dict_col_name] = tables_nan[i][col]

# data source holding all the data
table_source = ColumnDataSource(table_dict)
# loop to create annotation for each different ew bin plot
vel_row = []
colors = ['red','black','blue','green']
labels = ['0 < EW < 0.25','0.25 < EW < 0.5','0.5 < EW < 0.75','0.75 < EW']
text_sources = []

text_sources_gas_vel = []
tooltips_ews = []
for i in range(len(tables)):
    col = 'gas_vel_ew' + str(i)
    tot_pts_ew = len(table_source.data[col][np.isfinite(table_source.data[col])])
    num_pts_text_ew = '# data points: {}'.format(tot_pts_ew)
    text_source_gas_vel = ColumnDataSource(dict(x=[60], y=[-550], text=[num_pts_text_ew]))

    tooltips = [("Vel", "@gas_vel_ew" + str(i) + "{000.0}"), ("S/N", "@sn_bin_ew" + str(i) + "{000.0}"),
                ("EW", "@ews_ew" + str(i) + "{0.00}"), ("b_d", "@b_d_ew" + str(i) + "{000.0}"),
                ("C_f", "@C_f_ew" + str(i) + "{0.0}"),
                ("Bin ID", "@bin_id"), ("(x,y)", "[@x,@y]")]

    text_sources_gas_vel.append(text_source_gas_vel)
    tooltips_ews.append(tooltips)

TOOLS = "pan,box_zoom,wheel_zoom,lasso_select,box_select,tap,reset,help"
for i in range(len(tables)):
    vel_plot = figure(width=450, height=400, title=None, tools=TOOLS,
              background_fill_color="#fafafa",x_range=(0,100), y_range=(-600,600),match_aspect=True)

    vel_renderer = vel_plot.scatter("sn_bin_ew"+str(i), "gas_vel_ew"+str(i), color=colors[i], legend_label= labels[i],
                      source=table_source, alpha=0.5,size=5)

    vel_renderer.data_source.selected.on_change("indices", tap_callback)

    # add text
    gas_vel_glyph = Text(x="x", y="y", text="text", text_color="black", text_font_size="14px")
    vel_plot.add_glyph(text_sources_gas_vel[i], gas_vel_glyph)
    # have hover tool include wavelength, flux and model values using vline mode
    vel_plot.add_tools(HoverTool(tooltips=tooltips_ews[i], renderers=[vel_renderer], mode='mouse'))

    # labels
    if i == 0:
        vel_plot.yaxis.axis_label = "Na I Velocities (km/s)"
    vel_plot.xaxis.axis_label = "Bin S/N"
    vel_row.append(vel_plot)

# ColumnDataSource for gas velocity map
source_vel_map = ColumnDataSource( data = dict( gas_vel_map=[gas_vel_map],
                                       bin_id=[binid_map]))
# create tools and tooltips for each plot
tools1 = "pan,wheel_zoom,box_zoom,reset"
tooltips_vel_map = [("(x,y)", "($x{0}, $y{0})"), ("Vel", "@gas_vel_map{000.0}"),("Bin ID", "@bin_id")]
vel_map = figure(title='Na I Velocity Map',tools=tools1, width=500, height=500,
                     toolbar_location="below",background_fill_color="#fafafa",background_fill_alpha = 0.5,
                     match_aspect=True)
# plot 5 MUSE gas velocity map plot
color_mapper_vel_map = LinearColorMapper(palette=cc.coolwarm,low=-175,high=175)

gas_velocity = vel_map.image(image='gas_vel_map', source=source_vel_map,x=0,y=0,
                            dw=gas_vel_map.shape[0],dh=gas_vel_map.shape[1],
                            color_mapper=color_mapper_vel_map)

color_bar_vel_map = ColorBar(color_mapper=color_mapper_vel_map, label_standoff=12,name='Na I Velocity (km/s)')
vel_map.add_layout(color_bar_vel_map,'right')
vel_map.add_tools(HoverTool(tooltips = tooltips_vel_map, renderers = [gas_velocity]))
vel_map.x_range.range_padding = 0
vel_map.y_range.range_padding = 0

# add yellow box to highlight selected data points from the gas velocity EW bin plots
box_select_source = ColumnDataSource(dict(x_pix=[], y_pix=[]))
box_select = Rect(x="x_pix", y="y_pix", width=3, height=3, fill_color="yellow",fill_alpha=0.4)
vel_map.add_glyph(box_select_source, box_select)

# add grey image box to highlight cursor and selected region

# create gas velocity image box highlight in plot 5
# box 1: move the box along with the Mouse event
# box 2: move the box along with the Tap event
gas_vel_box_source1 = ColumnDataSource(dict(x=[155], y=[155]))
gas_vel_box_source2 = ColumnDataSource(dict(x=[], y=[]))
gasvel_box1 = Rect(x="x", y="y", width=3, height=3, fill_color="grey",fill_alpha=0.4)
gasvel_box2 = Rect(x="x", y="y", width=3, height=3, fill_color="grey",fill_alpha=0.4)
vel_map.add_glyph(gas_vel_box_source1,gasvel_box1)
vel_map.add_glyph(gas_vel_box_source1,gasvel_box2)


# Na I spectrum plot w/ best-fit absorption model
# get data around Na I region and the best-fit model to it
binid = binid_map[165, 165]
data_NaI, model_fit_bin_NaI = NaImcmc(table_uniq, binid, redshift, binid_map, stellar_vel,
                                      flux, model,error, wave, LSFvel)
spec_plot_source = ColumnDataSource(data = dict(rest_vel=data_NaI['rest_vel'], tot_flux=data_NaI['tot_flux'],
                                                err_flux=data_NaI['tot_err'],model_flux=data_NaI['model'],
                                                gas_flux = data_NaI['gas_flux'],
                                                gas_model_flux = model_fit_bin_NaI['modflx']))

spec_plot = figure(title='Na I Spectrum w/ MCMC Best-Fit Model',tools=tools1,width=500,height=500,
            toolbar_location="below", background_fill_color="#fafafa",background_fill_alpha = 0.5,
                     match_aspect=True,x_range=(-875,600), y_range=(0.55,1.15))

NaI_spec = spec_plot.step('rest_vel', 'tot_flux', source=spec_plot_source,legend_label = 'Total Flux',mode="center")
NaI_smodel = spec_plot.step('rest_vel', 'model_flux', source=spec_plot_source,legend_label = 'DAP Model',color='orange',mode="center")
NaI_gas = spec_plot.step('rest_vel', 'gas_flux', source=spec_plot_source,mode='center',color='green',legend_label='Gas Flux')
NaI_gas_mod = spec_plot.step('rest_vel','gas_model_flux',source=spec_plot_source,color='purple',legend_label='Gas Model',mode="center")
# have hover tool include wavelength, flux and model values using vline mode
#spec_plot.add_tools(HoverTool(tooltips = tools1, renderers =[NaI_spec] , mode='vline'))
spec_plot.legend.click_policy="hide"
spec_plot.yaxis.axis_label = "Normalized Flux"
spec_plot.xaxis.axis_label = "Velocity (km/s)"

# add informational text to the plot
binid_text = 'Bin ID: ' + str(binid)
ew_text = 'EW: {:.2f}'.format(table_compare['ews'][binid])
SN_text = 'S/N: {:.1f}'.format(data_NaI['S_N'])
vel_text = 'Velocity (km/s): {:.1f}'.format(table_compare['gas_vel'][binid])
bd_text = 'b_d (km/s): {:.1f} '.format(table_compare['b_d'][binid])
logN_text = 'logN (cm-2): {:.2f} '.format(table_compare['logN'][binid])
Cf_text = 'C_f: {:.2f}'.format(table_compare['C_f'][binid])

spec_text_source = ColumnDataSource(dict( binid_text = [binid_text], binid_x=[-800], binid_y=[0.67],
                                            ew_text = [ew_text], ew_x=[-800], ew_y=[0.635],
                                            SN_text = [SN_text], SN_x=[-800], SN_y=[0.6],
                                            vel_text=[vel_text], vel_x=[100], vel_y=[0.67],
                                            bd_text = [bd_text], bd_x=[100], bd_y=[0.635],
                                            logN_text=[logN_text], logN_x=[100], logN_y=[0.6],
                                            Cf_text = [Cf_text], Cf_x=[100], Cf_y=[0.565]))

binid_glyph = Text(x="binid_x", y="binid_y", text="binid_text", text_color="black", text_font_size="14px")
ew_glyph = Text(x="ew_x", y="ew_y", text="ew_text", text_color="black", text_font_size="14px")
SN_glyph = Text(x="SN_x", y="SN_y", text="SN_text", text_color="black", text_font_size="14px")
vel_glyph = Text(x="vel_x", y="vel_y", text="vel_text", text_color="black", text_font_size="14px")
bd_glyph = Text(x="bd_x", y="bd_y", text="bd_text", text_color="black", text_font_size="14px")
logN_glyph = Text(x="logN_x", y="logN_y", text="logN_text", text_color="black", text_font_size="14px")
Cf_glyph = Text(x="Cf_x", y="Cf_y", text="Cf_text", text_color="black", text_font_size="14px")

# create a vertical line in the plot from the emcee outputted velocity
vel_loc = Span(location=table_compare['gas_vel'][binid],
               dimension='height', line_color='grey',
               line_dash='dashed', line_width=2)
spec_plot.add_layout(vel_loc)

spec_plot.add_glyph(spec_text_source, binid_glyph)
spec_plot.add_glyph(spec_text_source, ew_glyph)
spec_plot.add_glyph(spec_text_source, SN_glyph)
spec_plot.add_glyph(spec_text_source, vel_glyph)
spec_plot.add_glyph(spec_text_source, bd_glyph)
spec_plot.add_glyph(spec_text_source, logN_glyph)
spec_plot.add_glyph(spec_text_source, Cf_glyph)


def update_data(attr):
    # x and y values when tapping the mouse over a pixel
    x = round(attr.x)
    y = round(attr.y)

    # get data around Na I region and the best-fit model to it
    binid = binid_map[int(y), int(x)]

    data_NaI, model_fit_bin_NaI = NaImcmc(table_uniq, binid, redshift, binid_map, stellar_vel,
                                          flux, model, error, wave, LSFvel)

    # update the source data to show the spectrum corresponding to that pixel
    spec_plot_source.data = dict(rest_vel=data_NaI['rest_vel'], tot_flux=data_NaI['tot_flux'], err_flux=data_NaI['tot_err'],
                        model_flux=data_NaI['model'], gas_flux=data_NaI['gas_flux'],
                        gas_model_flux=model_fit_bin_NaI['modflx'])

    #update annotations in spectrum plot
    bin_indx = np.where(table_uniq['bin'] == binid)
    binid_text = 'Bin ID: ' + str(binid)
    ew_text = 'EW: {:.2f}'.format(table_compare['ews'][bin_indx][0])
    SN_text = 'S/N: {:.1f}'.format(data_NaI['S_N'])
    vel_text = 'Velocity (km/s): {:.1f}'.format(table_compare['gas_vel'][bin_indx][0])
    bd_text = 'b_d (km/s): {:.1f} '.format(table_compare['b_d'][bin_indx][0])
    logN_text = 'logN (cm-2): {:.2f} '.format(table_compare['logN'][bin_indx][0])
    Cf_text = 'C_f: {:.2f}'.format(table_compare['C_f'][bin_indx][0])

    spec_text_source.data = dict( binid_text = [binid_text], binid_x=[-800], binid_y=[0.67],
                                            ew_text = [ew_text], ew_x=[-800], ew_y=[0.635],
                                            SN_text = [SN_text], SN_x=[-800], SN_y=[0.6],
                                            vel_text=[vel_text], vel_x=[100], vel_y=[0.67],
                                            bd_text = [bd_text], bd_x=[100], bd_y=[0.635],
                                            logN_text=[logN_text], logN_x=[100], logN_y=[0.6],
                                            Cf_text = [Cf_text], Cf_x=[100], Cf_y=[0.565])


    vel_loc.update(location=table_compare['gas_vel'][bin_indx][0])

    # update gas velocity box highlight in plot 5
    gasvel_box1.update(x=x, y=y)


def update_box(attr):
    # x and y values when moving the mouse over a pixel
    x = attr.x
    y = attr.y

    if x > gas_vel_map.shape[0]:
        # update stellar velocity image box highlight in plot 1
        #stellvel_box2.update(x=stellar_vel.shape[0], y=y)
        # update stellar dispersion box highlight in plot 3
        #equiv_width_box2.update(x=stellar_vel.shape[0], y=y)
        # update narrow-band image box highlight in plot 4
        # img_box2.update(x=img_data.shape[0],y=y)
        # update gas velocity image box highlight in plot 5
        gasvel_box2.update(x=gas_vel_map.shape[0], y=y)
        # update gas velocity image box highlight in plot 9
        # halpha_box2.update(x=stellar_vel.shape[0] ,y=y)
        # update gas velocity image box highlight in plot 10
        #SFR_box2.update(x=gas_vel_map.shape[0], y=y)

    elif x < 0:
        #stellvel_box2.update(x=0, y=y)
        #equiv_width_box2.update(x=0, y=y)
        # img_box2.update(x=0,y=y)
        gasvel_box2.update(x=0, y=y)
        # halpha_box2.update(x=0 ,y=y)
        #SFR_box2.update(x=0, y=y)

    elif y > stellar_vel.shape[1]:
        #stellvel_box2.update(x=x, y=stellar_vel.shape[1])
        #equiv_width_box2.update(x=x, y=stellar_vel.shape[1])
        # img_box2.update(x=x,y=img_data.shape[1])
        gasvel_box2.update(x=x, y=gas_vel_map.shape[1])
        # halpha_box2.update(x=x ,y=stellar_vel.shape[1])
        #SFR_box2.update(x=x, y=stellar_vel.shape[1])

    elif y < 0:
        #stellvel_box2.update(x=x, y=0)
        #equiv_width_box2.update(x=x, y=0)
        # img_box2.update(x=x,y=0)
        gasvel_box2.update(x=x, y=0)
        # halpha_box2.update(x=x ,y=0)
        #SFR_box2.update(x=x, y=0)

    else:
        #stellvel_box2.update(x=x, y=y)
        #equiv_width_box2.update(x=x, y=y)
        # img_box2.update(x=x,y=y)
        gasvel_box2.update(x=x, y=y)
        # halpha_box2.update(x=x ,y=y)
        #SFR_box2.update(x=x, y=y)

vel_map.on_event(events.Tap,update_data)
vel_map.on_event(events.MouseMove,update_box)

layout_row0 = row(children=[vel_map,spec_plot])
layout_row1 = row(children=vel_row)
#layout_row3 = row(children=bd_row1)
curdoc().add_root(layout_row0)
curdoc().add_root(layout_row1)
#curdoc().add_root(layout_row3)