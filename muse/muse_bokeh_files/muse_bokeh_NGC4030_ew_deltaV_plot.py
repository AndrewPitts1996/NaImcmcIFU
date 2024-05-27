#!/usr/bin/env python
# coding: utf-8

import numpy as np
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, CustomJS, Slider,Div,SingleIntervalTicker,ColorBar
from bokeh.models import Text
from bokeh.io import curdoc
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn,unique
from bokeh.plotting import figure, show

# output equivalent width (ew) and deltaV data file
file = fits.open('NGC4030_0.6_deltaV_ew_output.fits')
table_compare = Table(file[1].data)
print(table_compare.colnames)
# masks for different ew bins
ew_bin0_mask = (table_compare['ews']>0) & (table_compare['ews']<0.25)
ew_bin1_mask = (table_compare['ews']>0.25) & (table_compare['ews']<0.5)
ew_bin2_mask = (table_compare['ews']>0.5) & (table_compare['ews']<0.75)
ew_bin3_mask = (table_compare['ews']>0.75)
masks = [ew_bin0_mask,ew_bin1_mask,ew_bin2_mask,ew_bin3_mask]

# Astropy tables for different ew bins
tables = [table_compare[ew_bin_mask] for ew_bin_mask in masks]

# Filtered and non-filtered ColumnDataSources for deltaV plot
deltaV_data = dict( delta_v = table_compare['vel_widths'].data,
                    sn_bin = table_compare['sn_bin'].data)

source_deltaV_non_filt = ColumnDataSource(deltaV_data)
source_deltaV_filt = ColumnDataSource(deltaV_data)

# Filtered and non-filtered ColumnDataSources for different ew bins
# convert astropy tables to panda data frames, keeping only necessary columns
columns = ['gas_vel','vel_widths','sn_bin','ews','ews_sig2','logN','b_d','C_f']
non_filtered_sources = [ColumnDataSource(table.to_pandas()[columns]) for table in tables]
filtered_sources = [ColumnDataSource(table.to_pandas()[columns]) for table in tables]

# top plot showing the all the deltaV data points
TOOLS = "box_select,lasso_select,reset,help"
deltaV = figure(width=600, height=400, title=None, tools=TOOLS,
              background_fill_color="#fafafa",x_range=(0,100), y_range=(0,100))
deltaV.scatter("sn_bin", "delta_v", source=source_deltaV_non_filt,alpha=0.1)
line_source = ColumnDataSource(data={'x': [0,100], 'y': [100,100]})
deltaV.line("x","y",line_width=2,color='black',source=line_source)
deltaV.scatter("sn_bin", "delta_v", source=source_deltaV_filt)
deltaV.yaxis.axis_label = "Delta V"
deltaV.xaxis.axis_label = "Bin S/N"

tot_pts = len(table_compare['vel_widths'].data)
num_pts = len(table_compare['vel_widths'].data)
num_pts_text = '#: {}/{} \n   {}%'.format(num_pts,tot_pts,(num_pts/tot_pts)*100)
text_source = ColumnDataSource(dict(x=[2], y=[5], text=[num_pts_text]))
glyph_deltaV = Text(x="x", y="y", text="text", text_color="black",text_font_size="14px")
deltaV.add_glyph(text_source,glyph_deltaV)

# loop to create each row of plots below the deltaV plot
vel_row0 = []
bd_row1 = []
colors = ['red','black','blue','green']
labels = ['0 < EW < 0.25','0.25 < EW < 0.5','0.5 < EW < 0.75','0.75 < EW']
text_sources = []

for i in range(len(tables)):
    vel_plots = figure(width=400, height=300, title=None, tools=TOOLS,
              background_fill_color="#fafafa",x_range=(0,100), y_range=(-600,600))
    bd_plots = figure(width=400, height=300, title=None, tools=TOOLS,
              background_fill_color="#fafafa",x_range=(0,100), y_range=(0,105))
    vel_plots.scatter("sn_bin", "gas_vel", color=colors[i], source=non_filtered_sources[i], alpha=0.1)
    bd_plots.scatter("sn_bin", "b_d", color=colors[i], source=non_filtered_sources[i], alpha=0.1)

    # labels
    if i == 0:
        vel_plots.yaxis.axis_label = "Na I Velocities (km/s)"
        bd_plots.yaxis.axis_label = "b_D (km/s)"
    vel_plots.xaxis.axis_label = "Bin S/N"
    bd_plots.xaxis.axis_label = "Bin S/N"

    vel_row0.append(vel_plots)
    bd_row1.append(bd_plots)

text_sources_gas_vel = []
for i in range(len(tables)):
    tot_pts_ew = len(non_filtered_sources[i].data['gas_vel'])
    num_pts_ew = len(filtered_sources[i].data['gas_vel'])
    num_pts_text_ew = '#: {}/{} \n    {}%'.format(num_pts_ew, tot_pts_ew, (num_pts_ew / num_pts_ew) * 100)
    text_source_gas_vel = ColumnDataSource(dict(x=[70], y=[-550], text=[num_pts_text_ew]))
    text_sources_gas_vel.append(text_source_gas_vel)

# loop only through filtered data
for i in range(len(tables)):
    # produce dynamic plots that change with the slider using the filtered data
    vel_row0[i].scatter("sn_bin", "gas_vel", color=colors[i], legend_label= labels[i],source=filtered_sources[i])
    bd_row1[i].scatter("sn_bin", "b_d", color=colors[i], legend_label= labels[i],source=filtered_sources[i])

    # add text
    gas_vel_glyph = Text(x="x", y="y", text="text", text_color="black", text_font_size="14px")
    vel_row0[i].add_glyph(text_sources_gas_vel[i], gas_vel_glyph)

# Slider to filter out data
def slider_callback(attr,old,new):
    line_source.data.update({'y': [new,new]})
    new_indices = table_compare['vel_widths'].data < new
    new_num_pts = len(new_indices[new_indices==True])
    new_num_pts_text = '#: {}/{} \n    {:.1f}%'.format(new_num_pts, tot_pts, (new_num_pts / tot_pts) * 100)
    text_source.data['text'] = [new_num_pts_text]

    deltaV_filt_new = {col_name: deltaV_data[col_name][new_indices] for col_name in deltaV_data.keys()}
    source_deltaV_filt.data = deltaV_filt_new

    for i in range(len(filtered_sources)):
        new_filt_indices = tables[i]['vel_widths'] < new
        filtered_data = {col_name: tables[i][col_name][new_filt_indices] for col_name in columns}
        filtered_sources[i].data = filtered_data

        tot_pts_ew = len(non_filtered_sources[i].data['vel_widths'])
        num_pts_ew = len(filtered_sources[i].data['vel_widths'])
        num_pts_text_ew = '#: {}/{} \n    {:.1f}%'.format(num_pts_ew, tot_pts_ew, (num_pts_ew / tot_pts_ew) * 100)
        text_sources_gas_vel[i].data['text'] = [num_pts_text_ew]

# Create a slider widget
slider_deltaV = Slider(start=0, end=100, value=100, step=2,title="Delta V Threshold")
# Attach callback function to the slider value change
slider_deltaV.on_change('value', slider_callback)

layout_row1 = row(children=[deltaV, slider_deltaV])
layout_row2 = row(children=vel_row0)
layout_row3 = row(children=bd_row1)
curdoc().add_root(layout_row1)
curdoc().add_root(layout_row2)
curdoc().add_root(layout_row3)