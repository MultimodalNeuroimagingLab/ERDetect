"""
Module for the generation of output images


Copyright 2022, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from math import ceil
from utils.misc import create_figure
import numpy as np
from matplotlib import cm

def calc_sizes_and_fonts(image_size, num_stim_pairs, num_channels):
    """

    image_size (int):       The size of the output image

    """
    plot_props = dict()

    # adjust line and font sizes to resolution
    plot_props['zero_line_thickness'] = image_size / 2000
    plot_props['signal_line_thickness'] = image_size / 2000
    plot_props['legend_line_thickness'] = image_size / 500
    plot_props['title_font_size'] = round(image_size / 80)
    plot_props['axis_label_font_size'] = round(image_size / 85)
    plot_props['axis_ticks_font_size'] = round(image_size / 100)
    plot_props['legend_font_size'] = round(image_size / 90)

    # Adjust the font sizes of the tick according to the number of items (minimum font-size remains 4)
    if num_stim_pairs > 36 and plot_props['axis_ticks_font_size'] > 4:
        plot_props['stimpair_axis_ticks_font_size'] = 4 + (plot_props['axis_ticks_font_size'] - 4) * (36.0 / num_stim_pairs)
    else:
        plot_props['stimpair_axis_ticks_font_size'] = plot_props['axis_ticks_font_size']
    if num_channels > 36 and plot_props['axis_ticks_font_size'] > 4:
        plot_props['electrode_axis_ticks_font_size'] = 4 + (plot_props['axis_ticks_font_size'] - 4) * (36.0 / num_channels)
    else:
        plot_props['electrode_axis_ticks_font_size'] = plot_props['axis_ticks_font_size']

    # account for the situation where there are only a small number of stimulation-pairs.
    if num_stim_pairs < 10:
        plot_props['stimpair_y_image_height'] = 500 + (image_size - 500) * (num_stim_pairs / 10)
    else:
        plot_props['stimpair_y_image_height'] = image_size

    # account for a high number of electrodes
    if num_channels > 50:
        plot_props['electrode_y_image_height'] = 500 + (image_size - 500) * (num_stim_pairs / 50)
    else:
        plot_props['electrode_y_image_height'] = image_size

    # if there are 10 times more electrodes than stimulation-pairs, then allow
    # the matrix to squeeze horizontally
    plot_props['matrix_aspect'] = 1
    element_ratio = num_channels / num_stim_pairs
    if element_ratio > 10:
        plot_props['matrix_aspect'] = element_ratio / 8

    return plot_props


def calc_matrix_image_size(stimpair_y_image_height, num_stim_pairs, num_channels):

    # calculate the image width based on the number of stim-pair and electrodes
    image_width = stimpair_y_image_height / num_stim_pairs * num_channels
    image_width += 800

    # make sure the image width does not exceed the matplotlib limit of 2**16
    if image_width >= 2 ** 16:
        factor = (2 ** 16 - 50) / image_width
        image_width = int(round(image_width * factor))
        image_height = int(round(stimpair_y_image_height * factor))
    else:
        image_height = stimpair_y_image_height

    return image_width, image_height


def gen_amplitude_matrix(stim_pairs, channels, plot_props, image_width, image_height, matrix_amplitudes, positive):
    """

    """

    # adjust the padding between the matrix and the color-bar based on the image width
    colorbar_padding = 0.01 if image_width < 2000 else (0.01 * (2000 / image_width))

    # create a figure and retrieve the axis
    fig = create_figure(image_width, image_height, False)
    ax = fig.gca()

    # create a color map
    cmap = cm.get_cmap("autumn").copy()
    cmap.set_bad((.7, .7, .7, 1))

    # draw the matrix
    im = ax.imshow(np.transpose(matrix_amplitudes), origin='upper', vmin=0, vmax=500, cmap=cmap, aspect=plot_props['matrix_aspect'])

    # set labels and ticks
    ax.set_yticks(np.arange(0, len(stim_pairs), 1))
    ax.set_yticklabels(stim_pairs, fontsize=plot_props['stimpair_axis_ticks_font_size'])
    ax.set_xticks(np.arange(0, len(channels), 1))
    ax.set_xticklabels(channels,
                       rotation=90,
                       fontsize=plot_props['stimpair_axis_ticks_font_size'])  # deliberately using stimpair-fs here
    ax.set_xlabel('\nMeasured electrode', fontsize=plot_props['axis_label_font_size'])
    ax.set_ylabel('Stimulated electrode-pair\n', fontsize=plot_props['axis_label_font_size'])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)

    # set a color-bar
    cbar = fig.colorbar(im, pad=colorbar_padding)
    cbar.set_ticks([0, 100, 200, 300, 400, 500])

    if positive:
        y_tick_labels = ['0', '100 \u03bcV', '200 \u03bcV', '300 \u03bcV', '400 \u03bcV', '>= 500 \u03bcV']
    else:
        y_tick_labels = ['0', '-100 \u03bcV', '-200 \u03bcV', '-300 \u03bcV', '-400 \u03bcV', '<= -500 \u03bcV']
    cbar.ax.set_yticklabels(y_tick_labels, fontsize=plot_props['legend_font_size'] - 4)

    cbar.outline.set_linewidth(1.5)

    return fig


def gen_latency_matrix(stim_pairs, channels, plot_props, image_width, image_height, matrix_latencies):

    # adjust the padding between the matrix and the color-bar based on the image width
    colorbar_padding = 0.01 if image_width < 2000 else (0.01 * (2000 / image_width))

    # create a figure and retrieve the axis
    fig = create_figure(image_width, image_height, False)
    ax = fig.gca()

    # determine the latest negative response
    latest_neg = np.nanmax(matrix_latencies)
    if np.isnan(latest_neg):
        latest_neg = 10
    latest_neg = int(ceil(latest_neg / 10)) * 10

    # create a color map
    cmap = cm.get_cmap('summer_r').copy()
    cmap.set_bad((.7, .7, .7, 1))

    # draw the matrix
    im = ax.imshow(np.transpose(matrix_latencies), origin='upper', vmin=0, cmap=cmap, aspect=plot_props['matrix_aspect'])

    # set labels and ticks
    ax.set_yticks(np.arange(0, len(stim_pairs), 1))
    ax.set_yticklabels(stim_pairs, fontsize=plot_props['stimpair_axis_ticks_font_size'])
    ax.set_xticks(np.arange(0, len(channels), 1))
    ax.set_xticklabels(channels,
                       rotation=90,
                       fontsize=plot_props['stimpair_axis_ticks_font_size'])  # deliberately using stimpair-fs here
    ax.set_xlabel('\nMeasured electrode', fontsize=plot_props['axis_label_font_size'])
    ax.set_ylabel('Stimulated electrode-pair\n', fontsize=plot_props['axis_label_font_size'])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)

    # generate the legend tick values
    legend_tick_values = []
    legend_tick_labels = []
    for latency in range(0, latest_neg + 10, 10):
        legend_tick_values.append(latency)
        legend_tick_labels.append(str(latency) + ' ms')

    # set the color limits for the image based on the range display in the legend
    im.set_clim([legend_tick_values[0], legend_tick_values[-1]])

    # set a color-bar
    cbar = fig.colorbar(im, pad=colorbar_padding)
    cbar.set_ticks(legend_tick_values)
    cbar.ax.set_yticklabels(legend_tick_labels, fontsize=plot_props['legend_font_size'] - 4)
    cbar.ax.invert_yaxis()
    cbar.outline.set_linewidth(1.5)

    return fig
