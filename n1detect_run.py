#!/usr/bin/env python3
"""
Early response detection - docker entry-point
=====================================================
Entry-point python script for the automatic detection of early responses (N1) in CCEP data.


Copyright 2020, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys
import argparse
import os
from math import isnan, ceil
from glob import glob

from bids_validator import BIDSValidator
import numpy as np
import scipy.io as sio
from matplotlib import cm

from n1detect_config import default_config, read_config, write_config
from functions.load_bids import load_channel_info, load_event_info, load_data_epochs_averages
from functions.ieeg_detect_n1 import ieeg_detect_n1
from functions.visualization import create_figure
from functions.misc import print_progressbar, is_number



#
# constants
#
VALID_FORMAT_EXTENSIONS         = ('.mefd', '.edf', '.vhdr', '.vmrk', '.eeg')   # valid data format to search for (European Data Format, BrainVision and MEF3)
SUBPLOT_LAYOUT_RATIO            = (4, 3)
OUTPUT_IMAGE_RESOLUTION         = (2000, 2000)                                  # ; it is advisable to keep the resolution ratio in line with the SUBPLOT_LAYOUT_RATIO


#
# version and helper functions
#

__version__ = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'version')).read()


#
# define and parse the input arguments
#
parser = argparse.ArgumentParser(description='BIDS App for the automatic detection of early responses (N1) in CCEP data.')
parser.add_argument('bids_dir',
                    help='The directory with the input dataset formatted according to the BIDS standard.')
parser.add_argument('output_dir',
                    help='The directory where the output files should be stored. If you are running group level '
                         'analysis this folder should be prepopulated with the results of the participant level analysis.')
parser.add_argument('--participant_label',
                    help='The label(s) of the participant(s) that should be analyzed. The label corresponds '
                         'to sub-<participant_label> from the BIDS spec (so it does not include "sub-"). If this '
                         'parameter is not provided all subjects will be analyzed. Multiple participants can be '
                         'specified with a space separated list.',
                    nargs="+")
parser.add_argument('--subset_search_pattern',
                    help='The subset(s) of data that should be analyzed. The pattern should be part of a BIDS '
                         'compliant folder name (e.g. "task-ccep_run-01"). If this parameter is not provided all '
                         'the found subset(s) will be analyzed. Multiple subsets can be specified with a space '
                         'separated list.',
                    nargs="+")
parser.add_argument('--format_extension',
                    help='The data format(s) to include. The format(s) should be specified by their '
                         'extension (e.g. ".edf"). If this parameter is not provided, then by default the European '
                         'Data Format (''.edf''), BrainVision (''.vhdr'', ''.vmrk'', ''.eeg'') and MEF3 (''.mefd'') '
                         'formats will be included. Multiple formats can be specified with a space separated list.',
                    nargs="+")
parser.add_argument('--config_filepath',
                    help='Configures the app according to the settings in the JSON configuration file')
parser.add_argument('--skip_bids_validator',
                    help='Whether or not to perform BIDS data-set validation',
                    action='store_true')
parser.add_argument('-v', '--version',
                    action='version',
                    version='N1Detection BIDS-App version {}'.format(__version__))
args = parser.parse_args()

#
# display application information
#
print('BIDS app:               Detect N1 - ' + __version__)
print('BIDS input dataset:     ' + args.bids_dir)
print('Output location:        ' + args.output_dir)
if args.config_filepath:
    print('Configuration file      ' + args.config_filepath)
print('')


#
# check if the input is a valid BIDS dataset
#
#if not args.skip_bids_validator:
#    if not BIDSValidator().is_bids(args.bids_dir):
#        print('Error: BIDS input dataset did not pass BIDS validator. Datasets can be validated online '
#                          'using the BIDS Validator (http://incf.github.io/bids-validator/')
#        exit(1)


#
# configure
#
config = default_config()

#  read the configuration file (if passed)
if args.config_filepath:
    config = read_config(args.config_filepath)
    if config is None:
        exit(1)

# TODO: check on the configuration values


#
# process per subject and subset
#

# make sure the output directory exists
if not os.path.exists(args.output_dir):
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        print("Error: could not create output directory (\'" + args.output_dir + "\')", file=sys.stderr)
        exit(1)

# list the subject to analyze (either based on the input parameter or list all in the BIDS_dir)
subjects_to_analyze = []
if args.participant_label:

    # user-specified subjects
    subjects_to_analyze = args.participant_label

else:

    # all subjects
    subject_dirs = glob(os.path.join(args.bids_dir, 'sub-*'))
    subjects_to_analyze = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]


#
for subject in subjects_to_analyze:

    # see if the subject is exists (in case the user specified the labels)
    if os.path.isdir(os.path.join(args.bids_dir, subject)):

        # retrieve the data formats to include
        if args.format_extension:
            extensions = args.format_extension
            for extension in extensions:
                if not any(extension in x for x in VALID_FORMAT_EXTENSIONS):
                    print('Error: invalid data format extension \'' + extension + '\'', file=sys.stderr)
                    exit(1)
        else:
            extensions = VALID_FORMAT_EXTENSIONS

        # build path patterns for the search of subsets
        subset_patterns = args.subset_search_pattern if args.subset_search_pattern else ('',)
        subsets = []
        modalities = ('*eeg',)                    # ieeg and eeg
        for extension in extensions:
            for modality in modalities:
                for subset_pattern in subset_patterns:
                    subsets += glob(os.path.join(args.bids_dir, subject, modality, '*' + subset_pattern + '*' + extension)) + \
                               glob(os.path.join(args.bids_dir, subject, '*', modality, '*' + subset_pattern + '*' + extension))

        # bring subsets with multiple formats down to one format (prioritized to occurrence in the extension var)
        for subset in subsets:
            subset_name = subset[:subset.rindex(".")]
            for subset_other in reversed(subsets):
                if not subset == subset_other:
                    subset_other_name = subset_other[:subset_other.rindex(".")]
                    if subset_name == subset_other_name:
                        subsets.remove(subset_other)

        # loop through the participant's subsets for analysis
        for subset in subsets:

            # message subset start
            print('------')
            print('Subset:                                          ' + subset)
            print('')
            print('Trial epoch window:                              ' + str(config['trials']['trial_epoch'][0]) + 's < stim onset < ' + str(config['trials']['trial_epoch'][1]) + 's  (window size ' + str(abs(config['trials']['trial_epoch'][1] - config['trials']['trial_epoch'][0])) + 's)')
            print('Trial baseline window with each trial:           ' + str(config['trials']['baseline_epoch'][0]) + 's : ' + str(config['trials']['baseline_epoch'][1]) + 's')
            print('Trial baseline normalization:                    ' + str(config['trials']['baseline_norm']))
            print('Concatenate bidirectional stimulated pairs:      ' + ('Yes' if config['trials']['concat_bidirectional_pairs'] else 'No'))
            print('')
            print('Peak search window:                              ' + str(config['n1_detect']['peak_search_epoch'][0]) + 's : ' + str(config['n1_detect']['peak_search_epoch'][1]) + 's')
            print('N1 search window:                                ' + str(config['n1_detect']['n1_search_epoch'][0]) + 's : ' + str(config['n1_detect']['n1_search_epoch'][1]) + 's')
            print('N1 baseline window:                              ' + str(config['n1_detect']['n1_baseline_epoch'][0]) + 's : ' + str(config['n1_detect']['n1_baseline_epoch'][1]) + 's')
            print('N1 baseline threshold factor:                    ' + str(config['n1_detect']['n1_baseline_threshold_factor']))
            print('')
            print('Visualization display window                     ' + str(config['visualization']['lim_epoch'][0]) + 's : ' + str(config['visualization']['lim_epoch'][1]) + 's')
            print('Visualization stimulation epoch                  ' + str(config['visualization']['stim_epoch'][0]) + 's : ' + str(config['visualization']['stim_epoch'][1]) + 's')
            print('Generate electrode images                        ' + ('Yes' if config['visualization']['generate_electrode_images'] else 'No'))
            print('Generate stimulation-pair images                 ' + ('Yes' if config['visualization']['generate_stimpair_images'] else 'No'))
            print('Generate matrix images                           ' + ('Yes' if config['visualization']['generate_matrix_images'] else 'No'))
            print('')

            #
            # gather metadata information
            #

            # derive the bids roots (subject/session and subset) from the full path
            bids_subjsess_root = os.path.commonprefix(glob(os.path.join(os.path.dirname(subset), '*.*')))[:-1]
            bids_subset_root = subset[:subset.rindex('_')]

            # retrieve the channel metadata from the channels.tsv file
            csv = load_channel_info(bids_subset_root + '_channels.tsv')
            if csv is None:
                exit(1)

            # sort out the good, the bad and the... non-ieeg
            electrode_labels = []
            channels_bad = []
            channels_non_ieeg = []
            for index, row in csv.iterrows():
                excluded = False
                if row['status'].lower() == 'bad':
                    channels_bad.append(row['name'])
                    #excluded = True
                if not row['type'].upper() in ('ECOG', 'SEEG'):
                    channels_non_ieeg.append(row['name'])
                    excluded = True
                if not excluded:
                    electrode_labels.append(row['name'])

            # print channel information
            print('Non-IEEG channels:                               ' + ' '.join(channels_non_ieeg))
            print('IEEG electrodes:                                 ' + ' '.join(electrode_labels))
            # TODO: print like pairs, on long lines, wrap around with indenting on top
            print('Bad channels:                                    ' + ' '.join(channels_bad))

            # check if there are channels
            if len(electrode_labels) == 0:
                print('Error: no channels were found', file=sys.stderr)
                exit(1)


            # retrieve the electrical stimulation trials (onsets and pairs) from the events.tsv file
            csv = load_event_info(bids_subset_root + '_events.tsv', ('trial_type', 'electrical_stimulation_site'))
            if csv is None:
                exit(1)

            # acquire the onset and electrode-pair for each stimulation
            trial_onsets = []
            trial_pairs = []
            for index, row in csv.iterrows():
                if row['trial_type'].lower() == 'electrical_stimulation':
                    if not is_number(row['onset']) or isnan(float(row['onset'])) or float(row['onset']) < 0:
                        print('Error: invalid onset \'' + row['onset'] + '\' in events, should be a numeric value >= 0', file=sys.stderr)
                        #exit(1)
                        continue

                    pair = row['electrical_stimulation_site'].split('-')
                    if not len(pair) == 2 or len(pair[0]) == 0 or len(pair[1]) == 0:
                        print('Error: electrical stimulation site \'' + row['electrical_stimulation_site'] + '\' invalid, should be two values seperated by a dash (e.g. CH01-CH02)', file=sys.stderr)
                        exit(1)
                    trial_onsets.append(float(row['onset']))
                    trial_pairs.append(pair)

            # check if there are trials
            if len(trial_onsets) == 0:
                print('Error: no trials were found', file=sys.stderr)
                exit(1)


            #
            # determine the stimulation-pairs conditions (and the trial and electrodes that belong to them)
            # (note that the 'concat_bidirectional_pairs' configuration setting is taken into account here)
            #

            stimpair_labels = []                # for each pair, the labels of the electrodes that were stimulated
            stimpair_trial_indices = []         # for each pair, the indices of the trials that were involved
            stimpair_trial_onsets = []          # for each pair, the indices of the trials that were involved
            stimpair_electrode_indices = []     # for each pair, the indices of the electrodes that were stimulated
            for iChannel0 in range(len(electrode_labels)):
                for iChannel1 in range(len(electrode_labels)):

                    # retrieve the indices of all the trials that concern this stim-pair
                    indices = []
                    if config['trials']['concat_bidirectional_pairs']:
                        # allow concatenation of bidirectional pairs, pair order does not matter
                        if not iChannel1 < iChannel0:
                            # unique pairs while ignoring pair order
                            indices = [i for i, x in enumerate(trial_pairs) if
                                       (x[0] == electrode_labels[iChannel0] and x[1] == electrode_labels[iChannel1]) or (x[0] == electrode_labels[iChannel1] and x[1] == electrode_labels[iChannel0])]

                    else:
                        # do not concatenate bidirectional pairs, pair order matters
                        indices = [i for i, x in enumerate(trial_pairs) if
                                   x[0] == electrode_labels[iChannel0] and x[1] == electrode_labels[iChannel1]]

                    # add the pair if there are trials for it
                    if len(indices) > 0:
                        stimpair_labels.append(electrode_labels[iChannel0] + '-' + electrode_labels[iChannel1])
                        stimpair_electrode_indices.append((iChannel0, iChannel1))
                        stimpair_trial_indices.append(indices)
                        stimpair_trial_onsets.append([trial_onsets[i] for i in indices])


            # display Pair/Trial information
            print('Stimulation pairs:                               ' + str(len(stimpair_labels)))
            for iPair in range(len(stimpair_labels)):
                if iPair % 5 == 0:
                    print('                                                 ', end='')
                print(str(stimpair_labels[iPair]) + ' (' + str(len(stimpair_trial_indices[iPair])) + ' trials)   ', end='')
                if iPair > 0 and ((iPair + 1) % 5 == 0):
                    print('')

            #
            # read and epoch the data
            #

            # read, normalize by median and average the trials within the condition
            # Note: 'load_data_epochs_averages' is used instead of 'load_data_epochs_averages' here because it is more
            #       memory efficient is only the averages are needed
            print('- Reading data...')
            sampling_rate, ccep_average = load_data_epochs_averages(subset, electrode_labels, stimpair_trial_onsets,
                                                                    trial_epoch=config['trials']['trial_epoch'],
                                                                    baseline_norm=config['trials']['baseline_norm'],
                                                                    baseline_epoch=config['trials']['baseline_epoch'])
            if sampling_rate is None or ccep_average is None:
                print('Error: Could not load data (' + subset + ')', file=sys.stderr)
                exit(1)

            # for each stimulation pair, NaN out the values of the electrodes that were stimulated
            for iPair in range(len(stimpair_labels)):
                ccep_average[stimpair_electrode_indices[iPair][0], iPair, :] = np.nan
                ccep_average[stimpair_electrode_indices[iPair][1], iPair, :] = np.nan

            # determine the sample of stimulus onset (counting from the epoch start)
            onset_sample = int(round(abs(config['trials']['trial_epoch'][0] * sampling_rate)))
            # todo: handle trial epochs which start after the trial onset


            #
            # prepare an output directory
            #

            # make sure a subject directory exists
            output_root = os.path.join(args.output_dir, os.path.basename(os.path.normpath(bids_subset_root)))
            if not os.path.exists(output_root):
                try:
                    os.makedirs(output_root)
                except OSError as e:
                    print("Error: could not create subset output directory (\'" + output_root + "\')", file=sys.stderr)
                    exit(1)

            # intermediate saving of the ccep data as .mat
            sio.savemat(os.path.join(output_root, 'ccep_data.mat'),
                        {'sampling_rate': sampling_rate,
                         'onset_sample': onset_sample,
                         'ccep_average': ccep_average,
                         'stimpair_labels': stimpair_labels,
                         'electrode_labels': electrode_labels})

            # write the configuration
            write_config(os.path.join(output_root, 'default_config.json'), config)


            #
            # perform the N1 detection
            #

            # detect N1s
            print('- Detecting N1s...')
            n1_peak_indices, n1_peak_amplitudes = ieeg_detect_n1(ccep_average, onset_sample, int(sampling_rate),
                                                                 peak_search_epoch=config['n1_detect']['peak_search_epoch'],
                                                                 n1_search_epoch=config['n1_detect']['n1_search_epoch'],
                                                                 baseline_epoch=config['n1_detect']['n1_baseline_epoch'],
                                                                 baseline_threshold_factor=config['n1_detect']['n1_baseline_threshold_factor'])
            if n1_peak_indices is None or n1_peak_amplitudes is None:
                print('Error: N1 detection failed', file=sys.stderr)
                exit(1)

            # intermediate saving of the data and N1 detection as .mat
            sio.savemat(os.path.join(output_root, 'ccep_data.mat'),
                        {'sampling_rate': sampling_rate,
                         'onset_sample': onset_sample,
                         'ccep_average': ccep_average,
                         'stimpair_labels': stimpair_labels,
                         'electrode_labels': electrode_labels,
                         'n1_peak_indices': n1_peak_indices,
                         'n1_peak_amplitudes': n1_peak_amplitudes})


            #
            # generate images
            #

            if config['visualization']['generate_electrode_images'] or \
                config['visualization']['generate_stimpair_images'] or \
                config['visualization']['generate_matrix_images']:

                #
                # prepare some settings for plotting
                #

                # generate the x-axis values
                # TODO: what if TRIAL_EPOCH_START is after the stimulus onset
                x = np.arange(ccep_average.shape[2])
                x = x / sampling_rate + config['trials']['trial_epoch'][0]

                # determine the range on the x axis where the stimulus was in samples
                # TODO: what if TRIAL_EPOCH_START is after the stimulus onset
                x_stim_start = int(round(abs(config['trials']['trial_epoch'][0] - config['visualization']['stim_epoch'][0]) * sampling_rate)) - 1
                x_stim_end = x_stim_start + int(ceil(abs(config['visualization']['stim_epoch'][1] - config['visualization']['stim_epoch'][0]) * sampling_rate)) - 1

                # calculate the legend x position
                x_legend = config['visualization']['lim_epoch'][1] - .13

                # adjust line and font sizes to resolution
                zero_line_thickness = OUTPUT_IMAGE_RESOLUTION[0] / 2000
                signal_line_thickness = OUTPUT_IMAGE_RESOLUTION[0] / 2000
                legend_line_thickness = OUTPUT_IMAGE_RESOLUTION[0] / 500
                title_font_size = round(OUTPUT_IMAGE_RESOLUTION[0] / 80)
                axis_label_font_size = round(OUTPUT_IMAGE_RESOLUTION[0] / 85)
                axis_texts_font_size = round(OUTPUT_IMAGE_RESOLUTION[0] / 100)
                legend_font_size = round(OUTPUT_IMAGE_RESOLUTION[0] / 90)


                #
                if config['visualization']['generate_electrode_images']:

                    #
                    # generate the electrodes plot
                    #

                    # message
                    print('- Generating electrode plots...')

                    # create a progress bar
                    print_progressbar(0, len(electrode_labels), prefix='Progress:', suffix='Complete', length=50)

                    # loop through electrodes
                    for iElec in range(len(electrode_labels)):

                        # create a figure and retrieve the axis
                        fig = create_figure(OUTPUT_IMAGE_RESOLUTION[0], OUTPUT_IMAGE_RESOLUTION[1], False)
                        ax = fig.gca()

                        # set the title
                        ax.set_title(electrode_labels[iElec], fontSize=title_font_size, fontweight='bold')

                        # loop through the stimulation-pairs
                        for iPair in range(len(stimpair_labels)):

                            # draw 0 line
                            y = np.empty((ccep_average.shape[2], 1))
                            #y.fill(iPair + 1)
                            y.fill(len(stimpair_labels) - iPair)
                            ax.plot(x, y, linewidth=zero_line_thickness, color=(0.8, 0.8, 0.8))

                            # retrieve the signal
                            y = ccep_average[iElec, iPair, :] / 500
                            #y += iPair + 1
                            y += len(stimpair_labels) - iPair

                            # nan out the stimulation
                            y[x_stim_start:x_stim_end] = np.nan

                            # plot the signal
                            ax.plot(x, y, linewidth=signal_line_thickness)

                            # if N1 is detected, plot it
                            if not isnan(n1_peak_indices[iElec, iPair]):
                                xN1 = n1_peak_indices[iElec, iPair] / sampling_rate + config['trials']['trial_epoch'][0]
                                yN1 = n1_peak_amplitudes[iElec, iPair] / 500
                                #yN1 += iPair + 1
                                yN1 += len(stimpair_labels) - iPair
                                ax.plot(xN1, yN1, 'bo')

                        ax.set_xlabel('\ntime (s)', fontSize=axis_label_font_size)
                        ax.set_xlim(config['visualization']['lim_epoch'])
                        for label in ax.get_xticklabels():
                            label.set_fontsize(axis_texts_font_size)

                        # set the y axis
                        ax.set_ylabel('Stimulated electrode-pair\n', fontSize=axis_label_font_size)
                        ax.set_ylim((0, len(stimpair_labels) + 2))
                        ax.set_yticks(np.arange(1, len(stimpair_labels) + 1, 1))
                        #ax.set_yticklabels(pair_labels, fontSize=axis_texts_font_size)
                        ax.set_yticklabels(np.flip(stimpair_labels), fontSize=axis_texts_font_size)

                        # draw legend
                        ax.plot([x_legend, x_legend], [2.1, 2.9], linewidth=legend_line_thickness, color=(0, 0, 0))
                        ax.text(x_legend + .01, 2.3, '500 \u03bcV', fontSize=legend_font_size)

                        # Hide the right and top spines
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)

                        # display/save figure
                        # fig.show()
                        #fig.savefig(os.path.join(output_root, 'electrode_' + str(channels_include[iElec]) + '.png'))
                        fig.savefig(os.path.join(output_root, 'electrode_' + str(electrode_labels[iElec]) + '.png'), bbox_inches='tight')

                        # update progress bar
                        print_progressbar(iElec + 1, len(electrode_labels), prefix='Progress:', suffix='Complete', length=50)

                #
                if config['visualization']['generate_stimpair_images']:

                    #
                    # generate the stimulation-pair plots
                    #

                    # message
                    print('- Generating stimulation-pair plots...')

                    # create a progress bar
                    print_progressbar(0, len(stimpair_labels), prefix='Progress:', suffix='Complete', length=50)

                    # loop through the stimulation-pairs
                    for iPair in range(len(stimpair_labels)):

                        # create a figure and retrieve the axis
                        fig = create_figure(OUTPUT_IMAGE_RESOLUTION[0], OUTPUT_IMAGE_RESOLUTION[1], False)
                        ax = fig.gca()

                        # set the title
                        ax.set_title(stimpair_labels[iPair], fontSize=title_font_size, fontweight='bold')

                        # loop through the electrodes
                        for iElec in range(len(electrode_labels)):

                            # draw 0 line
                            y = np.empty((ccep_average.shape[2], 1))
                            #y.fill(iElec + 1)
                            y.fill(len(electrode_labels) - iElec)
                            ax.plot(x, y, linewidth=zero_line_thickness, color=(0.8, 0.8, 0.8))

                            # retrieve the signal
                            y = ccep_average[iElec, iPair, :] / 500
                            #y += iElec + 1
                            y += len(electrode_labels) - iElec

                            # nan out the stimulation
                            y[x_stim_start:x_stim_end] = np.nan

                            # plot the signal
                            ax.plot(x, y, linewidth=signal_line_thickness)

                            # if N1 is detected, plot it
                            if not isnan(n1_peak_indices[iElec, iPair]):
                                xN1 = n1_peak_indices[iElec, iPair] / sampling_rate + config['trials']['trial_epoch'][0]
                                yN1 = n1_peak_amplitudes[iElec, iPair] / 500
                                #yN1 += iElec + 1
                                yN1 += len(electrode_labels) - iElec
                                ax.plot(xN1, yN1, 'bo')

                        ax.set_xlabel('\ntime (s)', fontSize=axis_label_font_size)
                        ax.set_xlim(config['visualization']['lim_epoch'])
                        for label in ax.get_xticklabels():
                            label.set_fontsize(axis_texts_font_size)

                        # set the y axis
                        ax.set_ylabel('Measured electrodes\n', fontSize=axis_label_font_size)
                        ax.set_ylim((0, len(electrode_labels) + 2))
                        ax.set_yticks(np.arange(1, len(electrode_labels) + 1, 1))
                        #ax.set_yticklabels(channels_include, fontSize=axis_texts_font_size)
                        ax.set_yticklabels(np.flip(electrode_labels), fontSize=axis_texts_font_size)

                        # draw legend
                        ax.plot([x_legend, x_legend], [2.1, 2.9], linewidth=legend_line_thickness, color=(0, 0, 0))
                        ax.text(x_legend + .01, 2.3, '500 \u03bcV', fontSize=legend_font_size)

                        # Hide the right and top spines
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)

                        # display/save figure
                        # fig.show()
                        #fig.savefig(os.path.join(output_root, 'stimpair_' + str(pairs_label[iPair]) + '.png'))
                        fig.savefig(os.path.join(output_root, 'stimpair_' + str(stimpair_labels[iPair]) + '.png'), bbox_inches='tight')

                        # update progress bar
                        print_progressbar(iPair + 1, len(stimpair_labels), prefix='Progress:', suffix='Complete', length=50)

                #
                if config['visualization']['generate_matrix_images']:

                    #
                    # Amplitude matrix
                    #

                    # create a figure and retrieve the axis
                    fig = create_figure(OUTPUT_IMAGE_RESOLUTION[0], OUTPUT_IMAGE_RESOLUTION[1], False)
                    ax = fig.gca()

                    #
                    matrix_amplitudes = n1_peak_amplitudes
                    #matrix_amplitudes[np.isnan(matrix_amplitudes)] = 0
                    matrix_amplitudes *= -1

                    # create a color map
                    cmap = cm.get_cmap('viridis')
                    cmap.set_bad((.7, .7, .7, 1))

                    # draw the matrix
                    im = ax.imshow(np.transpose(matrix_amplitudes), origin='upper', vmin=0, vmax=500, cmap=cmap)

                    # set labels and ticks
                    ax.set_yticks(np.arange(0, len(stimpair_labels), 1))
                    ax.set_yticklabels(stimpair_labels, fontSize=axis_texts_font_size)
                    ax.set_xticks(np.arange(0, len(electrode_labels), 1))
                    ax.set_xticklabels(electrode_labels, rotation=90, fontSize=axis_texts_font_size)
                    ax.set_xlabel('\nMeasured electrode', fontSize=axis_label_font_size)
                    ax.set_ylabel('Stimulated electrode-pair\n', fontSize=axis_label_font_size)

                    # set a color-bar
                    cbar = fig.colorbar(im)
                    cbar.set_ticks([0, 100, 200, 300, 400, 500])
                    cbar.ax.set_yticklabels(['0', '-100 \u03bcV', '-200 \u03bcV', '-300 \u03bcV', '-400 \u03bcV', '-500 \u03bcV'], fontSize=legend_font_size - 4)
                    # TODO: colorbar to high

                    # display/save figure
                    # fig.show()
                    #fig.savefig(os.path.join(output_root, 'stimpair_' + str(pairs_label[iPair]) + '.png'))
                    fig.savefig(os.path.join(output_root, 'matrix_amplitude.png'), bbox_inches='tight')


                    #
                    # Latency matrix
                    #

                    # create a figure and retrieve the axis
                    fig = create_figure(OUTPUT_IMAGE_RESOLUTION[0], OUTPUT_IMAGE_RESOLUTION[1], False)
                    ax = fig.gca()

                    # retrieve the latencies and convert the indices (in samples) to time units (ms)
                    matrix_latencies = n1_peak_indices
                    matrix_latencies -= onset_sample
                    matrix_latencies /= sampling_rate
                    matrix_latencies *= 1000
                    #matrix_latencies[np.isnan(matrix_latencies)] = 0

                    # determine the latest
                    latest_N1 = np.nanmax(n1_peak_indices)
                    latest_N1 = int(ceil(latest_N1 / 10)) * 10

                    # create a color map
                    cmap = cm.get_cmap('viridis')
                    cmap.set_bad((.7, .7, .7, 1))

                    # draw the matrix
                    im = ax.imshow(np.transpose(matrix_latencies), origin='upper', vmin=0, cmap=cmap)

                    # set labels and ticks
                    ax.set_yticks(np.arange(0, len(stimpair_labels), 1))
                    ax.set_yticklabels(stimpair_labels, fontSize=axis_texts_font_size)
                    ax.set_xticks(np.arange(0, len(electrode_labels), 1))
                    ax.set_xticklabels(electrode_labels, rotation=90, fontSize=axis_texts_font_size)
                    ax.set_xlabel('\nMeasured electrode', fontSize=axis_label_font_size)
                    ax.set_ylabel('Stimulated electrode-pair\n', fontSize=axis_label_font_size)

                    # TODO: upper tick not displayed..
                    # generate the legend tick values
                    legend_tick_values = []
                    legend_tick_labels = []
                    for latency in range(0, latest_N1 + 10, 10):
                        legend_tick_values.append(latency)
                        legend_tick_labels.append(str(latency) + ' ms')

                    # set a colorbar
                    cbar = fig.colorbar(im)
                    cbar.set_ticks(legend_tick_values)
                    cbar.ax.set_yticklabels(legend_tick_labels, fontSize=legend_font_size - 4)
                    # TODO: colorbar to high

                    # display/save figure
                    # fig.show()
                    #fig.savefig(os.path.join(output_root, 'stimpair_' + str(pairs_label[iPair]) + '.png'))
                    fig.savefig(os.path.join(output_root, 'matrix_latency.png'), bbox_inches='tight')


            # message
            print('- Finished subset')

    else:
        #
        print('Warning: participant \'' + subject + '\' could not be found, skipping')


print('- Finished running')
