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
import logging
from math import isnan, ceil
from glob import glob

from bids_validator import BIDSValidator
import numpy as np
import scipy.io as sio
from matplotlib import cm

from n1detect_config import default_config, check_config, read_config, write_config
from functions.load_bids import load_channel_info, load_event_info, load_data_epochs_averages
from functions.ieeg_detect_n1 import ieeg_detect_n1
from functions.visualization import create_figure
from functions.misc import print_progressbar, is_number, CustomLoggingFormatter, multi_line_list, run_cmd


#
# constants
#
VALID_FORMAT_EXTENSIONS         = ('.mefd', '.edf', '.vhdr', '.vmrk', '.eeg')   # valid data format to search for (European Data Format, BrainVision and MEF3)
OUTPUT_IMAGE_SIZE               = 2000                                          # the number of pixels that is used as the "initial" height or width for the output images
LOGGING_CAPTION_INDENT_LENGTH   = 50                                            # the indent length of the caption in a logging output string

#
# version and logging
#

#
__version__ = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'version')).read()

#
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger_ch = logging.StreamHandler(stream=sys.stdout)
logger_ch.setFormatter(CustomLoggingFormatter())
logger.addHandler(logger_ch)


def log_indented_line(caption, text):
    logging.info(caption.ljust(LOGGING_CAPTION_INDENT_LENGTH, ' ') + text)


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
log_indented_line('BIDS app:', ('Detect N1 - ' + __version__))
log_indented_line('BIDS input path:', args.bids_dir)
log_indented_line('Output path:', args.output_dir)
if args.config_filepath:
    log_indented_line('Configuration file:', args.config_filepath)
logging.info('')


#
# check if the input is a valid BIDS dataset
#
if not args.skip_bids_validator:
    #process = run_cmd('bids-validator %s' % args.bids_dir)
    #logging.info(process.stdout)
    #if process.returncode != 0:
    #    logging.error('BIDS input dataset did not pass BIDS validator. Datasets can be validated online '
    #                    'using the BIDS Validator (http://incf.github.io/bids-validator/).\nUse the '
    #                    '--skip_bids_validator argument to run the detection without prior BIDS validation.')
    #    exit(1)
    bids_error = False
    for dir_, d, files in os.walk(args.bids_dir):
        for file in files:
            rel_file = os.path.relpath(dir_, args.bids_dir)
            if rel_file[0] == '.':
                rel_file = rel_file[1:]
            rel_file = os.path.join(rel_file, file)
            if not BIDSValidator().is_bids('/' + rel_file):
                logging.error('Invalid BIDS-file: ' + rel_file)
                bids_error = True
    if bids_error:
        logging.error('BIDS input dataset did not pass BIDS validator. Datasets can be validated online '
                        'using the BIDS Validator (http://incf.github.io/bids-validator/).\nUse the '
                        '--skip_bids_validator argument to run the detection without prior BIDS validation.')
        exit(1)


#
# configure
#
config = default_config()

#  read the configuration file (if passed)
if args.config_filepath:
    config = read_config(args.config_filepath)
    if config is None:
        logging.error('Could not load the configuration file, exiting...')
        exit(1)

# check on the configuration values
if not check_config(config):
    logging.error('Invalid configuration, exiting...')
    exit(1)

# print configuration information
log_indented_line('Trial epoch window:', str(config['trials']['trial_epoch'][0]) + 's < stim onset < ' + str(config['trials']['trial_epoch'][1]) + 's  (window size ' + str(abs(config['trials']['trial_epoch'][1] - config['trials']['trial_epoch'][0])) + 's)')
log_indented_line('Trial out-of-bounds handling:', str(config['trials']['out_of_bounds_handling']))
log_indented_line('Trial baseline window:', str(config['trials']['baseline_epoch'][0]) + 's : ' + str(config['trials']['baseline_epoch'][1]) + 's')
log_indented_line('Trial baseline normalization:', str(config['trials']['baseline_norm']))
log_indented_line('Concatenate bidirectional stimulated pairs:', ('Yes' if config['trials']['concat_bidirectional_pairs'] else 'No'))
log_indented_line('Minimum # of required stimulus-pair trials:', str(config['trials']['minimum_stimpair_trials']))
logging.info(multi_line_list(config['channels']['types'], LOGGING_CAPTION_INDENT_LENGTH, 'Channels types:', 20, ' '))
logging.info('')
log_indented_line('Peak search window:', str(config['n1_detect']['peak_search_epoch'][0]) + 's : ' + str(config['n1_detect']['peak_search_epoch'][1]) + 's')
log_indented_line('N1 search window:', str(config['n1_detect']['n1_search_epoch'][0]) + 's : ' + str(config['n1_detect']['n1_search_epoch'][1]) + 's')
log_indented_line('N1 baseline window:', str(config['n1_detect']['n1_baseline_epoch'][0]) + 's : ' + str(config['n1_detect']['n1_baseline_epoch'][1]) + 's')
log_indented_line('N1 baseline threshold factor:', str(config['n1_detect']['n1_baseline_threshold_factor']))
logging.info('')
log_indented_line('Visualization x-axis epoch:', str(config['visualization']['x_axis_epoch'][0]) + 's : ' + str(config['visualization']['x_axis_epoch'][1]) + 's')
log_indented_line('Visualization blank stimulation epoch:', str(config['visualization']['blank_stim_epoch'][0]) + 's : ' + str(config['visualization']['blank_stim_epoch'][1]) + 's')
log_indented_line('Generate electrode images:', ('Yes' if config['visualization']['generate_electrode_images'] else 'No'))
log_indented_line('Generate stimulation-pair images:', ('Yes' if config['visualization']['generate_stimpair_images'] else 'No'))
log_indented_line('Generate matrix images:', ('Yes' if config['visualization']['generate_matrix_images'] else 'No'))
logging.info('')


#
# process per subject and subset
#

# make sure the output directory exists
if not os.path.exists(args.output_dir):
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        logging.error('Could not create output directory (\'' + args.output_dir + '\'), exiting...')
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
                    logging.error('Invalid data format extension \'' + extension + '\', exiting...')
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

            # print subset information
            logging.info('------------------------')
            log_indented_line('Subset:', subset)
            logging.info('')

            # derive the bids roots (subject/session and subset) from the full path
            bids_subjsess_root = os.path.commonprefix(glob(os.path.join(os.path.dirname(subset), '*.*')))[:-1]
            bids_subset_root = subset[:subset.rindex('_')]


            #
            # retrieve channel metadata
            #

            # retrieve the channel metadata from the channels.tsv file
            csv = load_channel_info(bids_subset_root + '_channels.tsv')
            if csv is None:
                logging.error('Could not load the channel metadata, exiting...')
                exit(1)

            # sort out the good, the bad and the... non-ieeg
            channels_labels = []
            channels_bad = []
            channels_excluded_by_type = []
            channels_have_status = 'status' in csv.columns
            for index, row in csv.iterrows():
                excluded = False
                if channels_have_status:
                    if row['status'].lower() == 'bad':
                        channels_bad.append(row['name'])
                        excluded = True
                if not row['type'].upper() in config['channels']['types']:
                    channels_excluded_by_type.append(row['name'])
                    excluded = True
                if not excluded:
                    channels_labels.append(row['name'])

            # print channel information
            logging.info(multi_line_list(channels_excluded_by_type, LOGGING_CAPTION_INDENT_LENGTH, 'Channels excluded by type:', 20, ' '))
            logging.info(multi_line_list(channels_bad, LOGGING_CAPTION_INDENT_LENGTH, 'Bad channels (excluded):', 20, ' '))
            logging.info(multi_line_list(channels_labels, LOGGING_CAPTION_INDENT_LENGTH, 'Channels included:', 20, ' ', str(len(channels_labels))))

            # check if there are any channels
            if len(channels_labels) == 0:
                logging.error('No channels were found, exiting...')
                exit(1)
            logging.info('')


            #
            # retrieve events and stimulus-pairs
            #

            # retrieve the stimulation events (onsets and pairs) from the events.tsv file
            csv = load_event_info(bids_subset_root + '_events.tsv', ('trial_type', 'electrical_stimulation_site'))
            if csv is None:
                logging.error('Could not load the stimulation event metadata, exiting...')
                exit(1)

            # acquire the onset and electrode-pair for each stimulation
            trial_onsets = []
            trial_pairs = []
            trials_bad_onsets = []
            trials_have_status = 'status' in csv.columns
            for index, row in csv.iterrows():
                if row['trial_type'].lower() == 'electrical_stimulation':
                    if not is_number(row['onset']) or isnan(float(row['onset'])) or float(row['onset']) < 0:
                        logging.warning('Invalid onset \'' + row['onset'] + '\' in events, should be a numeric value >= 0. Discarding trial...')
                        continue

                    if trials_have_status:
                        if not row['status'].lower() == 'good':
                            trials_bad_onsets.append(row['onset'])
                            continue

                    pair = row['electrical_stimulation_site'].split('-')
                    if not len(pair) == 2 or len(pair[0]) == 0 or len(pair[1]) == 0:
                        logging.error('Electrical stimulation site \'' + row['electrical_stimulation_site'] + '\' invalid, should be two values separated by a dash (e.g. CH01-CH02), exiting...')
                        exit(1)
                    trial_onsets.append(float(row['onset']))
                    trial_pairs.append(pair)
            if len(trials_bad_onsets) > 0:
                log_indented_line('Number of trials marked as bad (excluded):', str(len(trials_bad_onsets)))

            # check if there are trials
            if len(trial_onsets) == 0:
                logging.error('No trials were found, exiting...')
                exit(1)


            # determine the stimulation-pairs conditions (and the trial and electrodes that belong to them)
            # (note that the 'concat_bidirectional_pairs' configuration setting is taken into account here)
            #
            stimpair_labels = []                # for each pair, the labels of the electrodes that were stimulated
            stimpair_trial_indices = []         # for each pair, the indices of the trials that were involved
            stimpair_trial_onsets = []          # for each pair, the indices of the trials that were involved
            stimpair_electrode_indices = []     # for each pair, the indices of the electrodes that were stimulated
            for iChannel0 in range(len(channels_labels)):
                for iChannel1 in range(len(channels_labels)):

                    # retrieve the indices of all the trials that concern this stim-pair
                    indices = []
                    if config['trials']['concat_bidirectional_pairs']:
                        # allow concatenation of bidirectional pairs, pair order does not matter
                        if not iChannel1 < iChannel0:
                            # unique pairs while ignoring pair order
                            indices = [i for i, x in enumerate(trial_pairs) if
                                       (x[0] == channels_labels[iChannel0] and x[1] == channels_labels[iChannel1]) or (x[0] == channels_labels[iChannel1] and x[1] == channels_labels[iChannel0])]

                    else:
                        # do not concatenate bidirectional pairs, pair order matters
                        indices = [i for i, x in enumerate(trial_pairs) if
                                   x[0] == channels_labels[iChannel0] and x[1] == channels_labels[iChannel1]]

                    # add the pair if there are trials for it
                    if len(indices) > 0:
                        stimpair_labels.append(channels_labels[iChannel0] + '-' + channels_labels[iChannel1])
                        stimpair_electrode_indices.append((iChannel0, iChannel1))
                        stimpair_trial_indices.append(indices)
                        stimpair_trial_onsets.append([trial_onsets[i] for i in indices])


            # search the stimulus-pair with too little trials
            stimpair_remove_indices = []
            for iPair in range(len(stimpair_labels)):
                if len(stimpair_trial_indices[iPair]) < config['trials']['minimum_stimpair_trials']:
                    stimpair_remove_indices.append(iPair)
            if len(stimpair_remove_indices) > 0:

                # message
                stimpair_print = [str(stimpair_labels[stimpair_remove_indices[i]]) + ' (' + str(len(stimpair_trial_indices[stimpair_remove_indices[i]])) + ' trials)' for i in range(len(stimpair_remove_indices))]
                stimpair_print = [str_print.ljust(len(max(stimpair_print, key=len)), ' ') for str_print in stimpair_print]
                logging.info(multi_line_list(stimpair_print, LOGGING_CAPTION_INDENT_LENGTH, 'Stim-pairs excluded by number of trials:', 4, '   '))

                # remove those stimulation-pairs
                for index in sorted(stimpair_remove_indices, reverse=True):
                    del stimpair_labels[index]
                    del stimpair_electrode_indices[index]
                    del stimpair_trial_indices[index]
                    del stimpair_trial_onsets[index]

            # display stimulation-pair/trial information
            stimpair_print = [str(stimpair_labels[i]) + ' (' + str(len(stimpair_trial_indices[i])) + ' trials)' for i in range(len(stimpair_labels))]
            stimpair_print = [str_print.ljust(len(max(stimpair_print, key=len)), ' ') for str_print in stimpair_print]
            logging.info(multi_line_list(stimpair_print, LOGGING_CAPTION_INDENT_LENGTH, 'Stimulation pairs included:', 4, '   ', str(len(stimpair_labels))))

            # check if there are stimulus-pairs
            if len(stimpair_labels) == 0:
                logging.error('No stimulus-pairs were found, exiting...')
                exit(1)


            #
            # read and epoch the data
            #

            # read, normalize by median and average the trials within the condition
            # Note: 'load_data_epochs_averages' is used instead of 'load_data_epochs_averages' here because it is more
            #       memory efficient is only the averages are needed
            logging.info('- Reading data...')
            sampling_rate, ccep_average = load_data_epochs_averages(subset, channels_labels, stimpair_trial_onsets,
                                                                    trial_epoch=config['trials']['trial_epoch'],
                                                                    baseline_norm=config['trials']['baseline_norm'],
                                                                    baseline_epoch=config['trials']['baseline_epoch'],
                                                                    out_of_bound_handling=config['trials']['out_of_bounds_handling'])
            if sampling_rate is None or ccep_average is None:
                logging.error('Could not load data (' + subset + '), exiting...')
                exit(1)

            # for each stimulation pair, NaN out the values of the electrodes that were stimulated
            for iPair in range(len(stimpair_labels)):
                ccep_average[stimpair_electrode_indices[iPair][0], iPair, :] = np.nan
                ccep_average[stimpair_electrode_indices[iPair][1], iPair, :] = np.nan

            # determine the sample of stimulus onset (counting from the epoch start)
            onset_sample = int(round(abs(config['trials']['trial_epoch'][0] * sampling_rate)))
            # todo: handle trial epochs which start after the trial onset, currently disallowed by config


            #
            # prepare an output directory
            #

            # make sure a subject directory exists
            output_root = os.path.join(args.output_dir, os.path.basename(os.path.normpath(bids_subset_root)))
            if not os.path.exists(output_root):
                try:
                    os.makedirs(output_root)
                except OSError as e:
                    logging.error("Could not create subset output directory (\'" + output_root + "\'), exiting...")
                    exit(1)

            # intermediate saving of the ccep data as .mat
            sio.savemat(os.path.join(output_root, 'ccep_data.mat'),
                        {'sampling_rate': sampling_rate,
                         'onset_sample': onset_sample,
                         'ccep_average': ccep_average,
                         'stimpair_labels': np.asarray(stimpair_labels, dtype='object'),
                         'channel_labels': np.asarray(channels_labels, dtype='object'),
                         'config': config})

            # write the configuration
            write_config(os.path.join(output_root, 'ccep_config.json'), config)


            #
            # perform the N1 detection
            #

            # detect N1s
            logging.info('- Detecting N1s...')
            n1_peak_indices, n1_peak_amplitudes = ieeg_detect_n1(ccep_average, onset_sample, int(sampling_rate),
                                                                 peak_search_epoch=config['n1_detect']['peak_search_epoch'],
                                                                 n1_search_epoch=config['n1_detect']['n1_search_epoch'],
                                                                 baseline_epoch=config['n1_detect']['n1_baseline_epoch'],
                                                                 baseline_threshold_factor=config['n1_detect']['n1_baseline_threshold_factor'])
            if n1_peak_indices is None or n1_peak_amplitudes is None:
                logging.error('N1 detection failed, exiting...')
                exit(1)

            # intermediate saving of the data and N1 detection as .mat
            sio.savemat(os.path.join(output_root, 'ccep_data.mat'),
                        {'sampling_rate': sampling_rate,
                         'onset_sample': onset_sample,
                         'ccep_average': ccep_average,
                         'stimpair_labels': np.asarray(stimpair_labels, dtype='object'),
                         'channel_labels': np.asarray(channels_labels, dtype='object'),
                         'n1_peak_indices': n1_peak_indices,
                         'n1_peak_amplitudes': n1_peak_amplitudes,
                         'config': config})


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
                # TODO: what if TRIAL_EPOCH_START is after the stimulus onset, currently disallowed by config
                x = np.arange(ccep_average.shape[2])
                x = x / sampling_rate + config['trials']['trial_epoch'][0]

                # determine the range on the x axis where the stimulus was in samples
                # TODO: what if TRIAL_EPOCH_START is after the stimulus onset, currently disallowed by config
                stim_start_x = int(round(abs(config['trials']['trial_epoch'][0] - config['visualization']['blank_stim_epoch'][0]) * sampling_rate)) - 1
                stim_end_x = stim_start_x + int(ceil(abs(config['visualization']['blank_stim_epoch'][1] - config['visualization']['blank_stim_epoch'][0]) * sampling_rate)) - 1

                # calculate the legend x position
                legend_x = config['visualization']['x_axis_epoch'][1] - .13

                # adjust line and font sizes to resolution
                zero_line_thickness = OUTPUT_IMAGE_SIZE / 2000
                signal_line_thickness = OUTPUT_IMAGE_SIZE / 2000
                legend_line_thickness = OUTPUT_IMAGE_SIZE / 500
                title_font_size = round(OUTPUT_IMAGE_SIZE / 80)
                axis_label_font_size = round(OUTPUT_IMAGE_SIZE / 85)
                axis_ticks_font_size = round(OUTPUT_IMAGE_SIZE / 100)
                legend_font_size = round(OUTPUT_IMAGE_SIZE / 90)

                # Adjust the font sizes of the tick according to the number of items (minimum font-size remains 4)
                if len(stimpair_labels) > 36 and axis_ticks_font_size > 4:
                    stimpair_axis_ticks_font_size = 4 + (axis_ticks_font_size - 4) * (36.0 / len(stimpair_labels))
                else:
                    stimpair_axis_ticks_font_size = axis_ticks_font_size

                 
                if len(channels_labels) > 36 and axis_ticks_font_size > 4:
                    electrode_axis_ticks_font_size = 4 + (axis_ticks_font_size - 4) * (36.0 / len(channels_labels))
                else:
                    electrode_axis_ticks_font_size = axis_ticks_font_size

                # account for the situation where there are only a small number of stimulation-pairs.
                if len(stimpair_labels) < 10:
                    stimpair_y_image_height = 500 + (OUTPUT_IMAGE_SIZE - 500) * (len(stimpair_labels) / 10)
                else:
                    stimpair_y_image_height = OUTPUT_IMAGE_SIZE

                # account for a high number of electrodes
                if len(channels_labels) > 50:
                    electrode_y_image_height = 500 + (OUTPUT_IMAGE_SIZE - 500) * (len(channels_labels) / 50)
                else:
                    electrode_y_image_height = OUTPUT_IMAGE_SIZE

                #
                # generate the electrodes plot
                #
                if config['visualization']['generate_electrode_images']:

                    #
                    logging.info('- Generating electrode plots...')

                    # create a progress bar
                    print_progressbar(0, len(channels_labels), prefix='Progress:', suffix='Complete', length=50)

                    # loop through electrodes
                    for iElec in range(len(channels_labels)):

                        # create a figure and retrieve the axis
                        fig = create_figure(OUTPUT_IMAGE_SIZE, stimpair_y_image_height, False)
                        ax = fig.gca()

                        # set the title
                        ax.set_title(channels_labels[iElec] + '\n', fontsize=title_font_size, fontweight='bold')

                        # loop through the stimulation-pairs
                        for iPair in range(len(stimpair_labels)):

                            # draw 0 line
                            y = np.empty((ccep_average.shape[2], 1))
                            y.fill(len(stimpair_labels) - iPair)
                            ax.plot(x, y, linewidth=zero_line_thickness, color=(0.8, 0.8, 0.8))

                            # retrieve the signal
                            y = ccep_average[iElec, iPair, :] / 500
                            y += len(stimpair_labels) - iPair

                            # nan out the stimulation
                            #TODO, only nan if within display range
                            y[stim_start_x:stim_end_x] = np.nan

                            # plot the signal
                            ax.plot(x, y, linewidth=signal_line_thickness)

                            # if N1 is detected, plot it
                            if not isnan(n1_peak_indices[iElec, iPair]):
                                xN1 = n1_peak_indices[iElec, iPair] / sampling_rate + config['trials']['trial_epoch'][0]
                                yN1 = n1_peak_amplitudes[iElec, iPair] / 500
                                yN1 += len(stimpair_labels) - iPair
                                ax.plot(xN1, yN1, 'bo')

                        # set the x axis
                        ax.set_xlabel('\ntime (s)', fontsize=axis_label_font_size)
                        ax.set_xlim(config['visualization']['x_axis_epoch'])
                        for label in ax.get_xticklabels():
                            label.set_fontsize(axis_ticks_font_size)

                        # set the y axis
                        ax.set_ylabel('Stimulated electrode-pair\n', fontsize=axis_label_font_size)
                        ax.set_ylim((0, len(stimpair_labels) + 1))
                        ax.set_yticks(np.arange(1, len(stimpair_labels) + 1, 1))
                        ax.set_yticklabels(np.flip(stimpair_labels), fontsize=stimpair_axis_ticks_font_size)
                        ax.spines['bottom'].set_linewidth(1.5)
                        ax.spines['left'].set_linewidth(1.5)

                        # draw legend
                        legend_y = 2 if len(stimpair_labels) > 4 else (1 if len(stimpair_labels) > 1 else 0)
                        ax.plot([legend_x, legend_x], [legend_y + .05, legend_y + .95], linewidth=legend_line_thickness, color=(0, 0, 0))
                        ax.text(legend_x + .01, legend_y + .3, '500 \u03bcV', fontsize=legend_font_size)

                        # Hide the right and top spines
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)

                        # save figure
                        fig.savefig(os.path.join(output_root, 'electrode_' + str(channels_labels[iElec]) + '.png'), bbox_inches='tight')

                        # update progress bar
                        print_progressbar(iElec + 1, len(channels_labels), prefix='Progress:', suffix='Complete', length=50)

                #
                # generate the stimulation-pair plots
                #
                if config['visualization']['generate_stimpair_images']:

                    #
                    logging.info('- Generating stimulation-pair plots...')

                    # create a progress bar
                    print_progressbar(0, len(stimpair_labels), prefix='Progress:', suffix='Complete', length=50)

                    # loop through the stimulation-pairs
                    for iPair in range(len(stimpair_labels)):

                        # create a figure and retrieve the axis
                        fig = create_figure(OUTPUT_IMAGE_SIZE, electrode_y_image_height, False)
                        ax = fig.gca()

                        # set the title
                        ax.set_title(stimpair_labels[iPair] + '\n', fontsize=title_font_size, fontweight='bold')

                        # loop through the electrodes
                        for iElec in range(len(channels_labels)):

                            # draw 0 line
                            y = np.empty((ccep_average.shape[2], 1))
                            y.fill(len(channels_labels) - iElec)
                            ax.plot(x, y, linewidth=zero_line_thickness, color=(0.8, 0.8, 0.8))

                            # retrieve the signal
                            y = ccep_average[iElec, iPair, :] / 500
                            y += len(channels_labels) - iElec

                            # nan out the stimulation
                            #TODO, only nan if within display range
                            y[stim_start_x:stim_end_x] = np.nan

                            # plot the signal
                            ax.plot(x, y, linewidth=signal_line_thickness)

                            # if N1 is detected, plot it
                            if not isnan(n1_peak_indices[iElec, iPair]):
                                xN1 = n1_peak_indices[iElec, iPair] / sampling_rate + config['trials']['trial_epoch'][0]
                                yN1 = n1_peak_amplitudes[iElec, iPair] / 500
                                yN1 += len(channels_labels) - iElec
                                ax.plot(xN1, yN1, 'bo')

                        # set the x axis
                        ax.set_xlabel('\ntime (s)', fontsize=axis_label_font_size)
                        ax.set_xlim(config['visualization']['x_axis_epoch'])
                        for label in ax.get_xticklabels():
                            label.set_fontsize(axis_ticks_font_size)

                        # set the y axis
                        ax.set_ylabel('Measured electrodes\n', fontsize=axis_label_font_size)
                        ax.set_ylim((0, len(channels_labels) + 1))
                        ax.set_yticks(np.arange(1, len(channels_labels) + 1, 1))
                        ax.set_yticklabels(np.flip(channels_labels), fontsize=electrode_axis_ticks_font_size)
                        ax.spines['bottom'].set_linewidth(1.5)
                        ax.spines['left'].set_linewidth(1.5)

                        # draw legend
                        legend_y = 2 if len(stimpair_labels) > 4 else (1 if len(stimpair_labels) > 1 else 0)
                        ax.plot([legend_x, legend_x], [legend_y + .05, legend_y + .95], linewidth=legend_line_thickness, color=(0, 0, 0))
                        ax.text(legend_x + .01, legend_y + .3, '500 \u03bcV', fontsize=legend_font_size)

                        # Hide the right and top spines
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)

                        # save figure
                        fig.savefig(os.path.join(output_root, 'stimpair_' + str(stimpair_labels[iPair]) + '.png'), bbox_inches='tight')

                        # update progress bar
                        print_progressbar(iPair + 1, len(stimpair_labels), prefix='Progress:', suffix='Complete', length=50)

                #
                # generate the matrices
                #
                if config['visualization']['generate_matrix_images']:

                    #
                    logging.info('- Generating matrices...')

                    # calculate the image width based on the number of stim-pair and electrodes
                    image_width = stimpair_y_image_height / len(stimpair_labels) * len(channels_labels)
                    image_width += 800

                    # make sure the image width does not exceed the matplotlib limit of 2**16
                    if image_width >= 2 ** 16:
                        factor = (2 ** 16 - 50) / image_width
                        image_width = int(round(image_width * factor))
                        image_height = int(round(stimpair_y_image_height * factor))
                    else:
                        image_height = stimpair_y_image_height

                    # adjust the padding between the matrix and the colorbar based on the image width
                    colorbar_padding = 0.01 if image_width < 2000 else (0.01 * (2000 / image_width))

                    # if there are 10 times more electrodes than stimulation-pairs, then allow
                    # the matrix to squeeze horizontally
                    matrix_aspect = 1
                    element_ratio = len(channels_labels) / len(stimpair_labels)
                    if element_ratio > 10:
                        matrix_aspect = element_ratio / 8


                    #
                    # Amplitude matrix
                    #
                    
                    #
                    matrix_amplitudes = n1_peak_amplitudes
                    #matrix_amplitudes[np.isnan(matrix_amplitudes)] = 0
                    matrix_amplitudes *= -1

                    # create a figure and retrieve the axis
                    fig = create_figure(image_width, image_height, False)
                    ax = fig.gca()

                    # create a color map
                    cmap = cm.get_cmap("viridis").copy()
                    cmap.set_bad((.7, .7, .7, 1))

                    # draw the matrix
                    im = ax.imshow(np.transpose(matrix_amplitudes), origin='upper', vmin=0, vmax=500, cmap=cmap, aspect=matrix_aspect)

                    # set labels and ticks
                    ax.set_yticks(np.arange(0, len(stimpair_labels), 1))
                    ax.set_yticklabels(stimpair_labels, fontsize=stimpair_axis_ticks_font_size)
                    ax.set_xticks(np.arange(0, len(channels_labels), 1))
                    ax.set_xticklabels(channels_labels,
                                       rotation=90,
                                       fontsize=stimpair_axis_ticks_font_size)  # deliberately using stimpair-fs here
                    ax.set_xlabel('\nMeasured electrode', fontsize=axis_label_font_size)
                    ax.set_ylabel('Stimulated electrode-pair\n', fontsize=axis_label_font_size)
                    for axis in ['top', 'bottom', 'left', 'right']:
                        ax.spines[axis].set_linewidth(1.5)

                    # set a color-bar
                    cbar = fig.colorbar(im, pad=colorbar_padding)
                    cbar.set_ticks([0, 100, 200, 300, 400, 500])
                    cbar.ax.set_yticklabels(['0', '-100 \u03bcV', '-200 \u03bcV', '-300 \u03bcV', '-400 \u03bcV', '-500 \u03bcV'], fontsize=legend_font_size - 4)
                    cbar.outline.set_linewidth(1.5)

                    # save figure
                    fig.savefig(os.path.join(output_root, 'matrix_amplitude.png'), bbox_inches='tight')


                    #
                    # Latency matrix
                    #

                    # create a figure and retrieve the axis
                    fig = create_figure(image_width, image_height, False)
                    ax = fig.gca()

                    # retrieve the latencies and convert the indices (in samples) to time units (ms)
                    matrix_latencies = n1_peak_indices
                    matrix_latencies -= onset_sample
                    matrix_latencies /= sampling_rate
                    matrix_latencies *= 1000
                    #matrix_latencies[np.isnan(matrix_latencies)] = 0

                    # determine the latest
                    latest_N1 = np.nanmax(n1_peak_indices)
                    if np.isnan(latest_N1):
                        latest_N1 = 10
                    latest_N1 = int(ceil(latest_N1 / 10)) * 10

                    # create a color map
                    cmap = cm.get_cmap('viridis').copy()
                    cmap.set_bad((.7, .7, .7, 1))

                    # draw the matrix
                    im = ax.imshow(np.transpose(matrix_latencies), origin='upper', vmin=0, cmap=cmap, aspect=matrix_aspect)

                    # set labels and ticks
                    ax.set_yticks(np.arange(0, len(stimpair_labels), 1))
                    ax.set_yticklabels(stimpair_labels, fontsize=stimpair_axis_ticks_font_size)
                    ax.set_xticks(np.arange(0, len(channels_labels), 1))
                    ax.set_xticklabels(channels_labels,
                                       rotation=90,
                                       fontsize=stimpair_axis_ticks_font_size)  # deliberately using stimpair-fs here
                    ax.set_xlabel('\nMeasured electrode', fontsize=axis_label_font_size)
                    ax.set_ylabel('Stimulated electrode-pair\n', fontsize=axis_label_font_size)
                    for axis in ['top', 'bottom', 'left', 'right']:
                        ax.spines[axis].set_linewidth(1.5)

                    # generate the legend tick values
                    legend_tick_values = []
                    legend_tick_labels = []
                    for latency in range(0, latest_N1 + 10, 10):
                        legend_tick_values.append(latency)
                        legend_tick_labels.append(str(latency) + ' ms')

                    # set a color-bar
                    cbar = fig.colorbar(im, pad=colorbar_padding)
                    im.set_clim([legend_tick_values[0], legend_tick_values[-1]])
                    cbar.set_ticks(legend_tick_values)
                    cbar.ax.set_yticklabels(legend_tick_labels, fontsize=legend_font_size - 4)
                    cbar.outline.set_linewidth(1.5)

                    # save figure
                    fig.savefig(os.path.join(output_root, 'matrix_latency.png'), bbox_inches='tight')


            #
            logging.info('- Finished subset')

    else:
        #
        logging.warning('Participant \'' + subject + '\' could not be found, skipping')


logging.info('- Finished running')
