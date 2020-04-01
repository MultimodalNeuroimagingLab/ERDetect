#!/usr/bin/env python3

# Early response detection - docker entry-point
# =====================================================
# Entry point script for the automatic detection of early responses (N1) in CCEP data.
#
#
# Copyright 2020, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
import argparse
import os
import csv
from math import isnan, ceil, floor
from glob import glob

from bids_validator import BIDSValidator
from mne.io import read_raw_edf, read_raw_brainvision
from pymef.mef_session import MefSession
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


# D:\BIDS_test D:\output --participant_label sub-MSEL01877 --subset_search_pattern task-ccep_run-01 --format_extension .mefd

#
# constants
#
VALID_FORMAT_EXTENSIONS         = ('.edf', '.vhdr', '.vmrk', '.eeg', '.mefd')   # valid data format to search for (European Data Format, BrainVision and MEF3)
TRIAL_EPOCH_START               = -1.0                                          # the timepoint (in seconds) relative to the stimulus onset that will be considered as the start of the trial epoch
TRIAL_EPOCH_END                 = 3.0                                           # the timepoint (in seconds) relative to the stimulus onset that will be considered as the end of the trial epoch
BASELINE_EPOCH_START            = -1.0                                          # the timepoint (in seconds) relative to the stimulus onset that will be considered as the start of the baseline epoch within each trial
BASELINE_EPOCH_END              = -0.1                                          # the timepoint (in seconds) relative to the stimulus onset that will be considered as the end of the baseline epoch within each trial
DISPLAY_X_RANGE                 = (-0.2, 1)                                     # the range for the x-axis in display, (in seconds) relative to the stimulus onset that will be used as the range
DISPLAY_STIM_RANGE              = (-0.015, 0.0025)                               # the range
SUBPLOT_LAYOUT_RATIO            = (4, 3)
OUTPUT_IMAGE_RESOLUTION         = (2000, 2000)                                   # ; it is advisable to koop the resolution ratio in line with the SUBPLOT_LAYOUT_RATIO

#
# version and helper functions
#
__version__ = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'version')).read()

def isNumber(value):
    try:
        float(value)
        return True
    except:
        return False


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
parser.add_argument('--no_concat_bidirectional_pairs',
                    help='Do not concatenate electrode pairs that were stimulated in both directions (e.g. CH01-CH02 and CH02-CH01)',
                    action='store_true')
parser.add_argument('--skip_bids_validator',
                    help='Whether or not to perform BIDS dataset validation',
                    action='store_true')
parser.add_argument('-v', '--version',
                    action='version',
                    version='N1Detection BIDS-App version {}'.format(__version__))
args = parser.parse_args()


# debug, print
print('BIDS input dataset:     ' + args.bids_dir)
print('Output location:        ' + args.output_dir)

#
# check if the input is a valid BIDS dataset
#
#if not args.skip_bids_validator:
#    if not BIDSValidator().is_bids(args.bids_dir):
#        print('Error: BIDS input dataset did not pass BIDS validator. Datasets can be validated online '
#                          'using the BIDS Validator (http://incf.github.io/bids-validator/')
#        exit()


#
# list the subject to analyze (either based on the input parameter or list all in the BIDS_dir)
#
subjects_to_analyze = []
if args.participant_label:

    # user-specified subjects
    subjects_to_analyze = args.participant_label

else:

    # all subjects
    subject_dirs = glob(os.path.join(args.bids_dir, 'sub-*'))
    subjects_to_analyze = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]


#
# loop through the participants and list the subsets
#
for subject_label in subjects_to_analyze:

    # see if the subject is exists (in case the user specified the labels)
    if os.path.isdir(os.path.join(args.bids_dir, subject_label)):

        # retrieve the subset search patterns
        subset_patterns = args.subset_search_pattern if args.subset_search_pattern else ('',)

        # retrieve the data formats to include
        if args.format_extension:
            extensions = args.format_extension
            for extension in extensions:
                if not any(extension in x for x in VALID_FORMAT_EXTENSIONS):
                    print('Error: invalid data format extension \'' + extension + '\'')
                    exit()
        else:
            extensions = VALID_FORMAT_EXTENSIONS

        # build path patterns for the search of subsets
        subsets = []
        modalities = ('*eeg',)                    # ieeg and eeg
        for extension in extensions:
            for modality in modalities:
                for subset_pattern in subset_patterns:
                    subsets += glob(os.path.join(args.bids_dir, subject_label, modality, '*' + subset_pattern + '*' + extension)) + \
                               glob(os.path.join(args.bids_dir, subject_label, '*', modality, '*' + subset_pattern + '*' + extension))

        # bring subsets with multiple formats down to one format (prioritized to occurrence in the extension var)
        for subset in subsets:
            subset_name = subset[:subset.rindex(".")]
            for subset_other in reversed(subsets):
                if not subset == subset_other:
                    subset_other_name = subset_other[:subset_other.rindex(".")]
                    if subset_name == subset_other_name:
                        subsets.remove(subset_other)

        # loop through the subsets for analysis
        for subset in subsets:

            # message subset start
            print('------')
            print('Subset:                                          ' + subset)
            print('Trial epoch window:                              ' + str(TRIAL_EPOCH_START) + 's < stim onset < ' + str(TRIAL_EPOCH_END) + 's  (window size ' + str(abs(TRIAL_EPOCH_END - TRIAL_EPOCH_START)) + 's)')
            print('Baseline window with each trial:                 ' + str(BASELINE_EPOCH_START) + 's : ' + str(BASELINE_EPOCH_END) + 's  (window size ' + str(abs(BASELINE_EPOCH_END - BASELINE_EPOCH_START)) + 's)')
            print('Concatenate bidirectional stimulated pairs:      ' + ('No' if args.no_concat_bidirectional_pairs else 'Yes'))


            #
            # gather metadata information
            #

            # derive the bids roots (subject/session and subset) from the full path
            bids_subjsess_root = os.path.commonprefix(glob(os.path.join(os.path.dirname(subset), '*.*')))[:-1];
            bids_subset_root = subset[:subset.rindex('_')]

            # retrieve the channel metadata from the channels.tsv file
            with open(bids_subset_root + '_channels.tsv') as csv_file:
                reader = csv.DictReader(csv_file, delimiter='\t')

                # make sure the required columns exist
                channels_include = []
                channels_bad = []
                channels_non_ieeg = []
                if not 'name' in reader.fieldnames:
                    print('Error: could not find the \'name\' column in \'' + bids_subjsess_root + '_channels.tsv\'')
                    exit()
                if not 'type' in reader.fieldnames:
                    print('Error: could not find the \'type\' column in \'' + bids_subjsess_root + '_channels.tsv\'')
                    exit()
                if not 'status' in reader.fieldnames:
                    print('Error: could not find the \'status\' column in \'' + bids_subjsess_root + '_channels.tsv\'')
                    exit()

                # sort out the good, the bad and the... non-ieeg
                for row in reader:
                    excluded = False
                    if row['status'].lower() == 'bad':
                        channels_bad.append(row['name'])
                        #excluded = True
                    if not row['type'].upper() in ('ECOG', 'SEEG'):
                        channels_non_ieeg.append(row['name'])
                        excluded = True
                    if not excluded:
                        channels_include.append(row['name'])

            # print channel information
            print('Bad channels:                                    ' + ' '.join(channels_bad))
            print('Non-IEEG channels:                               ' + ' '.join(channels_non_ieeg))
            print('Included channels:                               ' + ' '.join(channels_include))

            # check if there are channels
            if len(channels_include) == 0:
                print('Error: no channels were found')
                exit()

            # retrieve the electrical stimulation trials (onsets and pairs) from the events.tsv file
            with open(bids_subset_root + '_events.tsv') as csv_file:
                reader = csv.DictReader(csv_file, delimiter='\t')

                # make sure the required columns exist
                if not 'onset' in reader.fieldnames:
                    print('Error: could not find the \'onset\' column in \'' + bids_subset_root + '_events.tsv\'')
                    exit()
                if not 'trial_type' in reader.fieldnames:
                    print('Error: could not find the \'trial_type\' column in \'' + bids_subset_root + '_events.tsv\'')
                    exit()
                if not 'electrical_stimulation_site' in reader.fieldnames:
                    print('Error: could not find the \'electrical_stimulation_site\' column in \'' + bids_subset_root + '_events.tsv\'')
                    exit()

                # acquire the onset and electrode-pair for each stimulation
                trials_onset = []
                trials_pair = []
                for row in reader:
                    if row['trial_type'].lower() == 'electrical_stimulation':
                        if not isNumber(row['onset']) or isnan(float(row['onset'])):
                            print('Error: invalid onset \'' + row['onset'] + '\' in events, should be a numeric value')
                            #exit()
                            continue

                        pair = row['electrical_stimulation_site'].split('-')
                        if not len(pair) == 2 or len(pair[0]) == 0 or len(pair[1]) == 0:
                            print('Error: electrical stimulation site \'' + row['electrical_stimulation_site'] + '\' invalid, should be two values seperated by a dash (e.g. CH01-CH02)')
                            exit()
                        trials_onset.append(float(row['onset']))
                        trials_pair.append(pair)

            # check if there are trials
            if len(trials_onset) == 0:
                print('Error: no trials were found')
                exit()

            # debug, limit channels
            #channels_include = channels_include[0:6]


            #
            # read the data to a numpy array
            #

            #TODO: handle different units in types

            # read the dataset
            extension = subset[subset.rindex("."):]
            if extension == '.edf':


                # Alternative (use pyedflib), low memory usage solution since it has the ability to read per channel
                #from pyedflib import EdfReader
                #f = EdfReader(subset)
                #n = f.signals_in_file
                #signal_labels = f.getSignalLabels()
                #srate = f.getSampleFrequencies()[0]
                #size_time_t = POSTSTIM_EPOCH + PRESTIM_EPOCH
                #size_time_s = int(ceil(size_time_t * srate))
                #data = np.empty((len(trials_onset), len(channels_include), size_time_s))
                #data.fill(np.nan)
                #for iChannel in range(len(channels_include)):
                #    channel_index = signal_labels.index(channels_include[iChannel])
                #    signal = f.readSignal(channel_index)
                #    for iTrial in range(len(trials_onset)):
                #        sample_start = int(round(trials_onset[iTrial] * srate))
                #        data[iTrial, iChannel, :] = signal[sample_start:sample_start + size_time_s]


                # read the edf data
                edf = read_raw_edf(subset, eog=None, misc=None, stim_channel=[], exclude=channels_non_ieeg, preload=True, verbose=None)

                # retrieve the sample-rate
                srate = edf.info['sfreq']

                # calculate the size of the time dimension
                size_time_t = abs(TRIAL_EPOCH_END - TRIAL_EPOCH_START)
                size_time_s = int(ceil(size_time_t * srate))

                # initialize a data buffer (trials/epochs x channel x time)
                # Note: this order makes the time dimension contiguous in memory, which is handy for block copies
                data = np.empty((len(trials_onset), len(channels_include), size_time_s))
                data.fill(np.nan)

                # loop through the included channels
                for iChannel in range(len(channels_include)):

                    # (try to) retrieve the index of the channel
                    try:
                        channel_index = edf.ch_names.index(channels_include[iChannel])

                        # loop through the trials
                        for iTrial in range(len(trials_onset)):
                            sample_start = int(round((trials_onset[iTrial] + TRIAL_EPOCH_START) * srate))
                            data[iTrial, iChannel, :] = edf[channel_index, sample_start:sample_start + size_time_s][0]

                    except ValueError:
                        print('Error: could not find channel \'' + channels_include[iChannel] + '\' in dataset')
                        exit()

                edf.close()
                del edf

            elif extension == '.vhdr' or extension == '.vmrk' or extension == '.eeg':

                # read the BrainVision data
                bv = read_raw_brainvision(subset[:subset.rindex(".")] + '.vhdr', preload=True)

                # retrieve the sample-rate
                srate = round(bv.info['sfreq'])

                # calculate the size of the time dimension
                size_time_t = abs(TRIAL_EPOCH_END - TRIAL_EPOCH_START)
                size_time_s = int(ceil(size_time_t * srate))

                # initialize a data buffer (trials/epochs x channel x time)
                # Note: this order makes the time dimension contiguous in memory, which is handy for block copies
                data = np.empty((len(trials_onset), len(channels_include), size_time_s))
                data.fill(np.nan)

                # loop through the included channels
                for iChannel in range(len(channels_include)):

                    # (try to) retrieve the index of the channel
                    try:
                        channel_index = bv.info['ch_names'].index(channels_include[iChannel])

                        # loop through the trials
                        for iTrial in range(len(trials_onset)):
                            sample_start = int(round((trials_onset[iTrial] + TRIAL_EPOCH_START) * srate))
                            data[iTrial, iChannel, :] = bv[channel_index, sample_start:sample_start + size_time_s][0]

                    except ValueError:
                        print('Error: could not find channel \'' + channels_include[iChannel] + '\' in dataset')
                        exit()

            elif extension == '.mefd':

                # read the session metadata
                mef = MefSession(subset, '', read_metadata=True)

                # retrieve the sample-rate
                srate = mef.session_md['time_series_metadata']['section_2']['sampling_frequency'].item(0)

                # calculate the size of the time dimension
                size_time_t = abs(TRIAL_EPOCH_END - TRIAL_EPOCH_START)
                size_time_s = int(ceil(size_time_t * srate))

                # initialize a data buffer (trials/epochs x channel x time)
                # Note: this order makes the time dimension contiguous in memory, which is handy for block copies
                data = np.empty((len(trials_onset), len(channels_include), size_time_s))
                data.fill(np.nan)

                # loop through the trials
                for iTrial in range(len(trials_onset)):
                    sample_start = int(round((trials_onset[iTrial] + TRIAL_EPOCH_START) * srate))
                    sample_end = sample_start + size_time_s

                    #TODO: check if sample_start and sample_end are within range

                    # load the trial data
                    trial_data = mef.read_ts_channels_sample(channels_include, [sample_start, sample_end])

                    # loop through the channels
                    for iChannel in range(len(channels_include)):
                        data[iTrial, iChannel, :] = trial_data[iChannel]

            #
            #TODO: check for invalid data (trial, channel etc; could happen when onset is wrong etc.)


            #
            # baseline substract
            #

            #TODO: check if the baseline (epoch) window is within the trial epoch

            # determine the start and end sample of the baseline epoch
            baseline_start = int(round(abs(TRIAL_EPOCH_START - BASELINE_EPOCH_START) * srate))
            baseline_end = baseline_start + int(ceil(abs(BASELINE_EPOCH_END - BASELINE_EPOCH_START) * srate)) - 1

            # subtract the baseline median per trial
            for iTrial in range(len(trials_onset)):
                for iChannel in range(len(channels_include)):
                    data[iTrial, iChannel, :] = data[iTrial, iChannel, :] - np.nanmedian(data[iTrial, iChannel, baseline_start:baseline_end])
                    #data[iTrial, iChannel, :] = data[iTrial, iChannel, :] - np.nanmean(data[iTrial, iChannel, baseline_start:baseline_end])


            #
            # retrieve the stimulation-pairs and for each pair take the average over trials
            # (note that the 'no_concat_bidirectional_pairs' argument is taken into account here)
            #

            # variable to store the stimulation pair information in
            pairs_label = []                # for each pair, the label
            pairs_trialsIdx = []            # for each pair, the indices of the trials that were involved
            pairs_stim_electrodesIdx = []   # for each pair, the indices of the electrodes that were stimulated

            # loop through al possible channel/electrode combinations
            for iChannel0 in range(len(channels_include)):
                for iChannel1 in range(len(channels_include)):

                    # retrieve the indices of all the trials that concern this stim-pair
                    indices = []
                    if args.no_concat_bidirectional_pairs:
                        # do not concatenate bidirectional pairs, pair order matters
                        indices = [i for i, x in enumerate(trials_pair) if
                                   x[0] == channels_include[iChannel0] and x[1] == channels_include[iChannel1]]

                    else:
                        # allow concatenation of bidirectional pairs, pair order does not matter
                        if not iChannel1 < iChannel0:
                            # unique pairs while ignoring pair order
                            indices = [i for i, x in enumerate(trials_pair) if
                                       (x[0] == channels_include[iChannel0] and x[1] == channels_include[iChannel1]) or (x[0] == channels_include[iChannel1] and x[1] == channels_include[iChannel0])]

                    # add the pair if there are trials for it
                    if len(indices) > 0:
                        pairs_label.append(channels_include[iChannel0] + '-' + channels_include[iChannel1])
                        pairs_stim_electrodesIdx.append((iChannel0, iChannel1))
                        pairs_trialsIdx.append(indices)


            # display Pair/Trial information
            print('Stimulation pairs:                               ' + str(len(pairs_label)))
            for iPair in range(len(pairs_label)):
                if iPair % 10 == 0:
                    print('                                                 ', end='');
                print(str(pairs_label[iPair]) + ' (' + str(len(pairs_trialsIdx[iPair])) + ' trials)   ', end='')
                if iPair > 0 and ((iPair + 1) % 10 == 0):
                    print('')

            # create a variable to store each stimulation-pair average in (stim-pair x channel x time)
            pairs_average = np.empty((len(pairs_label), len(channels_include), size_time_s))
            pairs_average.fill(np.nan)

            # for each stimulation-pair, calculate the average over trials
            for iPair in range(len(pairs_label)):
                pairs_average[iPair, :, :] = np.nanmean(data[pairs_trialsIdx[iPair], :, :], axis=0)



            # for each stimulation pair, NaN out the values of the electrodes that were stimulated
            for iPair in range(len(pairs_label)):
                pairs_average[iPair, pairs_stim_electrodesIdx[iPair][0], :] = np.nan
                pairs_average[iPair, pairs_stim_electrodesIdx[iPair][1], :] = np.nan


            # calculate average signal per electrode over all pairs
            #pairs_electrode_average = np.nanmean(pairs_average, axis=0)

            # calculate average signal per pair over all electrodes
            #pairs_pairs_average = np.nanmean(pairs_average, axis=1)

            #
            # intermediate saving of the data as .mat
            #
            sio.savemat(os.path.join(args.output_dir, 'average_ccep.mat'), {'average_ccep': pairs_average})


            #
            # perform the detection
            #

            #TODO: N1 detection


            #
            # prepare some settings for plotting
            #

            # generate the x-axis values
            x = np.arange(size_time_s) + 1
            x = x / srate + TRIAL_EPOCH_START

            # determine the range on the x axis where the stimulus was in samples
            x_stim_start = int(round(abs(TRIAL_EPOCH_START - DISPLAY_STIM_RANGE[0]) * srate)) - 1
            x_stim_end = x_stim_start + int(ceil(abs(DISPLAY_STIM_RANGE[1] - DISPLAY_STIM_RANGE[0]) * srate)) - 1

            # calculate the legend x position
            x_legend = DISPLAY_X_RANGE[1] - .13

            # adjust line and font sizes to resolution
            zero_line_thickness = OUTPUT_IMAGE_RESOLUTION[0] / 2000
            signal_line_thickness = OUTPUT_IMAGE_RESOLUTION[0] / 2000
            legend_line_thickness = OUTPUT_IMAGE_RESOLUTION[0] / 500
            title_font_size = round(OUTPUT_IMAGE_RESOLUTION[0] / 80)
            axis_label_font_size = round(OUTPUT_IMAGE_RESOLUTION[0] / 85)
            axis_texts_font_size = round(OUTPUT_IMAGE_RESOLUTION[0] / 100)
            legend_font_size = round(OUTPUT_IMAGE_RESOLUTION[0] / 90)


            #
            # generate the electrodes plot
            #

            # loop through electrodes
            for iElec in range(len(channels_include)):

                # create a figure and resize the figure to the image output resolution
                from matplotlib.figure import Figure
                fig = Figure()
                #fig = plt.figure()
                DPI = fig.get_dpi()
                fig.set_size_inches(float(OUTPUT_IMAGE_RESOLUTION[0]) / float(DPI), float(OUTPUT_IMAGE_RESOLUTION[1]) / float(DPI))

                # retrieve the figure it's axis
                ax = fig.gca()

                # set the title
                ax.set_title(channels_include[iElec], fontSize=title_font_size, fontweight='bold')

                # loop through the stimulation-pairs
                for iPair in range(len(pairs_label)):

                    # draw 0 line
                    y = np.empty((size_time_s, 1))
                    y.fill(iPair + 1)
                    ax.plot(x, y, linewidth=zero_line_thickness, color=(0.8, 0.8, 0.8))

                    # retrieve the signal
                    y = pairs_average[iPair, iElec, :]
                    y = y / 500
                    y = y + iPair + 1

                    # nan out the stimulation
                    y[x_stim_start:x_stim_end] = np.nan

                    # plot the signal
                    ax.plot(x, y, linewidth=signal_line_thickness)

                ax.set_xlabel('time (s)', fontSize=axis_label_font_size)
                ax.set_xlim(DISPLAY_X_RANGE)
                for label in ax.get_xticklabels():
                    label.set_fontsize(axis_texts_font_size)

                # set the y axis
                ax.set_ylabel('Stimulated electrode', fontSize=axis_label_font_size)
                ax.set_ylim((0, len(pairs_label) + 2))
                ax.set_yticks(np.arange(1, len(pairs_label) + 1, 1))
                ax.set_yticklabels(pairs_label, fontSize=axis_texts_font_size)

                # draw legend
                ax.plot([x_legend, x_legend], [2.1, 2.9], linewidth=legend_line_thickness, color=(0, 0, 0))
                ax.text(x_legend + .01, 2.3, '500 \u03bcV', fontSize=legend_font_size)

                # Hide the right and top spines
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                # display/save figure
                # fig.show()
                #fig.savefig(os.path.join(args.output_dir, 'electrode_' + str(channels_include[iElec]) + '.png'))
                fig.savefig(os.path.join(args.output_dir, 'electrode_' + str(channels_include[iElec]) + '.png'), bbox_inches='tight')


            #
            # generate the pairs plot
            #

            # loop through the stimulation-pairs
            for iPair in range(len(pairs_label)):

                # create a figure and resize the figure to the image output resolution
                from matplotlib.figure import Figure
                fig = Figure()
                #fig = plt.figure()
                DPI = fig.get_dpi()
                fig.set_size_inches(float(OUTPUT_IMAGE_RESOLUTION[0]) / float(DPI), float(OUTPUT_IMAGE_RESOLUTION[1]) / float(DPI))

                # retrieve the figure it's axis
                ax = fig.gca()

                # set the title
                ax.set_title(pairs_label[iPair], fontSize=title_font_size, fontweight='bold')

                # loop through the electrodes
                for iElec in range(len(channels_include)):

                    # draw 0 line
                    y = np.empty((size_time_s, 1))
                    y.fill(iElec + 1)
                    ax.plot(x, y, linewidth=zero_line_thickness, color=(0.8, 0.8, 0.8))

                    # retrieve the signal
                    y = pairs_average[iPair, iElec, :]
                    y = y / 500
                    y = y + iElec + 1

                    # nan out the stimulation
                    y[x_stim_start:x_stim_end] = np.nan

                    # plot the signal
                    ax.plot(x, y, linewidth=signal_line_thickness)

                ax.set_xlabel('time (s)', fontSize=axis_label_font_size)
                ax.set_xlim(DISPLAY_X_RANGE)
                for label in ax.get_xticklabels():
                    label.set_fontsize(axis_texts_font_size)

                # set the y axis
                ax.set_ylabel('Measured electrodes', fontSize=axis_label_font_size)
                ax.set_ylim((0, len(channels_include) + 2))
                ax.set_yticks(np.arange(1, len(channels_include) + 1, 1))
                ax.set_yticklabels(channels_include, fontSize=axis_texts_font_size)

                # draw legend
                ax.plot([x_legend, x_legend], [2.1, 2.9], linewidth=legend_line_thickness, color=(0, 0, 0))
                ax.text(x_legend + .01, 2.3, '500 \u03bcV', fontSize=legend_font_size)

                # Hide the right and top spines
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                # display/save figure
                # fig.show()
                #fig.savefig(os.path.join(args.output_dir, 'condition_' + str(pairs_label[iPair]) + '.png'))
                fig.savefig(os.path.join(args.output_dir, 'condition_' + str(pairs_label[iPair]) + '.png'), bbox_inches='tight')

    else:
        #
        print('Warning: participant \'' + subject_label + '\' could not be found, skipping')


