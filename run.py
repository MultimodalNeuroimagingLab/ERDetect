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
import matplotlib.pyplot as plt


#
# constants
#
VALID_FORMAT_EXTENSIONS         = ('.edf', '.vhdr', '.vmrk', '.eeg', '.mefd')   # valid data format to search for (European Data Format, BrainVision and MEF3)
PRESTIM_EPOCH                   = 2.5                                           # the amount of time (in seconds) before the stimulus that will be considered as start of the epoch (for each trial)
POSTSTIM_EPOCH                  = 2.5                                           # the amount of time (in seconds) before the stimulus that will be considered as end of the epoch (for each trial)
SUBPLOT_LAYOUT_RATIO            = (4, 3)
OUTPUT_IMAGE_RESOLUTION         = (1024, 768)                                   # ; it is advisable to koop the resolution ratio in line with the SUBPLOT_LAYOUT_RATIO

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

# determine number of rows and columns for a plot given a specific n and a desired layout ratio
def getPlotLayout(layout, n):
    dirHor = layout[0] >= layout[1]
    ratio = (layout[1] / layout[0]) if dirHor else (layout[0] / layout[1])
    for ncols in range(1, n):
        nrows = ncols * ratio
        if not int(ncols * ratio) == nrows:
            nrows = floor(nrows) if n <= ncols * floor(nrows) else ceil(nrows)
        else:
            nrows = int(nrows)
        if n <= ncols * nrows:
            break
    return (ncols, nrows) if dirHor else (nrows, ncols)



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
            print('Epoch window:                                    -' + str(PRESTIM_EPOCH) + 's < stim onset < ' + str(POSTSTIM_EPOCH) + 's  (window size ' + str(POSTSTIM_EPOCH + PRESTIM_EPOCH) + 's)')
            print('Concatenate bidirectional stimulated pairs:      ' + ('No' if args.no_concat_bidirectional_pairs else 'Yes'))


            #
            # gather metadata information
            #

            # derive the bids roots (subject/session and subset) from the full path
            bids_subjsess_root = os.path.commonprefix(glob(os.path.join(os.path.dirname(subset), '*.*')))[:-1];
            bids_subset_root = subset[:subset.rindex('_')]

            # retrieve the channel metadata from the channels.tsv file
            with open(bids_subjsess_root + '_channels.tsv') as csv_file:
                reader = csv.DictReader(csv_file, delimiter='\t')

                # make sure the required columns exist
                channels_include = [];
                channels_bad = [];
                channels_non_ieeg = [];
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
                trials_onset = [];
                trials_pair = [];
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
            #channels_include = channels_include[0:4]


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
                size_time_t = POSTSTIM_EPOCH + PRESTIM_EPOCH
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
                            sample_start = int(round(trials_onset[iTrial] * srate))
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
                size_time_t = POSTSTIM_EPOCH + PRESTIM_EPOCH
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
                            sample_start = int(round(trials_onset[iTrial] * srate))
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
                size_time_t = POSTSTIM_EPOCH + PRESTIM_EPOCH
                size_time_s = int(ceil(size_time_t * srate))

                # initialize a data buffer (trials/epochs x channel x time)
                # Note: this order makes the time dimension contiguous in memory, which is handy for block copies
                data = np.empty((len(trials_onset), len(channels_include), size_time_s))
                data.fill(np.nan)

                # loop through the included channels
                for iChannel in range(len(channels_include)):

                    # load the channel data
                    #channel_data = mef.read_ts_channels_uutc(channels_include[iChannel], [None, None])
                    channel_data = mef.read_ts_channels_sample(channels_include[iChannel], [None, None])

                    # loop through the trials
                    for iTrial in range(len(trials_onset)):
                        sample_start = int(round(trials_onset[iTrial] * srate))
                        data[iTrial, iChannel, :] = channel_data[sample_start:sample_start + size_time_s]


            #
            #TODO: check for invalid data (could happen when onset is wrong etc.)

            #
            # retrieve the stimulation-pairs and for each pair take the average over trials
            # (note that the 'no_concat_bidirectional_pairs' argument is taken into account here)
            #

            # variable to store the stimulation pairs in
            pairs_label = []
            pairs_trials = []

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
                        pairs_trials.append(indices)

            # display Pair/Trial information
            print('Stimulation pairs:                               ' + str(len(pairs_label)))
            for iPair in range(len(pairs_label)):
                if iPair % 10 == 0:
                    print('                                                 ', end='');
                print(str(pairs_label[iPair]) + ' (' + str(len(pairs_trials[iPair])) + ' trials)   ', end='')
                if iPair > 0 and ((iPair + 1) % 10 == 0):
                    print('')

            # create a variable to store each stimulation-pair average in (stim-pair x channel x time)
            pairs_average = np.empty((len(pairs_label), len(channels_include), size_time_s))
            pairs_average.fill(np.nan)

            # for each stimulation-pair, calculate the average over trials
            for iPair in range(len(pairs_label)):
                pairs_average[iPair, :, :] = np.nanmean(data[pairs_trials[iPair], :, :], axis=0)

            # calculate average signal per electrode over all pairs
            pairs_electrode_average = np.nanmean(pairs_average, axis=0)

            # calculate average signal per pair over all electrodes
            pairs_pairs_average = np.nanmean(pairs_average, axis=1)


            #
            # perform the detection
            #

            #TODO: N1 detection



            #
            # generate the electrodes plot
            #

            # determine the electrode subplot layout
            ncols, nrows = getPlotLayout(SUBPLOT_LAYOUT_RATIO, len(channels_include))

            # create a figure and resize the figure to the image output resolution
            #from matplotlib.figure import Figure
            #fig = Figure()
            fig = plt.figure()
            DPI = fig.get_dpi()
            fig.set_size_inches(float(OUTPUT_IMAGE_RESOLUTION[0]) / float(DPI), float(OUTPUT_IMAGE_RESOLUTION[1]) / float(DPI))

            # generate the x-axis values
            x = np.arange(size_time_s)
            x = x / srate - PRESTIM_EPOCH

            # loop through the electrodes
            for iElec in range(len(channels_include)):

                # determine the x and y of the plot
                y_plot = floor(iElec / ncols)
                x_plot = iElec - y_plot * ncols

                # add the subplot
                ax = fig.add_subplot(nrows, ncols, iElec + 1)

                #
                ax.set_title(channels_include[iElec], fontsize=10)

                # plot each pair
                for iPair in range(len(pairs_label)):
                    ax.plot(x, pairs_average[iPair, iElec, :], linewidth=0.50)

                # plot the average over pairs
                ax.plot(x, pairs_electrode_average[iElec, :], linewidth=0.60, color='black')

                # x axis
                if iElec > len(channels_include) - ncols - 1:
                    ax.set_xlabel('Time (in secs)')
                else:
                    ax.get_xaxis().set_ticks([])

                # y axis
                if x_plot == 0:
                    ax.set_ylabel('Signal')
                else:
                    ax.get_yaxis().set_ticks([])

            # display/save figure
            fig.show()
            #fig.savefig(os.path.join(args.output_dir, 'electrodes.png'), bbox_inches='tight')
            #plt.savefig(os.path.join(args.output_dir, 'electrodes.png'), bbox_inches='tight', pad_inches = 0)
            #plt.savefig(os.path.join(args.output_dir, 'electrodes.png'), bbox_inches='tight')


            #
            # generate the pairs plot
            #

            # determine the pairs subplot layout
            ncols, nrows = getPlotLayout(SUBPLOT_LAYOUT_RATIO, len(pairs_label))

            # create a figure and resize the figure to the image output resolution
            #from matplotlib.figure import Figure
            #fig = Figure()
            fig = plt.figure()
            DPI = fig.get_dpi()
            fig.set_size_inches(float(OUTPUT_IMAGE_RESOLUTION[0]) / float(DPI), float(OUTPUT_IMAGE_RESOLUTION[1]) / float(DPI))

            # generate the x-axis values
            x = np.arange(size_time_s)
            x = x / srate - PRESTIM_EPOCH

            # loop through the stimulation-pairs
            for iPair in range(len(pairs_label)):

                # determine the x and y of the plot
                y_plot = floor(iPair / ncols)
                x_plot = iPair - y_plot * ncols

                # add the subplot
                ax = fig.add_subplot(nrows, ncols, iPair + 1)

                #
                ax.set_title(pairs_label[iPair], fontsize=10)

                # plot each electrode
                for iElec in range(len(channels_include)):
                    ax.plot(x, pairs_average[iPair, iElec, :], linewidth=0.50)

                # plot the average over pairs
                ax.plot(x, pairs_pairs_average[iPair, :], linewidth=0.60, color='black')

                # x axis
                if iPair > len(pairs_label) - ncols - 1:
                    ax.set_xlabel('Time (in secs)')
                else:
                    ax.get_xaxis().set_ticks([])

                # y axis
                if x_plot == 0:
                    ax.set_ylabel('Signal')
                else:
                    ax.get_yaxis().set_ticks([])

            # display/save figure
            fig.show()
            #fig.savefig(os.path.join(args.output_dir, 'electrodes.png'), bbox_inches='tight')
            #plt.savefig(os.path.join(args.output_dir, 'electrodes.png'), bbox_inches='tight', pad_inches = 0)
            #plt.savefig(os.path.join(args.output_dir, 'electrodes.png'), bbox_inches='tight')

    else:
        #
        print('Warning: participant \'' + subject_label + '\' could not be found, skipping')


