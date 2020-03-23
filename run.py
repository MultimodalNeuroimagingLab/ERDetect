#!/usr/bin/env python3
import argparse
import os
import csv
from math import isnan, ceil
from glob import glob

from bids_validator import BIDSValidator
from mne.io import read_raw_edf, read_raw_brainvision
from pymef.mef_session import MefSession
import numpy as np

#
# constants
#
VALID_FORMAT_EXTENSIONS         = ('.edf', '.vhdr', '.vmrk', '.eeg', '.mefd')   # valid data format to search for (European Data Format, BrainVision and MEF3)
PRESTIM_EPOCH                   = 2.5                                           # the amount of time (in seconds) before the stimulus that will be considered as start of the epoch (for each trial)
POSTSTIM_EPOCH                  = 2.5                                           # the amount of time (in seconds) before the stimulus that will be considered as end of the epoch (for each trial)


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
                trial_onsets = [];
                trial_stim_pairs = [];
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
                        trial_onsets.append(float(row['onset']))
                        trial_stim_pairs.append(pair)

            # remove stimulus trials which involve non-included (bad or non-ieeg) channels
            #num_trials_excluded = 0
            #for iPair in range(len(trial_stim_pairs) - 1, -1, -1):
            #    if not trial_stim_pairs[iPair][0] in channels_include or not trial_stim_pairs[iPair][1] in channels_include:
            #        del trial_onsets[iPair]
            #        del trial_stim_pairs[iPair]
            #        num_trials_excluded += 1

            # retrieve unique stimulation pairs and their onsets
            #stim_pairs = dict()
            #for iPair in range(len(trial_stim_pairs)):
            #    key = trial_stim_pairs[iPair][0] + "-" + trial_stim_pairs[iPair][1]
            #    if not key in stim_pairs:
            #        stim_pairs[key] = []
            #    stim_pairs[key].append(trial_onsets[iPair])

            # print trial information
            #print('Trial(s) removed (due to non-included channels): ' + str(num_trials_excluded))
            #print('Number of trials remaining:                      ' + str(len(trial_onsets)))
            #print('Number of unique stimulus pairs:                 ' + str(len(stim_pairs)) + '   (' + '   '.join(stim_pairs) + ')')

            # debug, limit channels
            channels_include = channels_include[0:4]


            #
            # read the data to a numpy array
            #

            # read the dataset
            extension = subset[subset.rindex("."):]
            if extension == '.edf':


                # Alternative (use pyedflib), low memory usage solution since it's ability to read per channel
                #from pyedflib import EdfReader
                #f = EdfReader(subset)
                #n = f.signals_in_file
                #signal_labels = f.getSignalLabels()
                #srate = f.getSampleFrequencies()[0]
                #size_time_t = POSTSTIM_EPOCH + PRESTIM_EPOCH
                #size_time_s = int(ceil(size_time_t * srate))
                #data = np.empty((len(trial_onsets), len(channels_include), size_time_s))
                #data.fill(np.nan)
                #for iChannel in range(len(channels_include)):
                #    channel_index = signal_labels.index(channels_include[iChannel])
                #    signal = f.readSignal(channel_index)
                #    for iTrial in range(len(trial_onsets)):
                #        sample_start = int(round(trial_onsets[iTrial] * srate))
                #        data[iTrial, iChannel, :] = signal[sample_start:sample_start + size_time_s]
                #        if iChannel == 0 and iTrial == 0:
                #            a = signal[sample_start:sample_start + size_time_s]
                #            b = signal[701495:701495 + 20]


                # read the edf data
                edf = read_raw_edf(subset, eog=None, misc=None, stim_channel=False, exclude=channels_non_ieeg, preload=True, verbose=None)

                # retrieve the sample-rate
                srate = edf.info['sfreq']

                # calculate the size of the time dimension
                size_time_t = POSTSTIM_EPOCH + PRESTIM_EPOCH
                size_time_s = int(ceil(size_time_t * srate))

                # initialize a data buffer (trials x channel x time)
                # Note: this order makes the time dimension contiguous in memory, which is handy for block copies
                data = np.empty((len(trial_onsets), len(channels_include), size_time_s))
                data.fill(np.nan)

                # loop through the included channels
                for iChannel in range(len(channels_include)):

                    # retrieve the index of the channel
                    try:

                        channel_index = edf.ch_names.index(channels_include[iChannel])

                        # loop through the trials
                        for iTrial in range(len(trial_onsets)):
                            sample_start = int(round(trial_onsets[iTrial] * srate))
                            data[iTrial, iChannel, :] = edf[channel_index, sample_start:sample_start + size_time_s][0]

                            if iChannel == 0 and iTrial == 0:
                                a = edf[channel_index, sample_start:sample_start + size_time_s][0]
                                b = edf[channel_index, 701495:701495 + 20][0]

                    except ValueError:
                        print('Error: could not find channel \'' + channels_include[iChannel] + '\' in dataset')
                        exit()

                #edf.close()
                #del edf

            elif extension == '.vhdr' or extension == '.vmrk' or extension == '.eeg':

                # read the BrainVision data
                bv = read_raw_brainvision(subset[:subset.rindex(".")] + '.vhdr', preload=True)

                # retrieve the sample-rate
                srate = round(bv.info['sfreq'])

                # calculate the size of the time dimension
                size_time_t = POSTSTIM_EPOCH + PRESTIM_EPOCH
                size_time_s = int(ceil(size_time_t * srate))

                # initialize a data buffer (trials x channel x time)
                # Note: this order makes the time dimension contiguous in memory, which is handy for block copies
                data = np.empty((len(trial_onsets), len(channels_include), size_time_s))
                data.fill(np.nan)

                # loop through the included channels
                for iChannel in range(len(channels_include)):

                    # retrieve the index of the channel
                    try:

                        channel_index = bv.info['ch_names'].index(channels_include[iChannel])

                        # loop through the trials
                        for iTrial in range(len(trial_onsets)):
                            sample_start = int(round(trial_onsets[iTrial] * srate))
                            data[iTrial, iChannel, :] = bv[channel_index, sample_start:sample_start + size_time_s][0]

                            if iChannel == 0 and iTrial == 0:
                                a = bv[channel_index, sample_start:sample_start + size_time_s][0]
                                b = bv[channel_index, 701495:701495 + 200][0]

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

                # initialize a data buffer (trials x channel x time)
                # Note: this order makes the time dimension contiguous in memory, which is handy for block copies
                data = np.empty((len(trial_onsets), len(channels_include), size_time_s))
                data.fill(np.nan)

                # loop through the included channels
                for iChannel in range(len(channels_include)):

                    # load the channel data
                    #channel_data = mef.read_ts_channels_uutc(channels_include[iChannel], [None, None])
                    channel_data = mef.read_ts_channels_sample(channels_include[iChannel], [None, None])

                    # loop through the trials
                    for iTrial in range(len(trial_onsets)):
                        sample_start = int(round(trial_onsets[iTrial] * srate))
                        data[iTrial, iChannel, :] = channel_data[sample_start:sample_start + size_time_s]

                        if iChannel == 0 and iTrial == 0:
                            a = channel_data[sample_start:sample_start + size_time_s]
                            b = channel_data[701495:701495 + 20]

            #
            # perform the detection
            #



    else:
        #
        print('Warning: participant \'' + subject_label + '\' could not be found, skipping')


