"""
Functions to load BIDS data
=====================================================



Copyright 2020, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys
from math import ceil
import numpy as np
from mne.io import read_raw_edf, read_raw_brainvision
from pymef.mef_session import MefSession
import pandas as pd


def load_channel_info(tsv_filepath):
    # retrieve the channel metadata from the channels.tsv file

    csv = pd.read_csv(tsv_filepath, sep='\t', header=0, encoding='unicode_escape', na_filter=False, dtype=str)
    if not 'name' in csv.columns:
        print('Error: could not find the \'name\' column in \'' + tsv_filepath + '\'', file=sys.stderr)
        return None
    if not 'type' in csv.columns:
        print('Error: could not find the \'type\' column in \'' + tsv_filepath + '\'', file=sys.stderr)
        return None
    if not 'status' in csv.columns:
        print('Error: could not find the \'status\' column in \'' + tsv_filepath + '\'', file=sys.stderr)
        return None

    return csv


def load_event_info(tsv_filepath, addition_required_columns=None):
    # retrieve the events from the events.tsv file

    csv = pd.read_csv(tsv_filepath, sep='\t', header=0, encoding='unicode_escape', na_filter=False, dtype=str)
    if not 'onset' in csv.columns:
        print('Error: could not find the \'onset\' column in \'' + tsv_filepath + '\'', file=sys.stderr)
        return None
    if addition_required_columns is not None:
        for column in addition_required_columns:
            if column not in csv.columns:
                print('Error: could not find the \'' + column + '\' column in \'' + tsv_filepath + '\'', file=sys.stderr)
                return None

    return csv


def load_data_epochs(data_path, channels, onsets, epoch_start, epoch_end):
    """
    load and epoch the data into a matrix based on channels, stimulus onsets and the epoch range (relative to the onsets)

    Parameters:
        data_path (str):                Path to the data file or folder
        channels (list or tuple):       The channels that should read from the data, the output will be sorted
                                        according to this input argument.
        onsets (list or tuple):         The onsets of the stimuli (e.g. trials) around which to epoch the data
        epoch_start (double):           The offset - in seconds relative to the onset - that will be considered as the
                                        starting-point of an epoch (e.g. -.5 would include half a second before the
                                        onset of the stimulus to each epoch; .2 would make the epoch start 200ms
                                        after the onset of the stimulus)
        epoch_end (double):             The offset - in seconds relative to the onset - that will be considered as the
                                        end-point of an epoch (e.g. 2.0 would include two seconds after the onset of the
                                        stimulus to each epoch)

    Returns:
        sampling rate (int or double):  the sampling rate at which the data was acquired
        data (ndarray):                 A three-dimensional array with data epochs per channel
                                        (matrix format: channel x trials/epochs x time))

    Note: this function works with onsets and epoch-offsets because the sample rate will only be known once we
          read the data
    """

    # TODO: input argument checks

    # read the dataset
    extension = data_path[data_path.rindex("."):]
    if extension == '.edf':


        # Alternative (use pyedflib), low memory usage solution since it has the ability to read per channel
        #from pyedflib import EdfReader
        #f = EdfReader(data_path)
        #n = f.signals_in_file
        #signal_labels = f.getSignalLabels()
        #srate = f.getSampleFrequencies()[0]
        #size_time_t = epoch_end + epoch_start
        #size_time_s = int(ceil(size_time_t * srate))
        #data = np.empty((len(channels_include), len(onsets), size_time_s))
        #data.fill(np.nan)
        #for iChannel in range(len(channels)):
        #    channel_index = signal_labels.index(channels[iChannel])
        #    signal = f.readSignal(channel_index)
        #    for iTrial in range(len(onsets)):
        #        sample_start = int(round(onsets[iTrial] * srate))
        #        data[iChannel, iTrial, :] = signal[sample_start:sample_start + size_time_s]


        # read the edf data
        edf = read_raw_edf(data_path, eog=None, misc=None, stim_channel=[], preload=True, verbose=None)
        #edf = read_raw_edf(data_path, eog=None, misc=None, stim_channel=[], exclude=channels_non_ieeg, preload=True, verbose=None)

        # retrieve the sample-rate
        srate = edf.info['sfreq']

        # calculate the size of the time dimension
        size_time_t = abs(epoch_end - epoch_start)
        size_time_s = int(ceil(size_time_t * srate))

        # initialize a data buffer (channel x trials/epochs x time)
        # Note: this order makes the time dimension contiguous in memory, which is handy for block copies
        data = np.empty((len(channels), len(onsets), size_time_s))
        data.fill(np.nan)

        # loop through the included channels
        for iChannel in range(len(channels)):

            # (try to) retrieve the index of the channel
            try:
                channel_index = edf.ch_names.index(channels[iChannel])

                # loop through the trials
                for iTrial in range(len(onsets)):
                    sample_start = int(round((onsets[iTrial] + epoch_start) * srate))
                    # TODO: check if sample_start and sample_end are within range
                    data[iChannel, iTrial, :] = edf[channel_index, sample_start:sample_start + size_time_s][0]

            except ValueError:
                print('Error: could not find channel \'' + channels[iChannel] + '\' in dataset')
                return None, None

        edf.close()
        del edf

    elif extension == '.vhdr' or extension == '.vmrk' or extension == '.eeg':

        # read the BrainVision data
        bv = read_raw_brainvision(data_path[:data_path.rindex(".")] + '.vhdr', preload=True)

        # retrieve the sample-rate
        srate = round(bv.info['sfreq'])

        # calculate the size of the time dimension
        size_time_t = abs(epoch_end - epoch_start)
        size_time_s = int(ceil(size_time_t * srate))

        # initialize a data buffer (channel x trials/epochs x time)
        # Note: this order makes the time dimension contiguous in memory, which is handy for block copies
        data = np.empty((len(channels), len(onsets), size_time_s))
        data.fill(np.nan)

        # loop through the included channels
        for iChannel in range(len(channels)):

            # (try to) retrieve the index of the channel
            try:
                channel_index = bv.info['ch_names'].index(channels[iChannel])

                # loop through the trials
                for iTrial in range(len(onsets)):
                    sample_start = int(round((onsets[iTrial] + epoch_start) * srate))
                    # TODO: check if sample_start and sample_end are within range
                    data[iChannel, iTrial, :] = bv[channel_index, sample_start:sample_start + size_time_s][0]

            except ValueError:
                print('Error: could not find channel \'' + channels[iChannel] + '\' in dataset')
                return None, None

    elif extension == '.mefd':

        # read the session metadata
        mef = MefSession(data_path, '', read_metadata=True)

        # retrieve the sample-rate
        srate = mef.session_md['time_series_metadata']['section_2']['sampling_frequency'].item(0)

        # calculate the size of the time dimension
        size_time_t = abs(epoch_end - epoch_start)
        size_time_s = int(ceil(size_time_t * srate))

        # initialize a data buffer (channel x trials/epochs x time)
        # Note: this order makes the time dimension contiguous in memory, which is handy for block copies
        data = np.empty((len(channels), len(onsets), size_time_s))
        data.fill(np.nan)

        # loop through the trials
        for iTrial in range(len(onsets)):
            sample_start = int(round((onsets[iTrial] + epoch_start) * srate))
            sample_end = sample_start + size_time_s

            # TODO: check if sample_start and sample_end are within range

            # load the trial data
            trial_data = mef.read_ts_channels_sample(channels, [sample_start, sample_end])

            # loop through the channels
            for iChannel in range(len(channels)):
                data[iChannel, iTrial, :] = trial_data[iChannel]


    return srate, data
