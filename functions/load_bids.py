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
import logging
from math import ceil
from mne.io import read_raw_edf, read_raw_brainvision
from pymef.mef_session import MefSession
import numpy as np
import pandas as pd
from functions.misc import print_progressbar, allocate_array


def load_channel_info(tsv_filepath):

    # retrieve the channel metadata from the channels.tsv file
    try:
        csv = pd.read_csv(tsv_filepath, sep='\t', header=0, encoding='unicode_escape', na_filter=False, dtype=str)
    except FileNotFoundError:
        logging.error('Could not find the file \'' + tsv_filepath + '\'')
        return None

    #
    if 'name' not in csv.columns:
        logging.error('Could not find the \'name\' column in \'' + tsv_filepath + '\'')
        return None
    if 'type' not in csv.columns:
        logging.error('Could not find the \'type\' column in \'' + tsv_filepath + '\'')
        return None
    if 'status' not in csv.columns:
        logging.error('Could not find the \'status\' column in \'' + tsv_filepath + '\'')
        return None

    return csv


def load_event_info(tsv_filepath, addition_required_columns=None):

    # retrieve the events from the events.tsv file
    try:
        csv = pd.read_csv(tsv_filepath, sep='\t', header=0, encoding='unicode_escape', na_filter=False, dtype=str)
    except FileNotFoundError:
        logging.error('Could not find the file \'' + tsv_filepath + '\'')
        return None

    #
    if 'onset' not in csv.columns:
        logging.error('Could not find the \'onset\' column in \'' + tsv_filepath + '\'')
        return None
    if addition_required_columns is not None:
        for column in addition_required_columns:
            if column not in csv.columns:
                logging.error('Could not find the \'' + column + '\' column in \'' + tsv_filepath + '\'')
                return None

    return csv


def load_data_epochs(data_path, channels, onsets, trial_epoch=(-1, 3), baseline_norm=None, baseline_epoch=(-1, -0.1)):
    """
    Load and epoch the data into a matrix based on channels, stimulus onsets and the epoch range (relative to the onsets)

    Args:
        data_path (str):                Path to the data file or folder
        channels (list or tuple):       The channels that should read from the data, the output will be sorted
                                        according to this input argument.
        onsets (list or tuple):         The onsets of the stimuli (e.g. trials) around which to epoch the data
        trial_epoch (tuple):            The time-span that will be considered as the signal belonging to a single trial.
                                        Expressed as a tuple with the start- and end-point in seconds relative to
                                        stimulation onset of the trial (e.g. the standard tuple of '-1, 3' will extract
                                        the signal in the period from 1s before stimulation onset to 3s after
                                        stimulation onset).
        baseline_norm (None or str):    Baseline normalization setting [None, 'Mean' or 'Median']. If other than None,
                                        normalizes each trial epoch by subtracting the mean or median of part of the
                                        trial (the epoch of the trial indicated in baseline_epoch)
        baseline_epoch (tuple):         The time-span on which the baseline is calculated, expressed as a tuple with the
                                        start- and end-point in seconds relative to stimulation onset (e.g. the
                                        standard tuple of '-1, -.1' will use the period from 1s before stimulation onset
                                        to 100ms before stimulation onset to calculate the baseline on); this arguments
                                        is only used when baseline_norm is set to mean or median

    Returns:
        sampling_rate (int or double):  the sampling rate at which the data was acquired
        data (ndarray):                 A three-dimensional array with data epochs per channel (format: channel x
                                        trials/epochs x time); or None when an error occurs

    Note: this function's units are in time relative to stimulation (e.g. trial) onset because the sample rate will
          only be known till after we read the data
    """

    # initialize the return variables to a default
    sampling_rate = None
    data = None

    # TODO: handle different units in data format

    #
    # check input
    #

    # data-set format
    data_extension = data_path[data_path.rindex("."):]
    if data_extension == '.edf':
        data_format = 0
    elif data_extension == '.vhdr' or data_extension == '.vmrk' or data_extension == '.eeg':
        data_format = 1
    elif data_extension == '.mefd':
        data_format = 2
    else:
        logging.error('Unknown data format (' + data_extension + ')')
        return None, None

    #
    if trial_epoch[1] < trial_epoch[0]:
        logging.error('Invalid \'trial_epoch\' parameter, the given end-point (at ' + str(trial_epoch[1]) + ') lies before the start-point (at ' + str(trial_epoch[0]) + ')')
        return None, None

    # baseline normalization
    baseline_method = 0
    if baseline_norm is not None or len(baseline_norm) > 0:
        if baseline_norm.lower() == 'mean' or baseline_norm.lower() == 'average':
            baseline_method = 1
        elif baseline_norm.lower() == 'median':
            baseline_method = 2
        elif baseline_norm.lower() == 'none':
            baseline_method = 0
        else:
            logging.error('Unknown normalization argument (' + baseline_norm + ')')
            return None, None

        #
        if baseline_epoch[1] < baseline_epoch[0]:
            logging.error('Invalid \'baseline_epoch\' parameter, the given end-point (at ' + str(baseline_epoch[1]) + ') lies before the start-point (at ' + str(baseline_epoch[0]) + ')')
            return None, None
        if data_format == 2:
            if baseline_epoch[0] < trial_epoch[0]:
                logging.error('Invalid \'baseline_epoch\' parameter, the given baseline start-point (at ' + str(baseline_epoch[0]) + ') lies before the trial start-point (at ' + str(trial_epoch[0]) + ')')
                return None, None
            if baseline_epoch[1] > trial_epoch[1]:
                logging.error('Invalid \'baseline_epoch\' parameter, the given baseline end-point (at ' + str(baseline_epoch[1]) + ') lies after the trial end-point (at ' + str(trial_epoch[1]) + ')')
                return None, None

    #
    # read and process the data
    #

    if data_format == 0 or data_format == 1:
        # EDF or BrainVision format, use MNE to read

        # Alternative for EDF (use pyedflib), low memory usage solution since it has the ability to read per channel
        #from pyedflib import EdfReader
        #f = EdfReader(data_path)
        #n = f.signals_in_file
        #signal_labels = f.getSignalLabels()
        #sampling_rate = f.getSampleFrequencies()[0]
        #size_time_s = int(ceil(abs(trial_epoch[1] - trial_epoch[0]) * sampling_rate))
        #data = np.empty((len(channels_include), len(onsets), size_time_s))
        #data.fill(np.nan)
        #for iChannel in range(len(channels)):
        #    channel_index = signal_labels.index(channels[iChannel])
        #    signal = f.readSignal(channel_index)
        #    for iTrial in range(len(onsets)):
        #        sample_start = int(round(onsets[iTrial] * sampling_rate))
        #        data[iChannel, iTrial, :] = signal[sample_start:sample_start + size_time_s]

        # read the data
        try:
            if data_format == 0:
                mne_raw = read_raw_edf(data_path, eog=[], misc=[], stim_channel=[], preload=True, verbose=None)
                #mne_raw = read_raw_edf(data_path, eog=None, misc=None, stim_channel=[], exclude=channels_non_ieeg, preload=True, verbose=None)
            if data_format == 1:
                mne_raw = read_raw_brainvision(data_path[:data_path.rindex(".")] + '.vhdr', preload=True)
        except Exception as e:
            logging.error('MNE could not read data, message: ' + str(e))
            return None, None

        # retrieve the sample-rate
        sampling_rate = mne_raw.info['sfreq']

        # calculate the size of the time dimension (in samples)
        size_time_s = int(ceil(abs(trial_epoch[1] - trial_epoch[0]) * sampling_rate))

        # initialize a data buffer (channel x trials/epochs x time)
        data = allocate_array((len(channels), len(onsets), size_time_s))
        if data is None:
            return None, None

        # loop through the included channels
        for iChannel in range(len(channels)):

            # (try to) retrieve the index of the channel
            try:
                channel_index = mne_raw.info['ch_names'].index(channels[iChannel])
            except ValueError:
                logging.error('Could not find channel \'' + channels[iChannel] + '\' in the dataset')
                return None, None

            # loop through the trials
            for iTrial in range(len(onsets)):

                #
                trial_sample_start = int(round((onsets[iTrial] + trial_epoch[0]) * sampling_rate))
                if trial_sample_start < 0 or trial_sample_start + size_time_s >= len(mne_raw):
                    logging.error('Cannot extract the trial with onset ' + str(onsets[iTrial]) + ', the range for extraction lies outside of the data')
                    return None, None

                #
                if baseline_method > 0:
                    baseline_start_sample = int(round((onsets[iTrial] + baseline_epoch[0]) * sampling_rate))
                    baseline_end_sample = int(round((onsets[iTrial] + baseline_epoch[1]) * sampling_rate))
                    if baseline_start_sample < 0 or baseline_end_sample >= len(mne_raw):
                        logging.error('Cannot extract the baseline for the trial with onset ' + str(onsets[iTrial]) + ', the range for the baseline lies outside of the data')
                        return None, None

                # extract the trial data and perform baseline normalization on the trial if needed
                if baseline_method == 0:
                    data[iChannel, iTrial, :] = mne_raw[channel_index, trial_sample_start:trial_sample_start + size_time_s][0]
                elif baseline_method == 1:
                    baseline_mean = np.nanmean(mne_raw[channel_index, baseline_start_sample:baseline_end_sample][0])
                    data[iChannel, iTrial, :] = mne_raw[channel_index, trial_sample_start:trial_sample_start + size_time_s][0] - baseline_mean
                elif baseline_method == 2:
                    baseline_median = np.nanmedian(mne_raw[channel_index, baseline_start_sample:baseline_end_sample][0])
                    data[iChannel, iTrial, :] = mne_raw[channel_index, trial_sample_start:trial_sample_start + size_time_s][0] - baseline_median

        # TODO: clear memory in MNE, close() doesn't seem to work, neither does remove the channels, issue MNE?
        mne_raw.close()
        del mne_raw

        # MNE always returns data in volt, convert to micro-volt
        data = data * 1000000

    elif data_format == 2:
        # MEF3 format

        # read the session metadata
        try:
            mef = MefSession(data_path, '', read_metadata=True)
        except Exception:
            logging.error('PyMef could not read data, either a password is needed or the data is corrupt')
            return None, None

        # retrieve the sample-rate and total number of samples in the data-set
        sampling_rate = mef.session_md['time_series_metadata']['section_2']['sampling_frequency'].item(0)
        num_samples = mef.session_md['time_series_metadata']['section_2']['number_of_samples'].item(0)

        # calculate the size of the time dimension (in samples)
        size_time_s = int(ceil(abs(trial_epoch[1] - trial_epoch[0]) * sampling_rate))

        # initialize a data buffer (channel x trials/epochs x time)
        data = allocate_array((len(channels), len(onsets), size_time_s))
        if data is None:
            return None, None

        # create a progress bar
        print_progressbar(0, len(onsets), prefix='Progress:', suffix='Complete', length=50)

        # loop through the trials
        for iTrial in range(len(onsets)):

            #
            trial_sample_start = int(round((onsets[iTrial] + trial_epoch[0]) * sampling_rate))
            if trial_sample_start < 0 or trial_sample_start + size_time_s >= num_samples:
                logging.error('Cannot extract the trial with onset ' + str(onsets[iTrial]) + ', the range for extraction lies outside of the data')
                return None, None

            #
            if baseline_method > 0:
                baseline_start_sample = int(round((onsets[iTrial] + baseline_epoch[0]) * sampling_rate)) - trial_sample_start
                baseline_end_sample = int(round((onsets[iTrial] + baseline_epoch[1]) * sampling_rate)) - trial_sample_start
                if baseline_start_sample < 0 or baseline_end_sample >= size_time_s:
                    logging.error('Cannot extract the baseline, the range for the baseline lies outside of the trial epoch')
                    return None, None

            # load the trial data
            try:
                trial_data = mef.read_ts_channels_sample(channels, [trial_sample_start, trial_sample_start + size_time_s])
                if trial_data is None or (len(trial_data) > 0 and trial_data[0] is None):
                    return None, None
            except Exception:
                logging.error('PyMef could not read data, either a password is needed or the data is corrupt')
                return None, None

            # loop through the channels
            for iChannel in range(len(channels)):

                if baseline_method == 0:
                    data[iChannel, iTrial, :] = trial_data[iChannel]
                elif baseline_method == 1:
                    baseline_mean = np.nanmean(trial_data[iChannel][baseline_start_sample:baseline_end_sample])
                    data[iChannel, iTrial, :] = trial_data[iChannel] - baseline_mean
                elif baseline_method == 2:
                    baseline_median = np.nanmedian(trial_data[iChannel][baseline_start_sample:baseline_end_sample])
                    data[iChannel, iTrial, :] = trial_data[iChannel] - baseline_median

            del trial_data

            # update progress bar
            print_progressbar(iTrial + 1, len(onsets), prefix='Progress:', suffix='Complete', length=50)

    #
    return sampling_rate, data


def load_data_epochs_averages(data_path, channels, conditions_onsets, trial_epoch=(-1, 3), baseline_norm=None, baseline_epoch=(-1, -0.1)):
    """
    Load, epoch and return for each channel and condition the average signal (i.e. the signal in time averaged over all
    stimuli/trials that belong to the same condition).

    Because this function only has to return the average signal for each channel and condition, it is much more memory
    efficient (this is particularly important when the amount of memory is limited by a Docker or VM)

    Args:
        data_path (str):                      Path to the data file or folder
        channels (list or tuple):             The channels that should read from the data, the output will be sorted
                                              according to this input argument.
        conditions_onsets (2d list or tuple): A two-dimensional list to indicate the conditions, and the onsets of the
                                              stimuli (e.g. trials) that belong to each condition.
                                              (format: conditions x condition onsets)
        trial_epoch (tuple):                  The time-span that will be considered as the signal belonging to a single
                                              trial. Expressed as a tuple with the start- and end-point in seconds
                                              relative to stimulation onset of the trial (e.g. the standard tuple of
                                              '-1, 3' will extract the signal in the period from 1s before stimulation
                                              onset to 3s after stimulation onset).
        baseline_norm (None or str):          Baseline normalization setting [None, 'Mean' or 'Median']. If other
                                              than None, normalizes each trial epoch by subtracting the mean or median
                                              of part of the trial (the epoch of the trial indicated in baseline_epoch)
        baseline_epoch (tuple):               The time-span on which the baseline is calculated, expressed as a tuple with
                                              the start- and end-point in seconds relative to stimulation onset (e.g. the
                                              standard tuple of '-1, -.1' will use the period from 1s before stimulation
                                              onset to 100ms before stimulation onset to calculate the baseline on);
                                              this arguments is only used when baseline_norm is set to mean or median

    Returns:
        sampling_rate (int or double):        The sampling rate at which the data was acquired
        data (ndarray):                       A three-dimensional array with signal averages per channel and condition
                                              (format: channel x condition x time); or None when an error occurs

    Note: this function's units are in time relative to stimulation (e.g. trial) onset because the sample rate will
          only be known till after we read the data
    """

    # initialize the return variables to a default
    sampling_rate = None
    data = None

    # TODO: handle different units in data format


    #
    # check input
    #

    # data-set format
    data_extension = data_path[data_path.rindex("."):]
    if data_extension == '.edf':
        data_format = 0
    elif data_extension == '.vhdr' or data_extension == '.vmrk' or data_extension == '.eeg':
        data_format = 1
    elif data_extension == '.mefd':
        data_format = 2
    else:
        logging.error('Unknown data format (' + data_extension + ')')
        return None, None

    #
    if trial_epoch[1] < trial_epoch[0]:
        logging.error('Invalid \'trial_epoch\' parameter, the given end-point (at ' + str(trial_epoch[1]) + ') lies before the start-point (at ' + str(trial_epoch[0]) + ')')
        return None, None

    # baseline normalization
    baseline_method = 0
    if baseline_norm is not None or len(baseline_norm) > 0:
        if baseline_norm.lower() == 'mean' or baseline_norm.lower() == 'average':
            baseline_method = 1
        elif baseline_norm.lower() == 'median':
            baseline_method = 2
        elif baseline_norm.lower() == 'none':
            baseline_method = 0
        else:
            logging.error('Unknown normalization argument (' + baseline_norm + ')')
            return None, None

        #
        if baseline_epoch[1] < baseline_epoch[0]:
            logging.error('Invalid \'baseline_epoch\' parameter, the given end-point (at ' + str(baseline_epoch[1]) + ') lies before the start-point (at ' + str(baseline_epoch[0]) + ')')
            return None, None
        if data_format == 2:
            if baseline_epoch[0] < trial_epoch[0]:
                logging.error('Invalid \'baseline_epoch\' parameter, the given baseline start-point (at ' + str(baseline_epoch[0]) + ') lies before the trial start-point (at ' + str(trial_epoch[0]) + ')')
                return None, None
            if baseline_epoch[1] > trial_epoch[1]:
                logging.error('Invalid \'baseline_epoch\' parameter, the given baseline end-point (at ' + str(baseline_epoch[1]) + ') lies after the trial end-point (at ' + str(trial_epoch[1]) + ')')
                return None, None

    #
    # read and process the data
    #

    if data_format == 0 or data_format == 1:
        # EDF or BrainVision format, use MNE to read

        # read the data
        try:
            if data_format == 0:
                mne_raw = read_raw_edf(data_path, eog=[], misc=[], stim_channel=[], preload=True, verbose=None)
                #mne_raw = read_raw_edf(data_path, eog=None, misc=None, stim_channel=[], exclude=channels_non_ieeg, preload=True, verbose=None)
            if data_format == 1:
                mne_raw = read_raw_brainvision(data_path[:data_path.rindex(".")] + '.vhdr', preload=True)
        except Exception as e:
            logging.error('MNE could not read data, message: ' + str(e))
            return None, None

        # retrieve the sample-rate
        sampling_rate = mne_raw.info['sfreq']

        # calculate the size of the time dimension (in samples)
        size_time_s = int(ceil(abs(trial_epoch[1] - trial_epoch[0]) * sampling_rate))

        # initialize a data buffer (channel x conditions x time)
        data = allocate_array((len(channels), len(conditions_onsets), size_time_s))
        if data is None:
            return None, None

        # loop through the conditions
        for iCondition in range(len(conditions_onsets)):

            # loop through the channels
            for iChannel in range(len(channels)):

                # retrieve the index of the channel
                try:
                    channel_index = mne_raw.info['ch_names'].index(channels[iChannel])
                except ValueError:
                    logging.error('Could not find channel \'' + channels[iChannel] + '\' in data-set')
                    return None, None

                # initialize a buffer to put all the data for this condition-channel in (trials x time)
                condition_channel_data = allocate_array((len(conditions_onsets[iCondition]), size_time_s))
                if condition_channel_data is None:
                    return None, None

                # loop through the trials in the condition
                for iTrial in range(len(conditions_onsets[iCondition])):

                    #
                    trial_sample_start = int(round((conditions_onsets[iCondition][iTrial] + trial_epoch[0]) * sampling_rate))
                    if trial_sample_start < 0 or trial_sample_start + size_time_s >= len(mne_raw):
                        logging.error('Cannot extract the trial with onset ' + str(conditions_onsets[iCondition][iTrial]) + ', the range for extraction lies outside of the data')
                        return None, None

                    #
                    if baseline_method > 0:
                        baseline_start_sample = int(round((conditions_onsets[iCondition][iTrial] + baseline_epoch[0]) * sampling_rate))
                        baseline_end_sample = int(round((conditions_onsets[iCondition][iTrial] + baseline_epoch[1]) * sampling_rate))
                        if baseline_start_sample < 0 or baseline_end_sample >= len(mne_raw):
                            logging.error('Cannot extract the baseline, the range for the baseline lies outside of the trial epoch')
                            return None, None

                    # extract the trial data and perform baseline normalization on the trial if needed
                    # MNE always returns data in volt, convert to micro-volt
                    if baseline_method == 0:
                        condition_channel_data[iTrial, :] = mne_raw[channel_index, trial_sample_start:trial_sample_start + size_time_s][0] * 1000000
                    elif baseline_method == 1:
                        baseline_mean = np.nanmean(mne_raw[channel_index, baseline_start_sample:baseline_end_sample][0] * 1000000)
                        condition_channel_data[iTrial, :] = mne_raw[channel_index, trial_sample_start:trial_sample_start + size_time_s][0] * 1000000 - baseline_mean
                    elif baseline_method == 2:
                        baseline_median = np.nanmedian(mne_raw[channel_index, baseline_start_sample:baseline_end_sample][0] * 1000000)
                        condition_channel_data[iTrial, :] = mne_raw[channel_index, trial_sample_start:trial_sample_start + size_time_s][0] * 1000000 - baseline_median

                # average the trials for each channel (within this condition) and store the results
                data[iChannel, iCondition, :] = np.nanmean(condition_channel_data, axis=0)

                # clear reference to data
                del condition_channel_data

        # TODO: clear memory in MNE, close() doesn't seem to work, neither does remove the channels, issue MNE?
        mne_raw.close()
        del mne_raw

    elif data_format == 2:
        # MEF3 format

        # read the session metadata
        try:
            mef = MefSession(data_path, '', read_metadata=True)
        except Exception:
            logging.error('PyMef could not read data, either a password is needed or the data is corrupt')
            return None, None

        # retrieve the sample-rate and total number of samples in the data-set
        sampling_rate = mef.session_md['time_series_metadata']['section_2']['sampling_frequency'].item(0)
        num_samples = mef.session_md['time_series_metadata']['section_2']['number_of_samples'].item(0)

        # calculate the size of the time dimension (in samples)
        size_time_s = int(ceil(abs(trial_epoch[1] - trial_epoch[0]) * sampling_rate))

        # initialize a data buffer (channel x conditions x time)
        data = allocate_array((len(channels), len(conditions_onsets), size_time_s))
        if data is None:
            return None, None

        # create a progress bar
        print_progressbar(0, len(conditions_onsets), prefix='Progress:', suffix='Complete', length=50)

        # loop through the conditions
        for iCondition in range(len(conditions_onsets)):

            # initialize a buffer to put all the data for this condition in (channels x trials x time)
            condition_data = allocate_array((len(channels), len(conditions_onsets[iCondition]), size_time_s))
            if condition_data is None:
                return None, None

            # loop through the trials in the condition
            for iTrial in range(len(conditions_onsets[iCondition])):

                #
                trial_sample_start = int(round((conditions_onsets[iCondition][iTrial] + trial_epoch[0]) * sampling_rate))
                if trial_sample_start < 0 or trial_sample_start + size_time_s >= num_samples:
                    logging.error('Cannot extract the trial with onset ' + str(conditions_onsets[iCondition][iTrial]) + ', the range for extraction lies outside of the data')
                    return None, None

                #
                if baseline_method > 0:
                    baseline_start_sample = int(round((conditions_onsets[iCondition][iTrial] + baseline_epoch[0]) * sampling_rate)) - trial_sample_start
                    baseline_end_sample = int(round((conditions_onsets[iCondition][iTrial] + baseline_epoch[1]) * sampling_rate)) - trial_sample_start
                    if baseline_start_sample < 0 or baseline_end_sample >= size_time_s:
                        logging.error('Cannot extract the baseline for the trial with onset ' + str(conditions_onsets[iCondition][iTrial]) + ', the range for the baseline lies outside of the data')
                        return None, None

                # load the trial data
                try:
                    trial_data = mef.read_ts_channels_sample(channels, [trial_sample_start, trial_sample_start + size_time_s])
                    if trial_data is None or (len(trial_data) > 0 and trial_data[0] is None):
                        return None, None
                except Exception as e:
                    logging.error('PyMef could not read data: ' + str(e))
                    return None, None

                # loop through the channels
                for iChannel in range(len(channels)):
                    if baseline_method == 0:
                        condition_data[iChannel, iTrial, :] = trial_data[iChannel]
                    elif baseline_method == 1:
                        baseline_mean = np.nanmean(trial_data[iChannel][baseline_start_sample:baseline_end_sample])
                        condition_data[iChannel, iTrial, :] = trial_data[iChannel] - baseline_mean
                    elif baseline_method == 2:
                        baseline_median = np.nanmedian(trial_data[iChannel][baseline_start_sample:baseline_end_sample])
                        condition_data[iChannel, iTrial, :] = trial_data[iChannel] - baseline_median

            # average the trials for each channel (within this condition) and store the results
            data[:, iCondition, :] = np.nanmean(condition_data, axis=1)

            # clear reference to data
            del condition_data, trial_data

            # update progress bar
            print_progressbar(iCondition + 1, len(conditions_onsets), prefix='Progress:', suffix='Complete', length=50)

    #
    return sampling_rate, data

