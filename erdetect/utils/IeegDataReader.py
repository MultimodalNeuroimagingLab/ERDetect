"""
Universal IEEG data reader class that wraps around PyMef and MNE
=====================================================


Copyright 2022, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import logging
from mne.io import read_raw_edf, read_raw_brainvision
from pymef.mef_session import MefSession


#
# constants
#
VALID_FORMAT_EXTENSIONS         = ('.mefd', '.edf', '.vhdr', '.vmrk', '.eeg')   # valid data format to search for (European Data Format, BrainVision and MEF3)


class IeegDataReader:

    initialized = False
    data_path = ''
    data_extension = ''
    data_format = -1

    mne_raw = None                  # raw data-object when MNE is used to access data
    mef_session = None              # MefSession when PyMef is used to access data

    sampling_rate = -1              # the sampling rate of the dataset (from metadata, assumes same rate over channels)
    num_samples = -1                # the total number of samples per channel (from metadata, assumes same rate over channels)

    def __init__(self, data_path):
        self.data_path = data_path

        # data-set format
        self.data_extension = data_path[data_path.rindex("."):]
        if self.data_extension == '.edf':
            self.data_format = 0
        elif self.data_extension == '.vhdr' or self.data_extension == '.vmrk' or self.data_extension == '.eeg':
            self.data_format = 1
        elif self.data_extension == '.mefd':
            self.data_format = 2
        else:
            logging.error('Unknown data format (' + self.data_extension + ')')
            raise ValueError('Unknown data format (' + self.data_extension + ')')

    def init(self):
        """
        Initialize the dataset. This will at least load the metadata. With EDF or Brainvision, MNE will already load
        the entire dataset into memory

        """

        # check if a valid dataset was set
        if self.data_format == -1:
            raise RuntimeError('No valid dataset was set')

        # TODO: handle different units in data format

        # check the format to decide what to load initially
        if self.data_format == 0 or self.data_format == 1:
            # if EDF or BrainVision format, use MNE to read
            # since MNE cannot read per channel, there is no other choice than to read the whole dataset


            # Alternative for EDF (use pyedflib), low memory usage solution since it has the ability to read per channel
            #from pyedflib import EdfReader
            #f = EdfReader(data_path)
            #n = f.signals_in_file
            #signal_labels = f.getSignalLabels()
            #sampling_rate = f.getSampleFrequencies()[0]
            #trial_num_samples = int(ceil(abs(trial_epoch[1] - trial_epoch[0]) * sampling_rate))
            #data = np.empty((len(channels_include), len(onsets), trial_num_samples))
            #data.fill(np.nan)
            #for channel_counter in range(len(channels)):
            #    channel_idx = signal_labels.index(channels[channel_counter])
            #    signal = f.readSignal(channel_idx)
            #    for trial_idx in range(len(onsets)):
            #        sample_start = int(round(onsets[trial_idx] * sampling_rate))
            #        data[channel_counter, trial_idx, :] = signal[sample_start:sample_start + trial_num_samples]

            # read the data
            try:
                if self.data_format == 0:
                    self.mne_raw = read_raw_edf(self.data_path, eog=[], misc=[], stim_channel=[], preload=True, verbose=None)
                    #self.mne_raw = read_raw_edf(data_path, eog=None, misc=None, stim_channel=[], exclude=channels_non_ieeg, preload=True, verbose=None)
                    self.mne_raw._data *= 1000000
                if self.data_format == 1:
                    self.mne_raw = read_raw_brainvision(self.data_path[:self.data_path.rindex(".")] + '.vhdr', preload=True)
                    self.mne_raw._data *= 1000000

            except RuntimeError as e:
                logging.error('MNE could not read data, message: ' + str(e))
                raise RuntimeError('MNE could not read data')

            # retrieve the sample-rate and the number of samples
            self.sampling_rate = self.mne_raw.info['sfreq']
            self.num_samples = self.mne_raw.n_times

        elif self.data_format == 2:
            # MEF3 format

            # read the session metadata
            try:
                self.mef_session = MefSession(self.data_path, '', read_metadata=True)
            except RuntimeError:
                logging.error('PyMef could not read data, either a password is needed or the data is corrupt')
                raise RuntimeError('PyMef could not read data')

            # TODO: check if sampling_rate and num_samples is equal for each channel

            # retrieve the sample-rate and total number of samples in the data-set
            self.sampling_rate = self.mef_session.session_md['time_series_metadata']['section_2']['sampling_frequency'].item(0)
            self.num_samples = self.mef_session.session_md['time_series_metadata']['section_2']['number_of_samples'].item(0)

        # return success on initializing the data
        self.initialized = True
        return True


    def close(self):

        if self.data_format == 0 or self.data_format == 1:
            # TODO: clear memory in MNE, close() doesn't seem to work, neither does remove the channels, issue MNE?
            self.mne_raw.close()
            del self.mne_raw._data
            del self.mne_raw

        elif self.data_format == 2:
            del self.mef_session


    def retrieve_channel_data(self, channel_name, ensure_own_data=True):
        """
        Retrieve the channel data (mef = numpy data-array, mne = numpy data-view)

        ensure_own_data (bool):           Makes sure that the returned numpy array has its own data (is not a view)
        """

        # if not initialized, try to initialize first
        if not self.initialized:
            if not self.init():
                raise RuntimeError('Could not initialize data-reader')

        # retrieve the channel data
        try:
            if self.data_format == 0 or self.data_format == 1:
                return IeegDataReader.__retrieve_channel_data_mne(self.mne_raw, channel_name, ensure_own_data)
            elif self.data_format == 2:
                return IeegDataReader.__retrieve_channel_data_mef(self.mef_session, channel_name)

        except Exception as e:
            logging.error('Could not retrieve data' + str(e))
            raise RuntimeError('Could not retrieve data')


    @staticmethod
    def __retrieve_channel_metadata_mef(mef_session, channel_name):
        """
        Retrieve the MEF3 channel metadata by channel name
        """

        channel_metadata = None
        for ts_channel_name, ts_channel_metadata in mef_session.session_md["time_series_channels"].items():
            if ts_channel_name == channel_name:
                channel_metadata = ts_channel_metadata
                break
        if channel_metadata is None:
            logging.error('Could not find metadata for channel ' + channel_name + ', assuming there is no such channel in the dataset')
            return None

        return channel_metadata


    @staticmethod
    def __retrieve_channel_data_mef(mef_session, channel_name):
        """
        Retrieve the MEF3 channel data by channel name (retrieve numpy data array)
        """

        # find the channel metadata by channel name
        channel_metadata = IeegDataReader.__retrieve_channel_metadata_mef(mef_session, channel_name)
        if channel_metadata is None:
            raise LookupError('Could not find channel')

        # load the channel data
        try:
            channel_data = mef_session.read_ts_channels_sample([channel_name], (None, None))
        except Exception:
            logging.error('PyMef could not read data, either a password is needed or the data is corrupt')
            raise RuntimeError('Could read data')

        # return and apply a conversion factor if needed
        channel_conversion_factor = channel_metadata['section_2']['units_conversion_factor'].item(0)

        if channel_conversion_factor != 0 and channel_conversion_factor != 1:
            channel_data[0] *= channel_conversion_factor
        return channel_data[0]


    @staticmethod
    def __retrieve_channel_index(mne_raw, channel_name):
        """
        Retrieve the MNE channel index by channel name
        """

        try:
            channel_idx = mne_raw.info['ch_names'].index(channel_name)
        except ValueError:
            logging.error('Could not find channel \'' + channel_name + '\' in the dataset')
            return None

        return channel_idx


    @staticmethod
    def __retrieve_channel_data_mne(mne_raw, channel_name, ensure_own_data):
        """
        Retrieve the channel data by channel name

        ensure_own_data (bool):           Makes sure that the returned numpy array has its own data (is not a view)
        """

        # retrieve the index of the channel
        channel_idx = IeegDataReader.__retrieve_channel_index(mne_raw, channel_name)
        if channel_idx is None:
            raise LookupError('Could not find channel')

        # return the channel data
        if ensure_own_data:
            return mne_raw._data[channel_idx, :].copy()
        else:
            return mne_raw._data[channel_idx, :]


    def retrieve_sample_range_data(self, channels, sample_start, sample_end, ensure_own_data=True):
        """
        Retrieve a specific range of samples of data of the indicated channels (mef = numpy data-array, mne = numpy data-view)
        Returns a list with the channel data, format <list of channels> x <requested samples>

        ensure_own_data (bool):           Makes sure that the returned numpy array has its own data (is not a view)
        """

        # make sure the input is a list
        if isinstance(channels, str):
            channels = [channels]

        # if not initialized, try to initialize first
        if not self.initialized:
            if not self.init():
                raise RuntimeError('Could not initialize data-reader')

        #
        if self.data_format == 0 or self.data_format == 1:

            # create a list with the numpy array(-views)
            sample_data = [None] * len(channels)

            # loop over the channels in the data
            for channel_counter in range(len(channels)):

                # retrieve the index of the channel
                channel_idx = IeegDataReader.__retrieve_channel_index(self.mne_raw, channels[channel_counter])
                if channel_idx is None:
                    raise LookupError('Could not find channel')

                # pick (slice) a channel data-range (numpy view)
                if ensure_own_data:
                    sample_data[channel_counter] = self.mne_raw._data[channel_idx, sample_start:sample_end].copy()
                else:
                    sample_data[channel_counter] = self.mne_raw._data[channel_idx, sample_start:sample_end]

            return sample_data
        
        elif self.data_format == 2:

            # load the trial data
            try:
                sample_data = self.mef_session.read_ts_channels_sample(channels, [sample_start, sample_end])
                if sample_data is None or (len(sample_data) > 0 and sample_data[0] is None):
                    raise RuntimeError('Could read data')

                # loop over the channels in the data
                for channel_counter in range(len(channels)):

                    # find the channel metadata by channel name
                    channel_metadata = IeegDataReader.__retrieve_channel_metadata_mef(self.mef_session, channels[channel_counter])
                    if channel_metadata is None:
                        raise LookupError('Could not find channel')

                    # apply a conversion factor if needed
                    channel_conversion_factor = channel_metadata['section_2']['units_conversion_factor'].item(0)
                    if channel_conversion_factor != 0 and channel_conversion_factor != 1:
                        sample_data[channel_counter] *= channel_conversion_factor

            except Exception:
                logging.error('PyMef could not read data, either a password is needed or the data is corrupt')
                raise RuntimeError('Could read data')

            #
            return sample_data
