"""
Function file for 'ieeg_detect_n1peak'
=====================================================
Detect N1 peaks


Copyright 2020, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)
Adapted from:
    Original author: Dorien van Blooijs (2018)
    Adjusted by: Jaap van der Aar, Dora Hermes, Dorien van Blooijs, Giulio Castegnaro, (UMC Utrecht, 2019)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import sys
import warnings
import numpy as np
from functions.peak_finder import peak_finder

# debug/development
#import scipy.io as sio
#mat = sio.loadmat('D:\\output\\average_ccep.mat')
#data = mat['average_ccep']
#del mat
#stim_onset_index = 2048     # 0-2047 are considered pre-stim
#sampling_rate = 2048
#n1_search_end = 0.09
#baseline_std_fact = 3.4

def ieeg_detect_n1peaks(data, stim_onset_index, sampling_rate, n1_search_end=0.09, baseline_std_fact=3.4):
    """
    ieeg_detect_n1peaks

    Parameters:
        data (ndarray):                 A three-dimensional array with the average signal per electrode and stimulus-pair.
                                        (matrix format: electrodes x stimulation-pairs x time)
        stim_onset_index (int):         the time-point on the input data's time-dimension of stimulation onset
                                        (as a 0-based sample-index, all indices before this value are considered pre-stim)
        sampling_rate (int or double):  the sampling rate at which the data was acquired
        n1_search_end (double):         the end-point of the time-span in which a N1 will be searched for, expressed
                                        in seconds relative to stimulation onset (e.g. a value 0.09 will have the algorithm
                                        search for an N1 up to 90ms after stimulation onset)
        baseline_std_fact (double):     the factor that is applied to the standard deviation of the baseline amplitude, that
                                        defines the threshold which needs to be exceeded to detect a peak (the minimum std
                                        is considered 50uV; therefore a factor of 3.4 is recommended to end up with a
                                        conservative threshold of 170 uV)

    Returns:
        ndarrayint: ...

    """

    STD_BASELINE_RANGE  = (-1, -.1)     # the time-window - (in seconds) relative to stimulation onset - onto which the baseline standard deviation is calculated
    PEAK_SEARCH_RANGE   = (0, .5)       # the time-window - (in seconds) relative to stimulation onset - in which to search for peaks
    N1_SEARCH_START    = 0.009          # the start-point - (in seconds) relative to stimulation onset - of the time-span in which peaks will be considered to be a N1

    #
    # data parameter
    #

    # TODO:


    num_samples = data.shape[2]

    #
    # onset parameter
    #

    # TODO:

    #
    #
    #

    # TODO: sampling_rate and n1_search_end checks


    #
    # peak detection
    #

    # determine the std baseline range in samples
    std_baseline_start = int(round(STD_BASELINE_RANGE[0] * sampling_rate)) + stim_onset_index
    std_baseline_end = int(round(STD_BASELINE_RANGE[1] * sampling_rate)) + stim_onset_index
    if std_baseline_start < 0:
        print("Error: " + os.path.basename(__file__) + " - the data epoch is not big enough, the baseline requires at least " + str(stim_onset_index + abs(std_baseline_start)) + " samples before stimulation onset", file=sys.stderr)
        return None, None

    # determine the peak search window in samples
    peak_search_start = int(round(PEAK_SEARCH_RANGE[0] * sampling_rate)) + stim_onset_index
    peak_search_end = int(round(PEAK_SEARCH_RANGE[1] * sampling_rate)) + stim_onset_index
    if peak_search_end > num_samples:
        print("Error: " + os.path.basename(__file__) + " - the data epoch is not big enough, the peak window requires at least " + str(stim_onset_index + abs(std_baseline_start)) + " samples after stimulation onset", file=sys.stderr)
        return None, None

    # determine the start- and end-point (in samples) of the time-span in which to search for an N1
    n1_search_start_sample = int(round(N1_SEARCH_START * sampling_rate)) + stim_onset_index
    n1_search_end_sample = int(round(n1_search_end * sampling_rate)) + stim_onset_index
    if n1_search_end_sample < n1_search_start_sample:
        print("Error: " + os.path.basename(__file__) + " - invalid 'n1_search_end' parameter, the given end-point (at " + str(n1_search_end) + ") lies before the start-point (at t = " + str(N1_SEARCH_START) + ")", file=sys.stderr)
        return None, None

    # initialize an output buffer (electrode x stimulation-pair)
    n1_peak_indices = np.empty((data.shape[0], data.shape[1]))
    n1_peak_indices.fill(np.nan)
    n1_peak_amplitudes = np.empty((data.shape[0], data.shape[1]))
    n1_peak_amplitudes.fill(np.nan)

    # for every electrode
    for iElec in range(data.shape[0]):

        # for every stimulation-pair
        for iPair in range(data.shape[1]):

            # retrieve the part of the signal to search for peaks in
            signal = data[iElec, iPair, peak_search_start + 1:peak_search_end]

            # continue if all are nan (often the case when the stim-electrodes are nan-ed out on the electrode dimensions)
            if np.all(np.isnan(signal)):
                continue

            # peak_finder is not robust against incidental nans, make 0
            signal[np.isnan(signal)] = 0

            # use peak_finder function to find the negative peak indices and their amplitude
            (neg_inds, neg_mags) = peak_finder(signal,
                                               sel=20,  # the number of samples around a peak not considered as another peak
                                               thresh=None,
                                               extrema=-1,
                                               include_endpoints=True,
                                               interpolate=False)

            # if a peak is found on the first sample, then that is not an actual peak, remove
            if neg_inds is not None and len(neg_inds) > 0 and neg_inds[0] == 0:
                neg_inds = np.delete(neg_inds, 0)
                neg_mags = np.delete(neg_mags, 0)

            # if there are no peaks, continue to next
            if neg_inds is None or len(neg_inds) == 0:
                continue

            # shift the indices to align with the full epoch (not the subsection that was passed to the peak_finder)
            neg_inds = neg_inds + peak_search_start

            # keep the peaks within the N1 search range, or continue if there are none
            in_range = (neg_inds >= n1_search_start_sample) & (neg_inds <= n1_search_end_sample)
            if any(in_range):
                neg_inds = neg_inds[in_range]
                neg_mags = neg_mags[in_range]
            else:
                continue

            # find the index of the first peak maximum
            max_ind = np.where(abs(neg_mags) == np.max(abs(neg_mags)))[0][0]

            # make sure the peak is negative, else wise continue to next
            if neg_mags[max_ind] > 0:
                continue

            # make sure the signal is not saturated, continue to next if it is
            if abs(neg_mags[max_ind]) > 3000:
                continue


            #
            # filtering by baseline
            #

            # calculate the std of the baseline samples
            warnings.filterwarnings('error')
            try:
                baseline_std = np.nanstd(data[iElec, iPair, std_baseline_start:std_baseline_end])
            except Warning:
                # assume because of nans; which is often the case when the stimulated electrodes are
                # nan-ed out on the electrode dimensions, just continue to next
                continue

            # if the baseline_std is smaller that the minimally needed SD,
            # which is validated at 50 uV, use this the minSD as baseline_std
            if baseline_std < 50:
                baseline_std = 50

            # check if the peak value does not exceeds the baseline standard deviation time a factor
            if abs(neg_mags[max_ind]) < baseline_std_fact * abs(baseline_std):
                continue

            # store the peak
            n1_peak_indices[iElec, iPair] = neg_inds[max_ind]
            n1_peak_amplitudes[iElec, iPair] = neg_mags[max_ind]


    # return results
    return n1_peak_indices, n1_peak_amplitudes
