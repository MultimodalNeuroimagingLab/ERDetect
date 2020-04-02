"""
Function file for 'ieeg_detect_n1peak'
=====================================================
Detect N1 peaks


% original author: Dorien van Blooijs, January 2018
% modified by: Jaap van der Aar, Dora Hermes, Dorien van Blooijs, Giulio Castegnaro, UMC Utrecht, 2019
"""
import os
import sys
import warnings
import numpy as np
import scipy.io as sio
from functions.peak_finder import peak_finder

# debug
mat = sio.loadmat('D:\\output\\average_ccep.mat')
data = mat['average_ccep']
del mat

stim_onset_index = 2048     # 0-2047 are considered pre-stim
sampling_rate = 2048


"""
ccep_detect_n1peak

Parameters:
    data (ndarray):                 multidimensional array with the average signal per electrode and stimulus-pair.
                                    (matrix format: electrodes x stimulation-pairs x time)
    stim_onset_index (int):         the time-point on the input data's time-dimension of stimulation onset
                                    (as a 0-based sample-index, all indices before this value are considered pre-stim)
    sampling_rate (int or double):  the sampling rate at which the data was acquired   

Returns:
    int: Description of return value

"""

STD_BASELINE_RANGE  = (-1, -.1)     # the time-range - (in seconds) relative to stimulation onset - onto which the baseline standard deviation is calculated
PEAK_FIND_RANGE     = (0, .5)     # the time-range - (in seconds) relative to stimulation onset - in which to search for peaks

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
# peak detection
#



# determine the std baseline range in samples
std_baseline_start = int(round(STD_BASELINE_RANGE[0] * sampling_rate)) + stim_onset_index
std_baseline_end = int(round(STD_BASELINE_RANGE[1] * sampling_rate)) + stim_onset_index
if std_baseline_start < 0:
    print("Error: " + os.path.basename(__file__) + " - the data epoch is not big enough, the baseline requires at least " + str(stim_onset_index + abs(std_baseline_start)) + " samples before stimulation onset", file=sys.stderr)
    #exit()  # return

# determine the peak finding window in samples
peak_find_start = int(round(PEAK_FIND_RANGE[0] * sampling_rate)) + stim_onset_index
peak_find_end = int(round(PEAK_FIND_RANGE[1] * sampling_rate)) + stim_onset_index
if peak_find_end > num_samples:
    print("Error: " + os.path.basename(__file__) + " - the data epoch is not big enough, the peak window requires at least " + str(stim_onset_index + abs(std_baseline_start)) + " samples after stimulation onset", file=sys.stderr)
    #exit()  # return

# initialize an output buffer (trials/epochs x channel x time)
n1_peak_sample = np.empty((data.shape[0], data.shape[1]))
n1_peak_sample.fill(np.nan)
n1_peak_amplitude = np.empty((data.shape[0], data.shape[1]))
n1_peak_amplitude.fill(np.nan)

# for every electrode
for iElec in range(data.shape[0]):

    # for every stimulation-pair
    for iPair in range(data.shape[1]):

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

        # use peakfinder to the negative peaks and their amplitude
        (neg_inds, neg_mags) = peak_finder(data[iElec, iPair, peak_find_start + 1:peak_find_end],
                                           sel=20,          # the number of samples around a peak not considered as another peak
                                           thresh=None,
                                           extrema=-1,
                                           include_endpoints=True,
                                           interpolate=False)

        # if a peak is found on the first sample, then that is not an actual peak, remove
        if len(neg_inds) > 0 and neg_inds[0] == 0:
            neg_inds = np.delete(neg_inds, 0)
            neg_mags = np.delete(neg_mags, 0)

        if len(neg_inds) > 0:
            pass


# debug/develop
iElec = 13
iPair = 13