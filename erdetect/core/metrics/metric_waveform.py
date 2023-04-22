"""
The waveform metric

Copyright 2022, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
from scipy import signal
from ...core.config import get as config


def metric_waveform(sampling_rate, data, baseline):
    """
    Calculate a waveform (10-30Hz) metric, called per (measurement) channel and per condition (=stim-pair)

    Args:
        sampling_rate (int):                  The sampling rate of the data
        data (ndarray):                       2D data matrix (represented as trials x samples)
        baseline (ndarray):                   2D baseline data matrix (represented as trials x samples)

    Returns:
        A single metric value
    """

    trial_epoch = config('trials', 'trial_epoch')
    baseline_norm = config('trials', 'baseline_norm')
    waveform_epoch = config('metrics', 'waveform', 'epoch')
    bandpass = config('metrics', 'waveform', 'bandpass')

    # calculate the sample indices for the waveform epoch (relative to the trial epoch)
    start_sample = round((waveform_epoch[0] - trial_epoch[0]) * sampling_rate)
    end_sample = round((waveform_epoch[1] - trial_epoch[0]) * sampling_rate)

    # extract the data to calculate the metric and normalize
    if baseline_norm.lower() == 'mean' or baseline_norm.lower() == 'average':
        metric_data = data[:, start_sample:end_sample] - np.nanmean(baseline, axis=1)[:, None]
    elif baseline_norm.lower() == 'median':
        metric_data = data[:, start_sample:end_sample] - np.nanmedian(baseline, axis=1)[:, None]
    else:
        # TODO: check when no normalization to baseline, whether waveform method still works, or should give warning
        return np.nan

    # take the average over all trials
    metric_data = np.nanmean(metric_data, axis=0)

    # recenter the segment to 0
    metric_data -= np.nanmean(metric_data)


    #
    # perform bandpass filtering using a butterworth filter
    #

    # third order Butterworth
    Rp = 3
    Rs = 60

    #
    delta = 0.001 * 2 / sampling_rate
    low_p = bandpass[1] * 2 / sampling_rate
    high_p = bandpass[0] * 2 / sampling_rate
    high_s = max(delta, high_p - 0.1)
    low_s = min(1 - delta, low_p + 0.1)

    # Design a butterworth (band-pass) filter
    # Note: the 'buttord' output here differs slight from matlab, because the scipy make a change in scipy 0.14.0 where
    #       the choice of which end of the transition region was switched from the stop-band edge to the pass-band edge
    n_band, wn_band = signal.buttord([high_p, low_p], [high_s, low_s], Rp, Rs, True)
    bf_b, bf_a = signal.butter(n_band, wn_band, 'band', analog=False)

    # band-pass can only be performed with enough samples, return nan elsewise
    if metric_data.shape[0] <= 3 * (max(len(bf_b), len(bf_a)) - 1):
        return np.nan

    # Perform the band-passing
    # Note: custom padlen to match the way matlab does it (-1 is omitted in scipy)
    metric_data = signal.filtfilt(bf_b, bf_a, metric_data, padtype='odd', padlen=3 * (max(len(bf_b), len(bf_a)) - 1))

    # calculate the band power using a hilbert transformation
    band_power_sm = np.power(abs(signal.hilbert(metric_data)), 2)

    # return the highest power value over time
    return np.max(band_power_sm)
