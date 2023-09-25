"""
The cross projection metric class
Cross-projection concept adapted from: Dora Hermes and Kai Miller


Copyright 2023, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
from scipy import stats
from erdetect.core.metrics.metric_interface import MetricInterface
from erdetect.core.config import get as config


class MetricCrossProj(MetricInterface):

    @staticmethod
    def process_callback(sampling_rate, data, baseline):
        """
        Callback that calculates the cross-projection metric for a subset of the data.
        This is called per (measurement) channel and per condition (=stim-pair)

        Args:
            sampling_rate (int):                  The sampling rate of the data
            data (ndarray):                       2D data matrix (represented as trials x samples)
            baseline (ndarray):                   2D baseline data matrix (represented as trials x samples)

        Returns:
            A single metric value
        """

        trial_epoch = config('trials', 'trial_epoch')
        baseline_norm = config('trials', 'baseline_norm')
        cross_proj_epoch = config('metrics', 'cross_proj', 'epoch')

        # calculate the sample indices for the cross-projection epoch (relative to the trial epoch)
        start_sample = round((cross_proj_epoch[0] - trial_epoch[0]) * sampling_rate)
        end_sample = round((cross_proj_epoch[1] - trial_epoch[0]) * sampling_rate)

        # extract the data to calculate the metric and normalize
        if baseline_norm.lower() == 'mean' or baseline_norm.lower() == 'average':
            metric_data = data[:, start_sample:end_sample] - np.nanmean(baseline, axis=1)[:, None]
        elif baseline_norm.lower() == 'median':
            metric_data = data[:, start_sample:end_sample] - np.nanmedian(baseline, axis=1)[:, None]
        else:
            #TODO:
            pass
        # TODO: check when no normalization to baseline, whether waveform method still works, or should give warning
        # check if data by ref
        # if config('trials', 'baseline_norm') == "None"


        # normalize (L2 norm) each trial
        norm_matrix = np.sqrt(np.power(metric_data, 2).sum(axis=1))
        norm_matrix[norm_matrix == 0] = np.nan                          # prevent division by 0
        norm_metric_data = metric_data / norm_matrix[:, None]

        # calculate internal projections
        proj = np.matmul(norm_metric_data, np.transpose(metric_data))

        # For the t-test each trial is represented half of the time as the normalized projected and half as un-normalized projected
        # Ref: Miller, K. J., Müller, K. R., & Hermes, D. (2021). Basis profile curve identification to understand electrical stimulation effects in human brain networks. PLoS computational biology, 17(9)
        test_values = np.array([])
        for diag_index in range(2, proj.shape[0], 2):
            test_values = np.append(test_values, np.diag(proj, diag_index))

        for diag_index in range(1, proj.shape[0], 2):
            test_values = np.append(test_values, np.diag(proj, -diag_index))

        # perform a one-sample t-test
        test_result = stats.ttest_1samp(test_values, 0, alternative='greater')

        # return the t-statistic, df and p-value as metrics
        return np.array([test_result.statistic, test_result.df, test_result.pvalue], dtype=np.float64)


    @staticmethod
    def evaluate_callback(channel_index, stimpair_index, metric_values):

        # (double) The threshold which needs to be exceeded to detect a peak
        cross_proj_threshold = config('detection', 'cross_proj', 'threshold')

        if metric_values[channel_index, stimpair_index, 0] == np.nan:
            return False
        return metric_values[channel_index, stimpair_index, 0] > cross_proj_threshold

    @staticmethod
    def append_output_dict_callback(output_dict, metric_values):

        if not 'metrics' in output_dict.keys():
            output_dict['metrics'] = dict()

        output_dict['metrics']['cross_proj_t'] = np.array(metric_values[:,:,0].tolist())
        output_dict['metrics']['cross_proj_df'] = np.array(metric_values[:,:,1].tolist())
        output_dict['metrics']['cross_proj_p'] = np.array(metric_values[:,:,2].tolist())
