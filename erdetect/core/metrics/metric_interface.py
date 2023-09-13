"""
Metric Interface class

Copyright 2023, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


class MetricInterface:

    @staticmethod
    def process_callback(sampling_rate, data, baseline):
        """
        Callback that calculates the metric for a subset of the data.
        This is called for each (measurement) channel and stimulated-pair combination

        Args:
            sampling_rate (int):                  The sampling rate of the data
            data (ndarray):                       2D data matrix (represented as trials x samples)
            baseline (ndarray):                   2D baseline data matrix (represented as trials x samples)

        Returns:
            One or multiple metric values as a numpy array
        """
        pass

    @staticmethod
    def evaluate_callback(channel_index, stimpair_index, metric_values):
        """
        Callback that evaluates metric values(s) to determine if they constitute an
        evoked response or not. This is called for each measurement channel and stimulated-pair combination

        Args:
            channel_index (int):                  The index of the current channel being evaluated
            stimpair_index (int):                 The index of the current stim-pair being evaluated
            metric_values (ndarray):              Metric data matrix (represented as channels x stim-pairs x <one or more metric values>)

        Returns:
            True to flag an evoked response, False to flag as not an evoked response
        """
        pass

    @staticmethod
    def append_output_dict_callback(output_dict, metric_values):
        """
        Callback that adds the metrics values(s) to the given output dictionary so that
        they will included in the output data file

        Args:
            output_dict (dict):                   Output dictionary to add the metric values to
            metric_values (ndarray):              Metric data matrix (represented as channels x stim-pairs x <one or more metric values>)
        """
        pass

