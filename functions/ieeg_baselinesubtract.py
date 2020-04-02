"""
Function file for 'ieeg_baselinesubtract'
=====================================================
Subtracts the median or mean signal during baseline-samples from each epoch.


Copyright 2020, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)
Adapted from Dora Hermes, Multimodal Neuroimaging Lab, 2020

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import warnings
import sys
import numpy as np

def ieeg_baselinesubtract(data, baseline):
    """
    Subtracts the median or mean signal during baseline-samples from each epoch.

    Parameters:
    data (ndarray):                 multidimensional array with the average signal per electrode and stimulus-pair.
                                    (matrix format: electrodes x stimulation-pairs x time)
    baseline (tuple or ndarray):    samples to serve as baseline (typically you want to select part of the averaged
                                    signal before stimulation). Either pass a 'tuple' with two values that indicate the
                                    start and end of the range, or pass a one-dimensional 'ndarray' of the same length
                                    as the data's time dimension with binary values to indicate which samples to use.

    Returns:
        int: Description of return value

    """

    #
    # data parameter
    #

    # TODO:

    num_samples = data.shape[2]

    #
    # baseline input parameter
    #

    if isinstance(baseline, (list, tuple)):
        baseline = np.array(baseline)
    if isinstance(baseline, tuple):
        if not len(baseline) == 2 or baseline[1] < baseline[0]:
            print("Error: " + os.path.basename(__file__) + " - invalid baseline parameter", file=sys.stderr)
            return
        if baseline[0] < 0 or baseline[1] < 0 or baseline[0] >= num_samples or baseline[1] >= num_samples:
            print("Error: " + os.path.basename(__file__) + " - invalid baseline parameter(s), parameter out of range (data samples/time dimension)", file=sys.stderr)
            return

    elif isinstance(baseline, np.ndarray):
        if not baseline.ndim == 1 or not len(baseline) == num_samples:
            print("Error: " + os.path.basename(__file__) + " - invalid baseline parameter, ndarray does not match the data samples/time dimension", file=sys.stderr)
            return

    else:
        print("Error: " + os.path.basename(__file__) + " - invalid baseline parameter", file=sys.stderr)
        return


    # for every electrode
    for iElec in range(data.shape[0]):

        # for every stimulation-pair
        for iPair in range(data.shape[1]):

            # calculate the std of the baseline samples
            warnings.filterwarnings('error')
            try:
                if isinstance(baseline, tuple):
                    baseline_std = np.nanstd(data[iElec, iPair, baseline[0]:baseline[1]])
                else:
                    baseline_std = np.nanstd(data[iElec, iPair, baseline == 1])
            except Warning:
                # assume because of nans; which is often the case when the stimulated electrodes are
                # nan-ed out on the electrode dimensions, just continue to next
                continue



