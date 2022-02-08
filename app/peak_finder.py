"""
Function file for 'peak_finder'
=====================================================
Noise tolerant fast peak finding algorithm.


Copyright 2020, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)
Adapted from Nathanael Yoder ("peakfinder", MATLAB Central File Exchange, 2016)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import logging
import numpy as np


def peak_finder(data, sel=None, thresh=None, extrema=1, include_endpoints=True, interpolate=False):

    #
    # input parameters
    #

    # data parameter
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    if not isinstance(data, np.ndarray) or not data.ndim == 1 or len(data) < 2:
        logging.error('The input data must be a one-dimensional array (list, tuple, ndarray) of at least 2 values')
        return None, None
    if np.any(~np.isreal(data)):
        logging.warning('Absolute values of data will be used')
        data = np.abs(data)

    # selection parameter
    if sel is None:
        sel = (np.nanmax(data) - np.nanmin(data)) / 4
    else:
        try:
            float(sel)
        except:
            logging.warning('The selectivity must be a real scalar. A selectivity of %.4g will be used')
            sel = (np.nanmax(data) - np.nanmin(data)) / 4

    # threshold parameter
    if thresh is not None:
        try:
            float(thresh)
        except:
            logging.warning('The threshold must be a real scalar. No threshold will be used')
            thresh = None

    # extrema parameter
    if not extrema == -1 and not extrema == 1:
        logging.error('The \'extrema\' argument has to be either 1 (for maxima) or -1 (for minima)')
        return None, None

    # include endpoints parameter
    if not isinstance(include_endpoints, bool):
        logging.error('The \'include_endpoints\' argument should be a boolean value')
        return None, None

    if not isinstance(interpolate, bool):
        logging.error('The \'interpolate\' argument should be a boolean value')
        return None, None

    #
    #
    #

    # if needed, flip the data and threshold, so we are finding maxima regardless
    if extrema < 0:
        data = data * extrema

        # adjust threshold according to extrema
        if thresh is not None:
            thresh = thresh * extrema

    # retrieve the number of data points
    len0 = len(data)

    # find derivative
    dx0 = data[1:] - data[0:-1]
    eps = np.spacing(1)
    dx0[dx0 == 0] = -eps             # This is so we find the first of repeated values

    # find where the derivative changes sign
    ind = np.where(dx0[0:-1] * dx0[1:] < 0)[0] + 1

    # include endpoints in potential peaks and valleys as desired
    if include_endpoints:
        x = np.concatenate(([data[0]], data[ind], [data[-1]]))
        ind = np.concatenate(([0], ind, [len0 - 1]))
        min_mag = x.min()
        left_min = min_mag
    else:
        x = data[ind]
        min_mag = x.min()
        left_min = np.min((x[0], data[0]))

    # x only has the peaks, valleys, and possibly endpoints
    len_x = len(x)

    if len_x > 2:
        # Function with peaks and valleys

        if include_endpoints:
            # Deal with first point a little differently since tacked it on

            # Calculate the sign of the derivative since we tacked the first
            #  point on it does not necessarily alternate like the rest.
            signDx = np.sign(x[1:3] - x[0:2])
            if signDx[0] <= 0:
                # The first point is larger or equal to the second

                if signDx[0] == signDx[1]:
                    # Want alternating signs
                    x = np.delete(x, 1)
                    ind = np.delete(ind, 1)
                    len_x -= 1

            else:
                # First point is smaller than the second

                if signDx[0] == signDx[1]:
                    # want alternating signs
                    x = np.delete(x, 0)
                    ind = np.delete(ind, 0)
                    len_x -= 1


        # set initial parameters for loop
        temp_mag = min_mag
        found_peak = False
        peak_loc = list()
        peak_mag = list()

        # Skip the first point if it is smaller so we always start on a maxima
        ii = -1 if x[0] >= x[1] else 0

        # Loop through extrema which should be peaks and then valleys
        while ii < len_x - 1:
            ii += 1     # This is a peak

            # reset peak finding if we had a peak and the next peak is bigger
            # than the last or the left min was small enough to reset.
            if found_peak:
                temp_mag = min_mag
                found_peak = False


            # Found new peak that was lower than temp mag and selectivity larger than the minimum to its left.
            if x[ii] > temp_mag and x[ii] > left_min + sel:
                temp_loc = ii
                temp_mag = x[ii]

            # Make sure we don't iterate past the length of our vector
            if ii == len_x - 1:
                break       # We assign the last point differently out of the loop

            ii += 1         # Move onto the valley
            # Come down at least sel from peak
            if not found_peak and temp_mag > sel + x[ii]:
                found_peak = True    # We have found a peak
                left_min = x[ii]
                peak_loc.append(temp_loc)     # Add peak to index
                peak_mag.append(temp_mag)
            elif x[ii] < left_min:
                # New left minima
                left_min = x[ii]


        # Check end point
        if include_endpoints:
            if x[-1] > temp_mag and x[-1] > left_min + sel:
                peak_loc.append(len_x - 1)
                peak_mag.append(x[-1])
            elif not found_peak and temp_mag > min_mag:  # Check if we still need to add the last point
                peak_loc.append(temp_loc)
                peak_mag.append(temp_mag)
        elif not found_peak:
            if x[-1] > temp_mag and x[-1] > left_min + sel:
                peak_loc.append(len_x - 1)
                peak_mag.append(x[-1])
            elif temp_mag > np.min((data[-1], x[-1])) + sel:
                peak_loc.append(temp_loc)
                peak_mag.append(temp_mag)

        # Create output
        if len(peak_loc) > 0:
            peak_inds = np.array(ind[peak_loc])
            peak_mags = np.array(peak_mag)
        else:
            return None, None

    else:
        # This is a monotone function where an endpoint is the only peak
        peak_mag = x.max()
        xInd = np.where(x == peak_mag)[0]
        if include_endpoints and peak_mag > min_mag + sel:
            peak_inds = np.array(ind[xInd])
            peak_mags = np.array(x[xInd])
        else:
            return None, None


    # apply threshold value
    # since always finding maxima it will always be larger than the thresh
    if not thresh is None:
        m = np.where(peak_mags > thresh)[0]
        peak_inds = np.array(peak_inds[m])
        peak_mags = np.array(peak_mags[m])

    # interpolate
    if interpolate and not peak_mags is None:
        middle_mask = (peak_inds > 0) & (peak_inds < len0 - 1)
        no_ends = peak_inds[middle_mask]

        mag_diff = data[no_ends + 1] - data[no_ends - 1]
        mag_sum = data[no_ends - 1] + data[no_ends + 1]  - 2 * data[no_ends]
        mag_ratio = mag_diff / mag_sum

        peak_inds = peak_inds.astype(float)
        peak_inds[middle_mask] = peak_inds[middle_mask] - mag_ratio / 2
        peak_mags[middle_mask] = peak_mags[middle_mask] - mag_ratio * mag_diff / 8

    # Change sign of data if was finding minima
    if extrema < 0:
        peak_mags = -peak_mags

    # return the peak indices and magnitudes
    return peak_inds, peak_mags
