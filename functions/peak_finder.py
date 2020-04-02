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
import os
import sys
import numpy as np
import scipy.io as sio

def peak_finder(data, sel=None, thresh=None, extrema=1, include_endpoints=True, interpolate=False):

    #
    # input parameters
    #

    # data parameter
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    if not isinstance(data, np.ndarray) or not data.ndim == 1 or len(data) < 2:
        print("Error: " + os.path.basename(__file__) + " - The data must be a one-dimensional array (list, tuple, ndarray) of at least 2 values", file=sys.stderr)
        return
    if np.any(~np.isreal(data)):
        print("Warning: " + os.path.basename(__file__) + " - Absolute values of data will be used", file=sys.stdout)
        data = np.abs(data)

    # selection parameter
    if sel is None:
        sel = (np.nanmax(data) - np.nanmin(data)) / 4
    else:
        try:
            float(sel)
        except:
            print("Warning: " + os.path.basename(__file__) + " - The selectivity must be a real scalar. A selectivity of %.4g will be used", file=sys.stdout)
            sel = (np.nanmax(data) - np.nanmin(data)) / 4

    # threshold parameter
    if not thresh is None:
        try:
            float(thresh)
        except:
            print("Warning: " + os.path.basename(__file__) + " - The threshold must be a real scalar. No threshold will be used", file=sys.stdout)
            thresh = None

    # extrema parameter
    if not extrema == -1 and not extrema == 1:
        print("Error: " + os.path.basename(__file__) + " - Either 1 (for maxima) or -1 (for minima) must be input for extrema", file=sys.stderr)
        exit()  # return

    # include endpoints parameter
    if not isinstance(include_endpoints, bool):
        print("Error: " + os.path.basename(__file__) + " - Either True or False must be input for include_endpoints", file=sys.stderr)
        exit()  # return

    if not isinstance(interpolate, bool):
        print("Error: " + os.path.basename(__file__) + " - Either True or False must be input for interpolate", file=sys.stderr)
        exit()  # return

    #
    #
    #

    # if needed, flip the data and threshold, so we are finding maxima regardless
    if extrema < 0:
        data = data * extrema

        # adjust threshold according to extrema
        if not thresh is None:
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
        minMag = x.min()
        leftMin = minMag
    else:
        x = data[ind]
        minMag = x.min()
        leftMin = np.min((x[0], data[0]))

    # x only has the peaks, valleys, and possibly endpoints
    lenx = len(x)

    if lenx > 2:
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
                    lenx -= 1

            else:
                # First point is smaller than the second

                if signDx[0] == signDx[1]:
                    # want alternating signs
                    x = np.delete(x, 0)
                    ind = np.delete(ind, 0)
                    lenx -= 1


        # set initial parameters for loop
        tempMag = minMag
        foundPeak = False
        peakLoc = list()
        peakMag = list()

        # Skip the first point if it is smaller so we always start on a maxima
        ii = -1 if x[0] >= x[1] else 0

        # Loop through extrema which should be peaks and then valleys
        while ii < lenx - 1:
            ii += 1     # This is a peak

            # reset peak finding if we had a peak and the next peak is bigger
            # than the last or the left min was small enough to reset.
            if foundPeak:
                tempMag = minMag
                foundPeak = False


            # Found new peak that was lower than temp mag and selectivity larger than the minimum to its left.
            if x[ii] > tempMag and x[ii] > leftMin + sel:
                tempLoc = ii
                tempMag = x[ii]

            # Make sure we don't iterate past the length of our vector
            if ii == lenx - 1:
                break       # We assign the last point differently out of the loop

            ii += 1         # Move onto the valley
            # Come down at least sel from peak
            if not foundPeak and tempMag > sel + x[ii]:
                foundPeak = True    # We have found a peak
                leftMin = x[ii]
                peakLoc.append(tempLoc)     # Add peak to index
                peakMag.append(tempMag)
            elif x[ii] < leftMin:
                # New left minima
                leftMin = x[ii]


        # Check end point
        if include_endpoints:
            if x[-1] > tempMag and x[-1] > leftMin + sel:
                peakLoc.append(lenx - 1)
                peakMag.append(x[-1])
            elif not foundPeak and tempMag > minMag:  # Check if we still need to add the last point
                peakLoc.append(tempLoc)
                peakMag.append(tempMag)
        elif not foundPeak:
            if x[-1] > tempMag and x[-1] > leftMin + sel:
                peakLoc.append(lenx - 1)
                peakMag.append(x[-1])
            elif tempMag > np.min((data[-1], x[-1])) + sel:
                peakLoc.append(tempLoc)
                peakMag.append(tempMag)

        # Create output
        if len(peakLoc) > 0:
            peakInds = np.array(ind[peakLoc])
            peakMags = np.array(peakMag)
        else:
            peakInds = None
            peakMags = None

    else:
        # This is a monotone function where an endpoint is the only peak
        peakMag = x.max()
        xInd = np.where(x == peakMag)[0]
        if include_endpoints and peakMag > minMag + sel:
            peakInds = np.array(ind[xInd])
            peakMags = np.array(x[xInd])
        else:
            peakInds = None
            peakMags = None


    # apply threshold value
    # since always finding maxima it will always be larger than the thresh
    if not thresh is None:
        m = np.where(peakMags > thresh)[0]
        peakInds = np.array(peakInds[m])
        peakMags = np.array(peakMags[m])

    # interpolate
    if interpolate and not peakMags is None:
        middleMask = (peakInds > 0) & (peakInds < len0 - 1)
        noEnds = peakInds[middleMask]

        magDiff = data[noEnds + 1] - data[noEnds - 1]
        magSum = data[noEnds - 1] + data[noEnds + 1]  - 2 * data[noEnds]
        magRatio = magDiff / magSum

        peakInds = peakInds.astype(float)
        peakInds[middleMask] = peakInds[middleMask] - magRatio / 2
        peakMags[middleMask] = peakMags[middleMask] - magRatio * magDiff / 8

    # Change sign of data if was finding minima
    if extrema < 0:
        peakMags = -peakMags

    # return the peak indices and magnitudes
    return (peakInds, peakMags)
