"""
Miscellaneous functions
=====================================================
A variety of helper functions


Copyright 2020, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys
import numpy as np
from psutil import virtual_memory


def print_progressbar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar

    Args:
        iteration (int):    current iteration (Int)
        total (int):        total iterations (Int)
        prefix (str):       prefix string (Str)
        suffix (str)        suffix string (Str)
        decimals (int)      positive number of decimals in percent complete (Int)
        length (int):       character length of bar (Int)
        fill (str):         bar fill character (Str)
        print_end (str):    end character (e.g. "\r", "\r\n") (Str)

        From: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


def allocate_array(dimensions, fill_value=np.nan, dtype='float64'):
    """
    Create and immediately allocate the memory for an x-dimensional array

    Before allocating the memory, this function checks if is enough memory is available (this is needed since when a
    numpy array is allocated and there is not enough memory, python crashes without the chance to catch an error).

    Args:
        dimensions (int or tuple):
        fill_value (any numeric):
        dtype (str):

    Returns:
        data (ndarray):             An initialized x-dimensional array, or None if insufficient memory available

    """
    # initialize a data buffer (channel x trials/epochs x time)
    try:

        # create a ndarray object (no memory is allocated here)
        data = np.empty(dimensions, dtype=dtype)
        data_bytes_needed = data.nbytes

        # check if there is enough memory available
        mem = virtual_memory()
        if mem.available <= data_bytes_needed:
            raise MemoryError()

        # allocate the memory
        data.fill(fill_value)

        #
        return data

    except MemoryError:
        print('Error: not enough memory available to create array.\nAt least ' + str(int((mem.used + data_bytes_needed) / (1024.0 ** 2))) + ' MB is needed, most likely more.\n(for docker users: extend the memory resources available to the docker service)', file=sys.stderr)
        return None
