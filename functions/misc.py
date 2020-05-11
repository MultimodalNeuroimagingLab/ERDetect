"""
Miscellaneous functions and classes
=====================================================
A variety of helper functions and classes


Copyright 2020, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import logging
import numpy as np
import subprocess
from psutil import virtual_memory
from math import ceil


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
        logging.error('Not enough memory available to create array.\nAt least ' + str(int((mem.used + data_bytes_needed) / (1024.0 ** 2))) + ' MB is needed, most likely more.\n(for docker users: extend the memory resources available to the docker service)')
        return None


def is_number(value):
    try:
        float(value)
        return True
    except:
        return False


def is_valid_numeric_range(value):
    """
    Check if the given value is a valid range; a tuple or list with two numeric values

    Args:
        value (tuple or list):  The input value to check

    Returns:
        True is valid range, false if not
    """
    if not isinstance(value, (list, tuple)):
        return False
    if not len(value) == 2:
        return False
    if not is_number(value[0]):
        return False
    if not is_number(value[1]):
        return False
    return True


def number_to_padded_string(value, width=0, pos_space=True):
    """
    Convert a number to a space padded string

    Args:
        value (int or float):   The value to convert to a fixed width string
        width (int):            The total length of the return string; < 0 is pad left; > 0 is pad right
        pos_space (bool):       Flag whether a space-character should be added before positive numbers

    """
    padded_str = ' ' if (pos_space and value >= 0) else ''
    padded_str += str(value)
    if width < 0:
        padded_str = padded_str.rjust(width * -1, ' ')
    elif width > 0:
        padded_str = padded_str.ljust(width, ' ')
    return padded_str


def numbers_to_padded_string(values, width=0, pos_space=True, separator=', '):
    """
    Convert multiple numbers to fixed width string with space padding in the middle

    Args:
        value (tuple or list):  The values that will be converted into a fixed width string
        width (int):            The total length of the return string
        pos_space (bool):       Flag whether a space-character should be added before positive numbers
        separator (string):     Separator string after each value

    """
    if len(values) == 0:
        return ''

    padded_values = []
    total_value_width = 0
    for value in values:
        padded_values.append(number_to_padded_string(value, 0, pos_space))
        total_value_width += len(padded_values[-1])

    padded_str = padded_values[0]

    if len(values) == 1:
        return padded_values[0].ljust(width, ' ')

    sep_width = (width - total_value_width - ((len(values) - 1) * len(separator))) / (len(values) - 1)
    if sep_width < 1:
        sep_width = 1
    else:
        sep_width = ceil(sep_width)

    for iValue in range(1,len(padded_values)):
        padded_str += separator
        if len(padded_str) + sep_width + len(padded_values[iValue]) > width:
            padded_str += ''.ljust(width - len(padded_str) - len(padded_values[iValue]), ' ')
        else:
            padded_str += ''.ljust(sep_width, ' ')
        padded_str += padded_values[iValue]

    return padded_str


def run_cmd(command, env={}):
    merged_env = os.environ
    merged_env.update(env)
    merged_env.pop('DEBUG', None)
    process = subprocess.run(command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             shell=True,
                             universal_newlines=True,
                             env=merged_env,
                             encoding='utf-8')
    return process


def multi_line_list(input_array, indent_length=45, first_line_caption='', items_per_line=4, item_delimiter=' ', first_line_single_item=None):
    """
    Convert a array of strings (list or tuple) to a formatted string where the items are
    equally spread over multiple lines, instead of one long line

    Args:
        input_array (arroy of str):    String to format as text
        indent_length (int):           The spacing in front of the listed items
        first_line_caption (str):      The text in the first line that precedes the listing of the items in the array.
        items_per_line (int):          The number of items per line
        item_delimiter (str):          The str that is added between the array-items in the text
        first_line_single_item (str):  If set, will display this string on the first line, the rest of the items will
                                       then be shown from the second line on

    """
    current_line = ''
    return_text = ''

    #
    if first_line_single_item is not None:
        return_text = first_line_caption.ljust(indent_length, ' ') + str(first_line_single_item)
    else:
        current_line = first_line_caption

    # generate and print the lines
    sub_line = ''
    for i in range(len(input_array)):
        if not len(sub_line) == 0:
            sub_line += item_delimiter
        sub_line += str(input_array[i])
        if i > 0 and (i + 1) % items_per_line == 0:
            if not len(return_text) == 0:
                return_text += '\n'
            return_text += current_line.ljust(indent_length, ' ') + sub_line
            sub_line = ''
            current_line = ''

    # print the remaining items (if there are any)
    if not len(sub_line) == 0:
        if not len(return_text) == 0:
            return_text += '\n'
        return_text += current_line.ljust(indent_length, ' ') + sub_line

    return return_text


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
        color (str):        text color (as ASCII code)

        Source: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=print_end)
    if iteration == total:
        print()

#
class CustomLoggingFormatter(logging.Formatter):
    black = "\033[0;30m"
    grey = "\033[0;37m"
    blue = "\033[0;34m"
    green = "\033[0;32m"
    red = "\033[0;31m"
    yellow = "\033[0;33;21m"
    reset = "\033[0m"

    FORMATS = {
        logging.ERROR: red + "ERROR: %(filename)s: %(message)s" + reset,
        logging.WARNING: red + "WARNING: %(message)s" + reset,
        logging.INFO: "%(message)s",                     # TODO: currently no explicit color set, breaks 'print_progressbar'
        logging.DEBUG: "DEBUG: %(message)s",
        "DEFAULT": "%(message)s"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
