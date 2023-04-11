"""
Miscellaneous functions and classes
=====================================================
A variety of helper functions and classes


Copyright 2022, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from ieegprep.utils.misc import is_number


def create_figure(width=500, height=500, on_screen=False):
    """
    Create a figure in memory or on-screen, and resize the figure to a specific resolution

    """

    if on_screen:
        fig = plt.figure()
    else:
        fig = Figure()

    # resize the figure
    dpi = fig.get_dpi()
    fig.set_size_inches(float(width) / float(dpi), float(height) / float(dpi))

    return fig


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
