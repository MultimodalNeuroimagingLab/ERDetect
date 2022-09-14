#!/usr/bin/env python3
"""
Evoked response detection - command-line entry-point
=====================================================
Command-line entry-point script for the automatic detection of evoked responses in CCEP data.


Copyright 2022, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import sys

# add a system path to ensure the absolute imports can be used
if not __package__:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
    if PACKAGE_DIR not in sys.path:
        sys.path.insert(0, PACKAGE_DIR)

#
from erdetect.main_cli import execute
if __name__ == "__main__":
    sys.exit(execute())



