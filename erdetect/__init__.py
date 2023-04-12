# define directory as package

# ensure minimum version
import sys
if sys.version_info < (3, 8, 0):
    sys.exit("Python 3.8 or later is required.")

# flatten access
from ieegprep.utils.console import CustomLoggingFormatter
from erdetect.version import __version__
from erdetect._erdetect import process_subset
from erdetect.views.gui import open_gui
__all__ = ['process_subset', 'open_gui', '__version__']

# logging
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger_ch = logging.StreamHandler(stream=sys.stdout)
logger_ch.setFormatter(CustomLoggingFormatter())
logger.addHandler(logger_ch)
