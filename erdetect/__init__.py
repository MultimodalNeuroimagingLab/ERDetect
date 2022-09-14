import logging
import sys
from erdetect.utils.misc import CustomLoggingFormatter
from erdetect.version import __version__
from erdetect._erdetect import process
__all__ = ['process', '__version__']

if sys.version_info < (3, 8, 0):
    sys.exit("Python 3.8 or later is required.")

#
# logging
#

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger_ch = logging.StreamHandler(stream=sys.stdout)
logger_ch.setFormatter(CustomLoggingFormatter())
logger.addHandler(logger_ch)
