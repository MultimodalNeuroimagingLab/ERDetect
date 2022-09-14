import logging
import sys
from erdetect.utils.misc import CustomLoggingFormatter
from erdetect.version import __version__
from erdetect import main_cli as __main__
from erdetect._erdetect import process
__all__ = ['process', '__version__']

#
# logging
#

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger_ch = logging.StreamHandler(stream=sys.stdout)
logger_ch.setFormatter(CustomLoggingFormatter())
logger.addHandler(logger_ch)
