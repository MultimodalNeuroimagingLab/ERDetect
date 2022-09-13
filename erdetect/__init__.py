import logging
import sys
from erdetect.utils.misc import CustomLoggingFormatter
from erdetect.version import __version__
from erdetect._erdetect import process
from erdetect.views.gui import open_gui
__all__ = ['process', 'open_gui', '__version__']

#
# logging
#

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger_ch = logging.StreamHandler(stream=sys.stdout)
logger_ch.setFormatter(CustomLoggingFormatter())
logger.addHandler(logger_ch)
