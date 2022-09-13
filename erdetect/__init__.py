import logging
import sys
from version import __version__
__all__ = ['__version__']
#__all__ = ['process_subset', 'open_gui', '__version__']

#
# logging
#

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger_ch = logging.StreamHandler(stream=sys.stdout)
logger_ch.setFormatter(CustomLoggingFormatter())
logger.addHandler(logger_ch)