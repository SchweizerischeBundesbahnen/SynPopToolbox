"""SynPopToolbox - A set of tools to analyse, modify or extend the Swiss synthetic population."""

__version__ = '1'
__author__ = 'Raphael Lüthi'
__all__ = []

import logging

# New logger: Change level between INFO or DEBUG to have more or less information
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')

# Mute other loggers
logging.getLogger('fiona').setLevel(logging.CRITICAL)  # mute logging from geopandas
logging.getLogger('shapely').setLevel(logging.CRITICAL)  # mute logging from geopandas
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)  # mute logging from matplotlib
logging.getLogger('urllib3').setLevel(logging.CRITICAL)  # mute external logger
