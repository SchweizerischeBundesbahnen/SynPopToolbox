"""
This module contains utils that can be used from a notebook or another module.
"""
import logging


# New logger: Change level between INFO or DEBUG to have more or less information
import os

import pandas

# Logger settings set in __init__.py
logger = logging.getLogger(__name__)


def bin_variable(raw_values, interval_size=5, max_regular_intervals=100, last_bin_name=None):
    if not last_bin_name:
        last_bin_name = '>{}'.format(max_regular_intervals)

    bin_names = ['{0}-{1}'.format(i, i + interval_size - 1) for i in range(0, max_regular_intervals, interval_size)]
    bin_names += [last_bin_name, ]  # last bin has anything above "max_regular_intervals"
    max_bound = max_regular_intervals * 1000  # a crazy number that will never be exceeded
    bin_edges = list(range(0, max_regular_intervals + 1, interval_size)) + [max_bound, ]

    binned_values = pandas.cut(raw_values, bin_edges, right=False, labels=bin_names)

    assert not binned_values.isnull().any(), 'Some values have not been binned correctly (Null found!)'

    return binned_values


def create_dir(output_dir):
    # Making sur the output directory exists
    try:
        os.makedirs(output_dir)
        logger.info('"{}" created'.format(output_dir))
    except FileExistsError:
        logger.info('"{}" exists already'.format(output_dir))
