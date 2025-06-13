"""This module contains utils that can be used from a notebook or another module."""
import logging
from pathlib import Path

import pandas


# New logger: Change level between INFO or DEBUG to have more or less information
# Logger settings set in __init__.py
logger = logging.getLogger(__name__)


def bin_variable(
    raw_values, interval_size=5, max_regular_intervals=100, last_bin_name=None
):
    """Bin a variable into regular intervals. Wrapper to pandas' cut."""
    if not last_bin_name:
        last_bin_name = f">{max_regular_intervals}"

    bin_names = [
        f"{i}-{i + interval_size - 1}"
        for i in range(0, max_regular_intervals, interval_size)
    ]
    bin_names += [
        last_bin_name,
    ]  # last bin has anything above "max_regular_intervals"
    max_bound = (
        max_regular_intervals * 1000
    )  # a crazy number that will never be exceeded
    bin_edges = list(range(0, max_regular_intervals + 1, interval_size)) + [
        max_bound,
    ]

    binned_values = pandas.cut(raw_values, bin_edges, right=False, labels=bin_names)

    assert (
        not binned_values.isnull().any()
    ), "Some values have not been binned correctly (Null found!)"

    return binned_values


class SynPopException(Exception):
    """Base class for SynPop exceptions."""


def remove(file):
    """Remove file if it exists."""
    if Path(file).exists():
        Path(file).unlink()
