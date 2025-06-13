"""This module is the only one to interact with the configuration files.

It offers an interface to the important configurations to the rest of the package.
"""
import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import numpy as np

HERE = Path(__file__).parent


def load_json(json_file: Union[Path, str]) -> Dict[str, Any]:
    """Load a json file and returns a dict."""
    with open(json_file, encoding="utf8") as json_config:
        data = json.load(json_config)
    return data


def reconstruct_integer_key_dict(string_key_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstructs a dict with integer keys from a dict with string keys.

    Dict from json cannot have a integer as the key.
    This function recreates the dicts with integer keys.
    The convention chosen for np.nan as key is the string: 'NaN'.
    """
    integer_key_dict = {}
    for key, value in string_key_dict.items():
        if key == "NaN":
            integer_key_dict[np.nan] = value
        else:
            integer_key_dict[int(key)] = value

    return integer_key_dict


class SwissCantons:
    """Class to map Canton abbreviations and full names based on canton_abbreviations.json."""

    canton_abbreviations_config = (
        HERE / "assets/resources/config/canton_abbreviations.json"
    )

    def __init__(self):  # noqa: D107
        self.abbreviations_to_full_names = load_json(self.canton_abbreviations_config)
        self.full_names_to_abbreviations = {
            v: k for k, v in self.abbreviations_to_full_names.items()
        }

    def all_canton_full_names(self) -> List[str]:
        """Return list of all canton full names."""
        return list(self.full_names_to_abbreviations.keys())

    def all_canton_abbreviations(self) -> List[str]:
        """Return list of all canton abbreviations."""
        return list(self.full_names_to_abbreviations.values())


class MSregion:
    """Class to map MS and AMR region names and indices based on config/ms_region_names.json."""

    def __init__(self):  # noqa: D107
        ms_region_names_config = HERE / "assets/resources/config/ms_region_names.json"
        self.index_to_name_mapping = {
            int(key): value for key, value in load_json(ms_region_names_config).items()
        }

        arbeitsmarktregionen_config = (
            HERE / "assets/resources/config/ms_region_to_arbeitsmarktregionen.json"
        )
        self.arbeitsmarktregion_per_ms_region_index = {
            int(key): value
            for key, value in load_json(arbeitsmarktregionen_config).items()
        }
        self.arbeitsmarktregion_per_ms_region_name = {
            msr_name: self.arbeitsmarktregion_per_ms_region_index[msr_index]
            for msr_index, msr_name in self.index_to_name_mapping.items()
        }


class SynPopConfig:
    """SynPop config for preprocessing."""

    DEFAULT_CONFIG = HERE / "assets/resources/config/synpop_loading_config.json"

    def __init__(self, config_file=None):  # noqa: D107
        if config_file is None:
            config_file = self.DEFAULT_CONFIG
        self.config_file = config_file
        self.features_to_load = load_json(self.config_file)["features_to_load"]
        self.features_to_rename = load_json(self.config_file)["features_to_rename"]
        self.index_column = load_json(self.config_file)["index_column"]
        self.categorical_keys = self._load_categorical_keys()
        self.date_features = load_json(self.config_file)["date_features"]

    def _load_categorical_keys(self):
        string_key_categorical_keys = load_json(self.config_file)["categorical_keys"]
        integer_key_categorical_keys = {}
        for file_name, config_data in string_key_categorical_keys.items():
            integer_key_categorical_keys[file_name] = {}
            for feature_name, key_map in config_data.items():
                integer_key_categorical_keys[file_name][
                    feature_name
                ] = reconstruct_integer_key_dict(key_map)
        return integer_key_categorical_keys
