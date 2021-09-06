"""
This module is the only one to interact with the configuration files.
It offers an interface to the important configurations to the rest of the package.
"""
import json
import os

import numpy as np

module_dir = os.path.dirname(os.path.realpath(__file__))


def load_json(json_file):
    with open(json_file, encoding='utf8') as json_config:
        data = json.load(json_config)
    return data


def reconstruct_integer_key_dict(string_key_dict):
    """
    Dict from json cannot have a integer as the key. This function recreates the dicts with integer keys.
    The convention chosen for np.nan as key is the string: 'NaN'.
    """
    integer_key_dict = dict()
    for key, value in string_key_dict.items():
        if key == 'NaN':
            integer_key_dict[np.nan] = value
        else:
            integer_key_dict[int(key)] = value

    return integer_key_dict


class SwissCantons:
    lib_dir = os.path.dirname(__file__)
    canton_abbreviations_config = os.path.join(lib_dir, 'config/canton_abbreviations.json')

    def __init__(self):
        self.abbreviations_to_full_names = load_json(self.canton_abbreviations_config)
        self.full_names_to_abbreviations = {v: k for k, v in self.abbreviations_to_full_names.items()}

    def all_canton_full_names(self):
        return list(self.full_names_to_abbreviations.keys())


class MSregion:
    def __init__(self):
        lib_dir = os.path.dirname(__file__)
        ms_region_names_config = os.path.join(lib_dir, 'config/ms_region_names.json')
        self.index_to_name_mapping = {int(key): value for key, value in load_json(ms_region_names_config).items()}

        arbeitsmarktregionen_config = os.path.join(lib_dir, 'config/ms_region_to_arbeitsmarktregionen.json')
        self.arbeitsmarktregion_per_ms_region_index = {
            int(key): value for key, value in load_json(arbeitsmarktregionen_config).items()
        }
        self.arbeitsmarktregion_per_ms_region_name = {
            msr_name: self.arbeitsmarktregion_per_ms_region_index[msr_index]
            for msr_index, msr_name in self.index_to_name_mapping.items()
        }


class SynPopConfig:
    lib_dir = os.path.dirname(__file__)
    DEFAULT_CONFIG = os.path.join(lib_dir, 'config/synpop_loading_config.json')

    def __init__(self, config_file=None):
        if config_file is None:
            config_file = self.DEFAULT_CONFIG
        self.config_file = config_file
        self.features_to_load = load_json(self.config_file)['features_to_load']
        self.features_to_rename = load_json(self.config_file)['features_to_rename']
        self.boolean_features = load_json(self.config_file)['boolean_features']
        self.categorical_keys = self._load_categorical_keys()
        self.date_features = load_json(self.config_file)['date_features']

    def _load_categorical_keys(self):
        string_key_categorical_keys = load_json(self.config_file)['categorical_keys']
        integer_key_categorical_keys = dict()
        for file_name, config_data in string_key_categorical_keys.items():
            integer_key_categorical_keys[file_name] = dict()
            for feature_name, key_map in config_data.items():
                integer_key_categorical_keys[file_name][feature_name] = reconstruct_integer_key_dict(key_map)
        return integer_key_categorical_keys


class RandomFittingConfig:
    lib_dir = os.path.dirname(__file__)
    config_file = os.path.join(lib_dir, 'config/random_fitting_config.json')

    def __init__(self):
        self.random_seed = int(load_json(self.config_file)['random_seed'])


if __name__ == '__main__':
    x = MSregion()
    x.arbeitsmarktregion_per_ms_region
