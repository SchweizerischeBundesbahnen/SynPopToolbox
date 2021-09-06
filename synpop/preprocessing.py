"""
Preprocessing:
    From raw "persons.csv", "households.csv", "businesses.csv", "locations.csv" to cleaned and compressed DataFrames.

All conventions are set in config/synpop_loading_config.json. This file can be modified to suite other pre-processing
 needs.
"""
import logging
import os

import geopandas as gpd
import pandas as pd
from pyarrow import feather

from synpop.config import SynPopConfig, SwissCantons
from synpop import anonymization
from synpop.zone_maps import MOBI_ZONES

# Logger settings set in __init__.py
logger = logging.getLogger(__name__)


class SynPopPreprocessor:
    def __init__(self, raw_data_dir: str, output_dir: str = None, year: int = 2017,
                 mobi_zones_shapefile_path: str = None, config_path=None):
        """
        SynPopPreprocessor

        :param raw_data_dir: directory where the original person.csv, households.csv ... are saved
        :param output_dir: directory in which to save the pre-processed and compressed persons DataFrame
        :param mobi_zones_shapefile_path: the path a a shapefile containing all mobi-zones with geographical attributes
        :param year: SynPop year
        """
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.year = year
        self.mobi_zones_shapefile_path = mobi_zones_shapefile_path

        self.synpop_config = SynPopConfig(config_path)  # all pre-processing steps are set in config

        if self.output_dir is not None:
            # Creating the output directory if id does now yet exist:
            try:
                os.mkdir(self.output_dir)
            except FileExistsError:
                pass

    def preprocess_persons_and_households(self, nrows: int = None, fix_locationid: bool = False,
                                          blur_households: bool = False, hh_min_size_threshold: int = 1):
        """
        This method loads persons.csv, enriches it by joining to mobi_zones and households (to get zone_id). It
        then runs all the pre-processing steps set up in the config file "synpop_loading_config.json".
        The output DataFrames are saved to feather files "persons.feather" and "households.feather".

        This function may take up to 15 minute per SynPop to run!

        :param nrows: Set nrows to speed up testing & debugging
        :param fix_locationid: Older versions of the synpop have two extra digits in their locationid
        :param blur_households:
        """
        # Loading MOBI zones
        zone_features = self.synpop_config.features_to_load['zones']
        mobi_zones = gpd.read_file(MOBI_ZONES)[zone_features]

        canton_abbreviations = SwissCantons().abbreviations_to_full_names
        mobi_zones['KT_full'] = mobi_zones['kt_name'].map(canton_abbreviations)

        persons = self.load_synpop_file(file_name='persons', nrows=nrows)

        # Replace dbirth with year_of_birth and age
        persons['year_of_birth'] = persons['dbirth'].dt.year
        persons['age'] = self.year - persons['year_of_birth']
        persons = persons.drop('dbirth', axis=1)
        logger.debug('"dbirth" in the "persons" table has been replaced by "year_of_birth" and "age".')

        # Expanding mobility tools
        if 'mobility' in persons.columns:
            persons['has_ga'] = persons['mobility'].isin(["ga", "car & ga"])
            persons['has_ht'] = persons['mobility'].isin(["ht", "va & ht", "car & ht", "car & va & ht"])
            persons['has_va'] = persons['mobility'].isin(["va", "va & ht", "car & va", "car & va & ht"])
            persons['car_available'] = (persons['mobility']
                                        .isin(["car",  "car & ht",  "car & ga",  "car & va", "car & va & ht"])
                                        )

        # Define boolean variable "is_swiss"
        persons['is_swiss'] = persons['nation'] == 'swiss'

        # Defining more derived variables and checking level_of_employment constraint
        if 'level_of_employment' in persons.columns:
            overworking = persons['level_of_employment'] > 100
            if overworking.any():
                logger.warning(f"{overworking.sum()} people have level_of_employment over 100 and were reduced to 100.")
                persons.loc[overworking, 'level_of_employment'] = 100
            persons['is_employed'] = persons['level_of_employment'] > 0

            # Level of employment group
            persons['loe_group'] = 'part-time'
            persons['loe_group'] = persons['loe_group'].mask(persons['level_of_employment'] == 0, 'non-employed')
            persons['loe_group'] = persons['loe_group'].mask(persons['level_of_employment'] == 100, 'full-time')
            persons['loe_group'] = pd.Categorical(persons['loe_group'], ordered=True,
                                                  categories=['non-employed', 'part-time', 'full-time'])

        # Defining 'current_edu'
        if ('position_in_bus' in persons.columns) and ('position_in_edu' in persons.columns):
            persons['current_edu'] = 'null'
            persons['current_edu'] = persons['current_edu'].mask(((persons['position_in_edu'] == 'pupil') &
                                                                  (persons['age'] <= 5)), 'kindergarten')
            persons['current_edu'] = persons['current_edu'].mask(((persons['position_in_edu'] == 'pupil') &
                                                                  (persons['age'] > 5) &
                                                                  (persons['age'] <= 12)
                                                                  ), 'pupil_primary')
            persons['current_edu'] = persons['current_edu'].mask(((persons['position_in_edu'] == 'pupil') &
                                                                  (persons['age'] > 12)
                                                                  ), 'pupil_secondary')
            persons['current_edu'] = persons['current_edu'].mask(persons['position_in_edu'] == 'student', 'student')
            persons['current_edu'] = persons['current_edu'].mask((persons['position_in_bus'] == 'apprentice'),
                                                                 'apprentice')

            ordered_cats = ['kindergarten', 'pupil_primary', 'pupil_secondary', 'student', 'apprentice', 'null']
            persons['current_edu'] = pd.Categorical(persons['current_edu'], ordered=True, categories=ordered_cats)

        # Defining 'current_job_rank'
        if 'position_in_bus' in persons.columns:
            persons['current_job_rank'] = 'null'
            persons['current_job_rank'] = persons['current_job_rank'].mask(persons['position_in_bus'] == 'apprentice',
                                                                           'apprentice')
            persons['current_job_rank'] = persons['current_job_rank'].mask(persons['position_in_bus'] == 'employee',
                                                                           'employee')
            persons['current_job_rank'] = persons['current_job_rank'].mask(
                persons['position_in_bus'].isin(['ceo', 'bus_management', 'management']), 'management'
            )

            persons['current_job_rank'] = pd.Categorical(persons['current_job_rank'], ordered=True,
                                                         categories=['apprentice', 'employee', 'management', 'null']
                                                         )

            # Additionally define "is_apprentice"
            persons['is_apprentice'] = persons['current_edu'] == 'apprentice'

        # Load households
        households = self.load_synpop_file(file_name='households', nrows=None)

        # Fix zone_id in 2016 table
        if fix_locationid:
            # The two additional id digits have to be removed
            households['zone_id'] = households['zone_id'].apply(lambda x: int(str(x)[:-2]))

        # Joining households to geographical zones
        logger.info('Joining households to mobi_zones...')
        households_with_geography = pd.merge(households, mobi_zones, how='left', on='zone_id')

        if blur_households:
            # Blurring households per zone by shuffling all the coordinates randomly
            households_with_geography = anonymization.blur_households_per_zone(
                households_with_geography, hh_min_size_threshold=hh_min_size_threshold)

        # Joining persons to households_with_geography
        logger.info('Joining persons to households...')
        persons = pd.merge(persons, households_with_geography, how='left', on='household_id')

        # Sanity check
        missing_geographies = persons['kt_id'].isnull().sum()
        if missing_geographies > 0:
            logger.warning('{} persons have not been joined to a mobi-zone!'.format(missing_geographies))

        if self.output_dir is not None:
            # Saving the pre-processed persons and households DataFrames to feather files
            file_path = os.path.join(self.output_dir, 'persons_{}.feather'.format(self.year))
            feather.write_feather(persons, file_path, compression_level=9, compression='zstd')
            logger.info('Pre-processed "persons" table has been saved to: {}'.format(file_path))

            file_path = os.path.join(self.output_dir, 'households_{}.feather'.format(self.year))
            feather.write_feather(households_with_geography, file_path, compression_level=9, compression='zstd')
            logger.info('Pre-processed "households" table has been saved to: {}'.format(file_path))

        # The main output is the feather file, but the dataframe is also returned for testing & debugging
        return persons, households_with_geography

    def preprocess_businesses(self, nrows: int = None, fix_locationid: bool = False):
        """
        This method loads businesses.csv, enriches it by joining to mobi_zones and then runs all the pre-processing
        steps set up in the config file "synpop_loading_config.json".
        The output DataFrames are saved to feather files "businesses.feather".

        :param nrows: Set nrows to speed up testing & debugging
        :param fix_locationid: Older versions of the synpop have two extra digits in their locationid
        """
        # Loading MOBI zones
        zone_features = self.synpop_config.features_to_load['zones']
        mobi_zones = gpd.read_file(self.mobi_zones_shapefile_path, encoding='utf8')[zone_features]

        canton_abbreviations = SwissCantons().abbreviations_to_full_names
        mobi_zones['KT_full'] = mobi_zones['kt_name'].map(canton_abbreviations)

        businesses = self.load_synpop_file(file_name='businesses', nrows=nrows)
        businesses['year_of_foundation'] = businesses['dfoundation'].dt.year
        businesses = businesses.drop('dfoundation', axis=1)
        logger.debug('"dfoundation" in the "businesses" table has been replaced by "year_of_foundation".')

        # Fix zone_id in 2016 table
        if fix_locationid:
            # The two additional id digits have to be removed
            businesses['zone_id'] = businesses['zone_id'].apply(lambda x: int(str(x)[:-2]))

        # Joining businesses to geographical zones
        logger.info('Joining households to mobi_zones...')
        businesses = pd.merge(businesses, mobi_zones, how='left', on='zone_id')

        # Final tweaks
        businesses['noga_code'] = businesses['noga_code'].fillna(-1).astype(int)

        # Sanity check
        missing_geographies = businesses['kt_id'].isnull().sum()
        if missing_geographies > 0:
            logger.warning('{} businesses have not been joined to a mobi-zone!'.format(missing_geographies))

        if self.output_dir is not None:
            # Saving the pre-processed persons and households DataFrames to feather files
            file_path = os.path.join(self.output_dir, 'businesses_{}.feather'.format(self.year))
            feather.write_feather(businesses, file_path, compression_level=9, compression='zstd')
            logger.info('Pre-processed "businesses" table has been saved to: {}'.format(file_path))

        # The main output is the feather file, but the dataframe is also returned for testing & debugging
        return businesses

    def load_synpop_file(self, file_name: str, nrows: int = None):
        # Loading file
        features_to_load = self.synpop_config.features_to_load[file_name]
        df = pd.read_csv(os.path.join(self.raw_data_dir, '{}.csv'.format(file_name)), sep=';',
                         usecols=features_to_load, nrows=nrows)

        info_msg = 'The following columns of {name}-{year} have been loaded: {columns}'.format(name=file_name,
                                                                                               year=self.year,
                                                                                               columns=features_to_load)
        logger.info(info_msg)
        # Renaming column names
        try:
            df = df.rename(columns=self.synpop_config.features_to_rename[file_name])
            logger.debug('Columns renamed: "{}".'.format(self.synpop_config.features_to_rename[file_name]))
        except KeyError:
            logger.debug('No columns to rename in table: "{}". Moving on...'.format(file_name))

        # Cast features to boolean (better memory efficiency)
        try:
            boolean_columns = df[self.synpop_config.boolean_features[file_name]].fillna(0).astype(bool)
            df.loc[:, self.synpop_config.boolean_features[file_name]] = boolean_columns
            logger.debug('Columns casted to boolean: "{}".'.format(self.synpop_config.boolean_features[file_name]))

        except KeyError:
            logger.debug('No boolean feature in table: "{}". Moving on...'.format(file_name))

        # Renaming category names
        try:
            for feature_name, key_map in self.synpop_config.categorical_keys[file_name].items():
                df[feature_name] = df[feature_name].map(key_map).astype('category')
                if df[feature_name].isnull().any():
                    raise Exception('Column: "{feature_name}" has a category number that is not mapped to a category '
                                    'name. key_map={key_map}. This must be fixed in "synpop_loading_config.json"!'
                                    .format(feature_name=feature_name,
                                            key_map=key_map)
                                    )

            logger.debug('Categories renamed in columns: "{}"'.format(
                self.synpop_config.categorical_keys[file_name].keys())
            )

        except KeyError:
            logger.debug('No category to rename in table: "{}". Moving on...'.format(file_name))

        # Cast date into pandas datetime format
        try:
            for feature_name, date_format in self.synpop_config.date_features[file_name].items():
                df[feature_name] = pd.to_datetime(df[feature_name], format=date_format)
        except KeyError:
            logger.debug('No date column to rename in table: "{}". Moving on...'.format(file_name))

        return df

