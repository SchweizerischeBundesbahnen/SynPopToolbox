"""Parse SynPop from raw CSV files.

All conventions are set in config/synpop_loading_config.json. This
config file can be modified to suite other pre-processing needs.
"""
import logging
import os
from pathlib import Path
from typing import Optional
from typing import Union

import geopandas as gpd
import pandas as pd

from synpop import NEW_CATEGORIES
from synpop.config import SwissCantons
from synpop.config import SynPopConfig
from synpop.utils import SynPopException
from synpop.visualization.zone_maps import mobi_zones_path

# Logger settings set in __init__.py
logger = logging.getLogger(__name__)


class SynPopPreprocessor:
    """SynPopPreprocessor."""

    def __init__(
        self,
        raw_data_dir: Union[Path, str],
        output_dir: Optional[str] = None,
        year: int = 2017,
        config_path: Optional[Union[Path, str]] = None,
    ):
        """SynPopPreprocessor.

        :param raw_data_dir: directory where the original CSV files are located.
        :param output_dir: directory in which to save the pre-processed
            and compressed persons DataFrame.
        :param mobi_zones_shapefile_path: the path a a shapefile containing
            all mobi-zones with geographical attributes.
        :param year: SynPop year.
        """
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.year = year

        self.synpop_config = SynPopConfig(
            config_path
        )  # all pre-processing steps are set in config

        if self.output_dir is not None:
            # Creating the output directory if id does now yet exist:
            try:
                os.mkdir(self.output_dir)
            except FileExistsError:
                pass

    def preprocess_persons_and_households(  # pylint: disable=too-many-statements
        self,
        nrows: Optional[int] = None,
        fix_locationid: bool = False,
    ):
        """Preprocess persons and households jointly.

        This method loads persons.csv, enriches it by joining to
        mobi_zones and households (to get zone_id). It then runs all the
        pre-processing steps set up in the config file
        "synpop_loading_config.json".

        This function may take up to 15 minute per SynPop to run!

        :param nrows: Set nrows to speed up testing & debugging
        :param fix_locationid: Older versions of the synpop have two
            extra digits in their locationid
        """
        # Loading MOBI zones
        mobi_zones = gpd.read_file(mobi_zones_path(), driver="GPKG").set_index(
            "zone_id"
        )
        if "zones" in self.synpop_config.features_to_load.keys():
            mobi_zones = mobi_zones[self.synpop_config.features_to_load["zones"]]

        canton_abbreviations = SwissCantons().abbreviations_to_full_names
        mobi_zones["KT_full"] = mobi_zones["kt_name"].map(canton_abbreviations)

        persons = self.load_synpop_file(file_name="persons", nrows=nrows)

        # Replace dbirth with year_of_birth and age
        persons["year_of_birth"] = persons["dbirth"].dt.year
        persons["age"] = self.year - persons["year_of_birth"]
        persons = persons.drop("dbirth", axis=1)
        logger.debug(
            '"dbirth" in the "persons" table has been replaced by "year_of_birth" and "age".'
        )

        # Expanding mobility tools
        if "mobility" in persons.columns:
            persons["has_ga"] = persons["mobility"].isin(["ga", "car & ga"])
            persons["has_ht"] = persons["mobility"].isin(
                ["ht", "va & ht", "car & ht", "car & va & ht"]
            )
            persons["has_va"] = persons["mobility"].isin(
                ["va", "va & ht", "car & va", "car & va & ht"]
            )
            persons["car_available"] = persons["mobility"].isin(
                ["car", "car & ht", "car & ga", "car & va", "car & va & ht"]
            )

        # Define variable "is_swiss" and "analysis_subpopulation"
        persons["is_swiss"] = persons["nation"] == "swiss"
        persons["analysis_subpopulation"] = "regular"

        # Defining more derived variables and checking level_of_employment constraint
        if "level_of_employment" in persons.columns:
            overworking = persons["level_of_employment"] > 100
            if overworking.any():
                logger.warning(
                    "%d people have level_of_employment over 100 and were reduced to 100.",
                    overworking.sum(),
                )
                persons.loc[overworking, "level_of_employment"] = 100
            persons["is_employed"] = persons["level_of_employment"] > 0

            # Level of employment group
            persons["loe_group"] = "part-time"
            persons["loe_group"] = persons["loe_group"].mask(
                persons["level_of_employment"] == 0, "non-employed"
            )
            persons["loe_group"] = persons["loe_group"].mask(
                persons["level_of_employment"] == 100, "full-time"
            )
            persons["loe_group"] = pd.Categorical(
                persons["loe_group"],
                ordered=True,
                categories=["non-employed", "part-time", "full-time"],
            )

        # Defining 'current_edu'
        if ("position_in_bus" in persons.columns) and (
            "position_in_edu" in persons.columns
        ):
            persons["current_edu"] = "null"
            persons["current_edu"] = persons["current_edu"].mask(
                ((persons["position_in_edu"] == "pupil") & (persons["age"] <= 5)),
                "kindergarten",
            )
            persons["current_edu"] = persons["current_edu"].mask(
                (
                    (persons["position_in_edu"] == "pupil")
                    & (persons["age"] > 5)
                    & (persons["age"] <= 12)
                ),
                "pupil_primary",
            )
            persons["current_edu"] = persons["current_edu"].mask(
                ((persons["position_in_edu"] == "pupil") & (persons["age"] > 12)),
                "pupil_secondary",
            )
            persons["current_edu"] = persons["current_edu"].mask(
                persons["position_in_edu"] == "student", "student"
            )
            persons["current_edu"] = persons["current_edu"].mask(
                (persons["position_in_bus"] == "apprentice"), "apprentice"
            )
            persons["current_edu"] = pd.Categorical(
                persons["current_edu"],
                ordered=True,
                categories=NEW_CATEGORIES["current_edu"],
            )

        # Defining 'current_job_rank'
        if "position_in_bus" in persons.columns:
            persons["current_job_rank"] = "null"
            persons["current_job_rank"] = persons["current_job_rank"].mask(
                persons["position_in_bus"] == "apprentice", "apprentice"
            )
            persons["current_job_rank"] = persons["current_job_rank"].mask(
                persons["position_in_bus"] == "employee", "employee"
            )
            persons["current_job_rank"] = persons["current_job_rank"].mask(
                persons["position_in_bus"].isin(
                    ["ceo", "bus_management", "management"]
                ),
                "management",
            )

            persons["current_job_rank"] = pd.Categorical(
                persons["current_job_rank"],
                ordered=True,
                categories=NEW_CATEGORIES["current_job_rank"],
            )

            # Additionally define "is_apprentice"
            persons["is_apprentice"] = persons["current_edu"] == "apprentice"

        # Load households
        households = self.load_synpop_file(file_name="households", nrows=None)

        # Fix zone_id in 2016 table
        if fix_locationid:
            # The two additional id digits have to be removed
            households["zone_id"] = households["zone_id"].apply(
                lambda x: int(str(x)[:-2])
            )

        households_with_persons = households.query(
            "household_id in @persons.household_id"
        )
        if len(households_with_persons) < len(households):
            logger.warning(
                "Dropping %d households without any persons.",
                (len(households) - len(households_with_persons)),
            )
            households = households_with_persons.copy()

        # Joining households to geographical zones
        logger.info("Joining households to mobi_zones...")
        households_with_geography = households.join(mobi_zones, on="zone_id")

        # Joining persons to households_with_geography
        logger.info("Joining persons to households...")
        persons = persons.join(households_with_geography, on="household_id")

        # Sanity check
        missing_geographies = persons["kt_id"].isnull().sum()
        if missing_geographies > 0:
            logger.warning(
                "%d persons have not been joined to a mobi-zone!", missing_geographies
            )

        return persons, households_with_geography

    def preprocess_businesses(
        self, nrows: Optional[int] = None, fix_locationid: bool = False
    ):
        """Preprocess businesses.

        This method loads businesses.csv, enriches it by joining to
        mobi_zones and then runs all the pre-processing steps set up in
        the config file "synpop_loading_config.json".

        :param nrows: Set nrows to speed up testing & debugging
        :param fix_locationid: Older versions of the synpop have two
            extra digits in their locationid
        """
        # Loading MOBI zones
        mobi_zones = gpd.read_file(mobi_zones_path(), driver="GPKG").set_index(
            "zone_id"
        )
        if "zones" in self.synpop_config.features_to_load.keys():
            mobi_zones = mobi_zones[self.synpop_config.features_to_load["zones"]]

        canton_abbreviations = SwissCantons().abbreviations_to_full_names
        mobi_zones["KT_full"] = mobi_zones["kt_name"].map(canton_abbreviations)

        businesses = self.load_synpop_file(file_name="businesses", nrows=nrows)

        # Fix zone_id in 2016 table
        if fix_locationid:
            # The two additional id digits have to be removed
            businesses["zone_id"] = businesses["zone_id"].apply(
                lambda x: int(str(x)[:-2])
            )

        # Joining businesses to geographical zones
        logger.info("Joining households to mobi_zones...")
        businesses = businesses.join(mobi_zones, on="zone_id")

        # Final tweaks
        businesses["noga_code"] = businesses["noga_code"].fillna(-1).astype(int)

        # Sanity check
        missing_geographies = businesses["kt_id"].isnull().sum()
        if missing_geographies > 0:
            logger.warning(
                "%d businesses have not been joined to a mobi-zone!",
                missing_geographies,
            )

        return businesses

    def load_synpop_file(self, file_name: str, nrows: Optional[int] = None):
        """Load single SynPop Raw CSV."""
        features_to_load = self.synpop_config.features_to_load[file_name]
        dataframe = pd.read_csv(
            os.path.join(self.raw_data_dir, f"{file_name}.csv"),
            sep=";",
            usecols=features_to_load,
            nrows=nrows,
            index_col=self.synpop_config.index_column[file_name],
        )

        logger.info(
            "The following columns of %s-%d have been loaded: %s",
            file_name,
            self.year,
            features_to_load,
        )
        # Renaming column names
        try:
            dataframe = dataframe.rename(
                columns=self.synpop_config.features_to_rename[file_name]
            )
            logger.debug(
                'Columns renamed: "%s".',
                self.synpop_config.features_to_rename[file_name],
            )
        except KeyError:
            logger.debug('No columns to rename in table: "%s". Moving on...', file_name)

        # Renaming category names
        try:
            for feature_name, key_map in self.synpop_config.categorical_keys[
                file_name
            ].items():
                remaped = dataframe[feature_name].map(key_map).astype("category")
                if remaped.isnull().any():
                    raise SynPopException(
                        f'Column: "{feature_name}" has a category number that is not mapped to a '
                        f"category name ({dataframe[feature_name].loc[remaped.isna()].unique()}). key_map={key_map}. "
                        'This must be fixed in "synpop_loading_config.json"!'
                    )
                dataframe[feature_name] = remaped
            logger.debug(
                'Categories renamed in columns: "%s"',
                self.synpop_config.categorical_keys[file_name].keys(),
            )

        except KeyError:
            logger.debug(
                'No category to rename in table: "%s". Moving on...', file_name
            )

        # Cast date into pandas datetime format
        try:
            for feature_name, date_format in self.synpop_config.date_features[
                file_name
            ].items():
                dataframe[feature_name] = pd.to_datetime(
                    dataframe[feature_name], format=date_format
                )
        except KeyError:
            logger.debug(
                'No date column to rename in table: "%s". Moving on...', file_name
            )

        return dataframe
