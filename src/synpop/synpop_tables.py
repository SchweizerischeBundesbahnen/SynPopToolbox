"""This module contains high level objects to work with the SynPop."""
import contextlib
import json
import logging
import lzma
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy
import pandas as pd
from pyarrow import feather

from synpop import NEW_CATEGORIES
from synpop import validation
from synpop.config import SynPopConfig

# Logger settings set in __init__.py
logger = logging.getLogger(__name__)

CATEGORICAL_COLS = SynPopConfig().categorical_keys
BOOLEAN_COLS = ["is_swiss"]
POSSIBLE_NA_VALUES = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
]


class SynPopTable:
    """Parent class for SynPop Tables: Persons, Households and Businesses."""

    table_name: str
    index_name: str
    total_rows: int
    cols_to_write: List[str]

    # For documentation only
    source_file: Path
    modified_columns: List[str] = []

    def __init__(self, year: int, data: Optional[pd.DataFrame] = None):  # noqa: D107
        self.year = year
        if data is not None:
            self._data = data.copy()
        else:
            self._data = data

    def update(self, data: pd.DataFrame) -> None:
        """Update the data of the table with a new DataFrame."""
        self._data = data
        self.total_rows = self._data.shape[0]

    @property
    def data(self) -> pd.DataFrame:
        """Return the data of the table as a pandas DataFrame."""
        assert self._data is not None
        return self._data

    def load(
        self,
        path: Union[str, Path],
        validate: bool = True,
        force_update_data: bool = False,
    ) -> None:
        """Load a table from a file.

        SQLite, Feather and Pickle (deprecated) are supported.
        """
        if self._data is not None and not force_update_data:
            raise RuntimeError(
                "Data already exists. Re-run with 'force_update_data=True' to force updating."
            )
        path = _resolve_synpop_path(Path(path), self.table_name)
        logger.info("Loading %s from %s ...", self.table_name, path)
        if path.suffixes[0] == ".sqlite":
            data = self.load_from_sqlite_db(path)
        elif path.suffixes[-1] == ".feather":
            data = pd.read_feather(path)
        elif path.suffixes[-2:] == [".pickle", ".gzip"]:
            data = pd.read_pickle(path, compression="gzip")
        elif ".csv" in path.suffixes:
            data = pd.read_csv(
                path, sep=";", keep_default_na=False, na_values=POSSIBLE_NA_VALUES
            )
        else:
            raise IOError(f"Unsupported file format {path.suffixes}.")

        # backwards compatibility: attempt rename columns of old synpops
        data = data.rename(
            errors="ignore",
            columns={
                "location_id": "zone_id",
                "ID": "zone_id",
                "N_KT": "kt_name",
                "KT": "kt_id",
                "N_Gem": "mun_name",
                "ID_Gem": "mun_id",
                "jobs_ch": "jobs_endo",
                "jobs_cb": "jobs_exo",
                "fte_ch": "fte_endo",
                "fte_cb": "fte_exo",
            },
        )

        # avoid strange encoding issues
        for col in data.columns:
            if isinstance(data[col].values[0], bytes):
                data[col] = data[col].str.decode("latin1")

        self.update(data)
        self.source_file = path

        if validate:
            self.validate()

        logger.info("Table %s loaded with %d rows.", self.table_name, self.total_rows)

    def drop_column(self, column_name: Union[List[str], str]) -> None:
        """Drop columns from the table if it exists."""
        try:
            self.update(self.data.drop(column_name, axis=1))
            logger.debug('Columns "%s" has been dropped.', column_name)
        except KeyError:
            logger.warning(
                'No columns "%s" in the %s table. Nothing to drop.',
                column_name,
                self.table_name,
            )

    def validate(self) -> None:
        """Validate the table."""
        logging.info(
            "Validating table %s with %d rows...", self.table_name, len(self.data)
        )
        validation.validate_categories(self)

    def write(self, path: Union[str, Path]) -> None:
        """Write the table to a file (feather format only)."""
        path = Path(path)
        logger.info("Writing to %s ...", path)
        if path.suffixes == [".feather"]:
            feather.write_feather(
                self.data.drop("geometry", axis=1, errors="ignore"),
                path,
                compression_level=9,
                compression="zstd",
            )
        else:
            raise IOError(f"Unsupported file format {path.suffixes}.")
        logger.info(
            "%s table of %d saved to file: %s", self.table_name, self.year, path
        )

    def load_from_sqlite_db(
        self,
        file_path: Path,
        add_zonal_vars: bool = False,
        add_hh_vars: bool = False,
    ) -> pd.DataFrame:
        """Load table from a SQLite database, which may be 7z-compressed."""
        if file_path.suffixes[-1] == ".7z":
            with tempfile.NamedTemporaryFile() as tmp, file_path.open("rb") as file_7z:
                tmp.write(lzma.decompress(file_7z.read()))
                file_path = Path(tmp.name)
        with contextlib.closing(sqlite3.connect(file_path)) as connection:
            dataframe = pd.read_sql_query(
                f"SELECT * FROM {self.table_name}", connection
            )
        if add_zonal_vars:
            zones = pd.read_sql_query("SELECT * FROM zones", connection)
            dataframe = pd.merge(dataframe, zones, on="zone_id")
        if self.table_name == "persons" and add_hh_vars:
            households = pd.read_sql_query("SELECT * FROM households", connection)
            dataframe = pd.merge(
                dataframe, households.drop("zone_id", axis=1), on="household_id"
            )

        # fix column types not handled natively by sqlite
        if dataframe.columns.isin(BOOLEAN_COLS).any():
            for col in BOOLEAN_COLS:
                if col in dataframe.columns:
                    dataframe[col] = dataframe[col].astype(bool)
        if dataframe.columns.isin(CATEGORICAL_COLS.get(self.table_name, [])).any():
            categorical_cols = CATEGORICAL_COLS[self.table_name]
            categorical_cols.update(NEW_CATEGORIES)
            for col in categorical_cols:
                if col in dataframe.columns:
                    dataframe[col] = dataframe[col].astype("category")

        return dataframe.set_index(self.index_name)


class Persons(SynPopTable):
    """All persons and their attributes."""

    def __init__(self, year, data=None):  # noqa: D107
        super().__init__(year, data)
        self.table_name = "persons"
        self.index_name = "person_id"
        self.cols_to_write = [
            "person_id",
            "household_id",
            "zone_id",
            "language",
            "level_of_employment",
            "age",
            "highest_education",
            "current_edu",
            "current_job_rank",
            "is_swiss",
            "sl3_id",
            "msr_id",
            "KT_full",
            "mun_name",
            "analysis_subpopulation",
        ]

    def load(self, path, validate=True, force_update_data=False):
        """Load Persons table from a file."""
        super().load(path, validate=validate, force_update_data=force_update_data)
        if "is_swiss" not in self.data.columns:
            assert self._data is not None
            self._data["is_swiss"] = self.data["nation"] == "swiss"
        if "analysis_subpopulation" not in self.data.columns:
            assert self._data is not None
            self._data["analysis_subpopulation"] = "regular"

    def validate(self):
        """Validate Persons table."""
        super().validate()
        validation.validate_persons(self.data)

    def overwrite_column(self, attribute_name, new_values):
        """Overwrite attribute while doing some safety checks.

        This attribute in the persons table will be overwritten using
        person_id as joining key. This function only works if all the
        person_ids have a matching new id. It may be extended later to
        overwrite only when exists.

        :param attribute_name: name of the attribute
        :param new_values: Must be a series with the new values and with
            person_id as index
        """
        if attribute_name == "is_swiss":
            self.overwrite_is_swiss(new_values)
            return
        # Sanity checking first
        logger.info('Overwriting "%s" with new values ...', attribute_name)
        assert new_values.index.nunique() == len(
            new_values.index
        ), "New values have duplicated indices!"
        assert set(self.data["person_id"]) == set(new_values.index)

        self.drop_column(attribute_name)

        self.update(
            pd.merge(
                self.data,
                new_values.rename(attribute_name),
                how="left",
                left_on="person_id",
                right_index=True,
            )
        )
        self.modified_columns.append(attribute_name)
        logger.info('"%s" has been overwritten.', attribute_name)

    def overwrite_is_swiss(self, new_values):
        """Overwrite is_swiss while doing some safety checks.

        When fixing is_swiss column, the nationality column must also be
        fixed. Since we do not know which non-swiss nationality the
        people are, the nationality is chosen randomly with wights
        according to the proportions in the global population.

        :param new_values: Must be a series with the new is_swiss values
            and with person_id as index
        """
        # Sanity checking first
        logger.info('Overwriting "nation" where new "is_swiss" values do not match...')
        assert (
            "nation" in self.data.columns
        ), "Attribute name is not in the persons table!"

        # Adding "is_swiss" to data
        self.overwrite_column("is_swiss", new_values)

        # Changing  "nation" to "swiss" for all people where "is_swiss" is "True"
        self.data["nation"] = self.data["nation"].mask(self.data["is_swiss"], "swiss")

        # Changing "nation" something other than "swiss" for all people where "is_swiss" is "False"
        is_new_foreigner = (~self.data["is_swiss"]) & (self.data["nation"] == "swiss")

        global_counts_per_nation = (
            self.data.groupby(["nation"], observed=True).size().drop("swiss")
        )

        foreign_nations = numpy.array(global_counts_per_nation.index)
        proportion = (global_counts_per_nation / global_counts_per_nation.sum()).values
        random_foreign_nations_as_replacements = numpy.random.choice(
            foreign_nations, self.total_rows, replace=True, p=proportion
        )

        self.data["nation"] = self.data["nation"].mask(
            is_new_foreigner, random_foreign_nations_as_replacements
        )

        logger.info('"nation" has been overwritten to match new "is_swiss" input.')

    def _fix_mobility(self):
        assert self._data is not None
        # fix Abos
        self._data["has_va"] = self._data["has_va"].mask(self._data["has_ga"], False)
        self._data["has_ht"] = self._data["has_ht"].mask(self._data["has_ga"], False)
        # fix mobility for car_available
        self._data["mobility"] = self._data["mobility"].mask(
            self._data["car_available"], "car"
        )
        # fix mobility for GA
        self._data["mobility"] = self._data["mobility"].mask(self._data["has_ga"], "ga")
        self._data["mobility"] = self._data["mobility"].mask(
            self._data["has_ga"] & self._data["car_available"], "car & ga"
        )
        # fix mobility for VA
        self._data["mobility"] = self._data["mobility"].mask(self._data["has_va"], "va")
        self._data["mobility"] = self._data["mobility"].mask(
            self._data["has_va"] & self._data["car_available"], "car & va"
        )
        # fix mobility for HT
        self._data["mobility"] = self._data["mobility"].mask(self._data["has_ht"], "ht")
        self._data["mobility"] = self._data["mobility"].mask(
            self._data["has_ht"] & self._data["car_available"], "car & ht"
        )
        self._data["mobility"] = self._data["mobility"].mask(
            self._data["has_va"] & self._data["has_ht"], "va & ht"
        )
        self._data["mobility"] = self._data["mobility"].mask(
            self._data["has_va"] & self._data["has_ht"] & self._data["car_available"],
            "car & va & ht",
        )
        # fix mobility for nothing
        self._data["mobility"] = self._data["mobility"].mask(
            ~self._data["has_ga"]
            & ~self._data["has_va"]
            & ~self._data["has_ht"]
            & ~self._data["car_available"],
            "nothing",
        )


class Businesses(SynPopTable):
    """All businesses and their attributes."""

    def __init__(self, year, data=None):  # noqa: D107
        super().__init__(year, data)
        self.total_businesses = None
        self.table_name = "businesses"
        self.index_name = "business_id"
        self.cols_to_write = [
            "business_id",
            "zone_id",
            "sector",
            "noga_code",
            "school_type",
            "jobs_endo",
            "fte_endo",
            "jobs_exo",
            "fte_exo",
            "KT_full",
            "mun_name",
            "xcoord",
            "ycoord",
        ]


class Households(SynPopTable):
    """All households and their attributes."""

    def __init__(self, year, data=None):  # noqa: D107
        super().__init__(year, data)
        self.total_households = None
        self.table_name = "households"
        self.index_name = "household_id"
        self.cols_to_write = [
            "household_id",
            "zone_id",
            "collective_hh_type",
            "xcoord",
            "ycoord",
        ]


class SynPop:
    """Synthetic population for a given year.

    Wrapper for the persons, households and businesses tables.
    """

    def __init__(
        self,
        year: int,
        persons: Optional[pd.DataFrame] = None,
        households: Optional[pd.DataFrame] = None,
        businesses: Optional[pd.DataFrame] = None,
    ):  # noqa: D107
        self.year = year
        self.persons = Persons(year, persons)
        self.households = Households(year, households)
        self.businesses = Businesses(year, businesses)

    def load(self, path: Union[str, Path], validate: bool = False):
        """Load SynPop.

        Supports a SQLite-DB or folder with feather files.
        """
        path = Path(path)
        self.persons.load(path, validate)
        self.households.load(path, validate)
        self.businesses.load(path, validate)

    def stats(self) -> pd.Series:
        """Return basic stats about the SynPop."""
        return pd.Series(
            {
                "persons": self.persons.data.shape[0],
                "households": self.households.data.shape[0],
                "businesses": self.businesses.data.shape[0],
                "zones": len(self.households.data["zone_id"].unique()),
            }
        )

    @staticmethod
    def from_path(
        path: Union[str, Path], year: Optional[int] = None, validate: bool = False
    ):
        """Load SynPop from path. Parse year if not provided."""
        if not year:
            stem = _resolve_synpop_path(Path(path), "persons").stem
            guess = re.findall(r"\d{4}", stem)
            if len(guess):
                year = int(guess[0])
        synpop = SynPop(year)
        synpop.load(path, validate)
        return synpop

    def update(
        self, persons: pd.DataFrame, households: pd.DataFrame, businesses: pd.DataFrame
    ) -> None:
        """Update SynPop with new data."""
        self.persons.update(persons)
        self.households.update(households)
        self.businesses.update(businesses)

    def copy(self) -> "SynPop":
        """Return copy of the SynPop."""
        return SynPop(
            self.year,
            self.persons.data.copy(),
            self.households.data.copy(),
            self.businesses.data.copy(),
        )

    def sample(self, frac: float, seed: int = 42, ensure_all_zones=False) -> "SynPop":
        """Sample SynPop. Optionally while ensuring all zones are still present (frac won't be as precise)."""
        n_zones = len(self.households.data["zone_id"].unique())
        if ensure_all_zones and (
            frac * self.households.data.shape[0] < n_zones
            or frac * self.businesses.data.shape[0] < n_zones
        ):
            min_frac = max(
                [
                    n_zones / self.households.data.shape[0],
                    n_zones / self.businesses.data.shape[0],
                ]
            )
            raise ValueError(
                f"frac must be greater or equal to {min_frac:.6f} when keeping all zones."
            )
        sampled_synpop = self.copy()

        if ensure_all_zones:
            sampled_synpop.households.update(
                self._sample_df_by_zones(sampled_synpop.households.data, frac, seed)
            )
            sampled_synpop.businesses.update(
                self._sample_df_by_zones(sampled_synpop.businesses.data, frac, seed)
            )
        else:
            sampled_synpop.households.update(
                sampled_synpop.households.data.sample(frac=frac, random_state=seed)
            )
            sampled_synpop.businesses.update(
                sampled_synpop.businesses.data.sample(frac=frac, random_state=seed)
            )
        sampled_synpop.persons.update(
            sampled_synpop.persons.data.query(
                "household_id in @sampled_synpop.households.data.household_id"
            )
        )
        return sampled_synpop

    @staticmethod
    def _sample_df_by_zones(dataframe, frac: float, seed: int = 42) -> pd.DataFrame:
        sampled = dataframe.groupby("zone_id", observed=True).sample(
            1, random_state=seed
        )
        sample_size = frac * dataframe.shape[0]
        if sample_size > len(sampled):
            frac = (sample_size - len(sampled)) / dataframe.shape[0]
            rest_sampled = (
                dataframe.drop(sampled.index)
                .groupby("zone_id", observed=True)
                .sample(frac=frac, random_state=seed)
            )
            sampled = pd.concat([sampled, rest_sampled], axis=0).sort_index()
        assert sampled["zone_id"].isin(dataframe["zone_id"]).all()
        return sampled

    def validate(self) -> None:
        """Validate SynPop."""
        self.persons.validate()
        self.households.validate()
        self.businesses.validate()

    def write(
        self,
        path: Union[str, Path],
        suffix: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Write SynPop to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(parents=True, exist_ok=True)
        self.persons.write(path / f"persons_{self.year}{suffix}.feather")
        self.households.write(path / f"households_{self.year}{suffix}.feather")
        self.businesses.write(path / f"businesses_{self.year}{suffix}.feather")
        if metadata is not None:
            with open(
                f"synpop_metadata_{self.year}{suffix}.json", encoding="utf-8"
            ) as file:
                json.dump(metadata, file)


def _resolve_synpop_path(path: Path, table_name: str) -> Path:
    if len(path.suffixes) and path.suffixes[0] == ".sqlite":
        return path
    files = list(path.glob(f"*{table_name}*.*"))
    assert (
        len(files) == 1
    ), f"Found multiple candidate files ({len(files)}) for {table_name} within {path}."
    return files[0]
