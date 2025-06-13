# pylint: disable=line-too-long
"""This module contains the functionality to get marginals distributions of the Swiss population.

See testing function at the bottom for example of how to use this module.
"""
import logging
import tempfile
import time
from hashlib import sha1
from pathlib import Path
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import pandas as pd
import requests.exceptions
from pyaxis import pyaxis

from synpop import utils
from synpop.config import SwissCantons

# Logger settings set in __init__.py
logger = logging.getLogger(__name__)

cantons = SwissCantons()

MAX_HTTP_CALL_ATTEMPTS = 5


def fetch_rawdata_from_bfs(url: str, col_filter: Dict[str, List[str]]) -> pd.DataFrame:
    """Download data from BFS in raw format and cache according to col_filter."""
    # get temporary directory (ensure it exists)
    temp_dir = Path(tempfile.gettempdir()).joinpath("BFS_rawData")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # check if cached results are available
    hash_params = sorted({**col_filter, "url": url}.items())
    cached_file = temp_dir / Path(
        sha1(repr(hash_params).encode("utf-8")).hexdigest()
    ).with_suffix(".pickle")
    if cached_file.exists():
        dataframe = pd.read_pickle(cached_file)
    else:
        logger.info(
            " Cache not available. "
            "Parsing data from %s. This could take a couple of minutes...",
            url,
        )
        # download raw data
        attempt = 0
        while (
            True
        ):  # Sometimes the HTTP call fails for no apparent reason -> try several times
            attempt += 1
            try:
                response = requests.get(url, timeout=10)
                break
            except requests.exceptions.RequestException as err:
                # HTTP calls sometimes fail, it usually is enough to wait a few seconds.
                logger.warning(
                    "HTTP call attempt #%d failed, attempting again... "
                    "(max. attempts: %d)",
                    attempt,
                    MAX_HTTP_CALL_ATTEMPTS,
                )
                time.sleep(3)
                if attempt == MAX_HTTP_CALL_ATTEMPTS:
                    logger.error(
                        "The maximum number of HTTP call attempts has been reached!"
                    )
                    raise err
        # parse into DataFrame
        tmp = Path(
            tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
                delete=False
            ).name
        )
        tmp.write_bytes(response.content)
        dataframe = pyaxis.parse(str(tmp), encoding="latin1")["DATA"]
        tmp.unlink()

        # apply filters
        ind = [True] * len(dataframe)
        for col, vals in col_filter.items():  # Loop through filters, updating index
            if col in dataframe.columns:
                new_ind = dataframe[col].isin(vals)
                if not any(new_ind):
                    raise AttributeError(
                        f"Values are not available: {vals}. "
                        f"Options are: {dataframe[col].unique().tolist()}."
                    )
                ind = ind & new_ind
            else:
                raise AttributeError(
                    f"Column not available: {col}. Options are: {dataframe.columns.tolist()}."
                )
        dataframe = dataframe.loc[ind]

        # cache results in disk
        dataframe.to_pickle(cached_file)

    return dataframe


class PopPredictionsClient:  # pylint: disable=too-many-instance-attributes
    """Client to load population scenarion data from the FSO.

    Kantonale Bevölkerungsszenarien 2020-2050 -
    Zukünftige Bevölkerungsentwicklung nach Szenario, Staatsangehörigkeit
    (Kategorie), Geschlecht, Altersklasse / Alter und Jahr.
    links:
    - https://www.bfs.admin.ch/asset/de/px-x-0104020000_106
    - https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken/tabellen.assetdetail.12947637.html

    Only permanent residents are counted in theses predictions!
    """

    def __init__(self):  # noqa: D107
        # The following fields will be filled after the data is queried
        self.year = None
        self.scenario_name = None

        self._pop_by_canton_age_and_nationality = None
        self._pop_by_canton_and_age = None
        self._pop_by_canton_and_age_group = None
        self._pop_by_age_group = None
        self._pop_by_canton = None
        self._pop_total = None

    @property
    def pop_by_canton_age_and_nationality(self) -> pd.DataFrame:  # noqa: D102
        assert self._pop_by_canton_age_and_nationality is not None
        return self._pop_by_canton_age_and_nationality

    @property
    def pop_by_canton_and_age(self) -> pd.DataFrame:  # noqa: D102
        assert self._pop_by_canton_and_age is not None
        return self._pop_by_canton_and_age

    @property
    def pop_by_canton_and_age_group(self) -> pd.DataFrame:  # noqa: D102
        assert self._pop_by_canton_and_age_group is not None
        return self._pop_by_canton_and_age_group

    @property
    def pop_by_age_group(self) -> pd.DataFrame:  # noqa: D102
        assert self._pop_by_age_group is not None
        return self._pop_by_age_group

    @property
    def pop_by_canton(self) -> pd.DataFrame:  # noqa: D102
        assert self._pop_by_canton is not None
        return self._pop_by_canton

    @property
    def pop_total(self) -> pd.DataFrame:  # noqa: D102
        assert self._pop_total is not None
        return self._pop_total

    def load(
        self, year: int, szenario_name: str = "Referenzszenario AR-00-2020"
    ) -> "PopPredictionsClient":
        """Load population data from FSO."""
        self.year = year
        self.scenario_name = szenario_name

        self._pop_by_canton_age_and_nationality = (
            self._get_population_by_canton_age_and_nationality()
        )

        self._pop_by_canton_and_age = (
            self.pop_by_canton_age_and_nationality.groupby(
                ["KT_full", "kt_name", "age"], observed=True
            )["pop"]
            .sum()
            .reset_index()
        )

        self._pop_by_canton_and_age_group = (
            self.pop_by_canton_and_age.assign(
                age_group=utils.bin_variable(
                    self.pop_by_canton_and_age["age"], last_bin_name="100-150"
                )
            )
            .groupby(["KT_full", "kt_name", "age_group"], observed=True)["pop"]
            .sum()
            .reset_index()
        )

        self._pop_by_age_group = (
            self.pop_by_canton_and_age_group.groupby("age_group", observed=True)["pop"]
            .sum()
            .reset_index()
        )

        self._pop_by_canton = (
            self.pop_by_canton_and_age.groupby(["KT_full", "kt_name"], observed=True)
            .sum()
            .reset_index()
        )

        self._pop_total = self.pop_by_canton["pop"].sum()

        return self  # returns an instance of this class with the data loaded and the variables set

    def _get_population_by_canton_and_age_group(self) -> pd.DataFrame:
        # Query data
        url = "https://www.bfs.admin.ch/bfsstatic/dam/assets/12947640/master"

        # Create DataFrame and format
        col_filter = {
            "Szenario-Variante": [self.scenario_name],
            "Jahr": [f"{self.year}"],
            "Kanton": cantons.all_canton_full_names(),
            "Geschlecht": ["Geschlecht - Total"],
            "Staatsangehörigkeit (Kategorie)": ["Schweiz", "Ausland"],
            "Altersklasse": [
                "0-4 Jahre",
                "5-9 Jahre",
                "10-14 Jahre",
                "15-19 Jahre",
                "20-24 Jahre",
                "25-29 Jahre",
                "30-34 Jahre",
                "35-39 Jahre",
                "40-44 Jahre",
                "45-49 Jahre",
                "50-54 Jahre",
                "55-59 Jahre",
                "60-64 Jahre",
                "65-69 Jahre",
                "70-74 Jahre",
                "75-79 Jahre",
                "80-84 Jahre",
                "85-89 Jahre",
                "90-94 Jahre",
                "95-99 Jahre",
                "100 Jahre oder mehr",
            ],
            "Beobachtungseinheit": [
                "Bevölkerungsstand am 31. Dezember"
            ],  # SynPop is calibrated on this variable
        }

        age_groups = [
            "0-4",
            "5-9",
            "10-14",
            "15-19",
            "20-24",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            "60-64",
            "65-69",
            "70-74",
            "75-79",
            "80-84",
            "85-89",
            "90-94",
            "95-99",
            "100-150",
        ]

        dataframe = fetch_rawdata_from_bfs(url, col_filter)

        dataframe = (
            dataframe.drop(["Szenario-Variante", "Jahr", "Beobachtungseinheit"], axis=1)
            .rename(
                columns={
                    "Kanton": "KT_full",
                    "Altersklasse": "age_group",
                    "DATA": "pop",
                }
            )
            .assign(
                kt_name=lambda x: x["KT_full"].map(cantons.full_names_to_abbreviations)
            )
            .assign(
                age_group=lambda x: x["age_group"]
                .str[:-6]
                .str.replace("100 Jahre ode", "100-150")
            )
            .assign(
                age_group=lambda x: pd.Categorical(
                    x["age_group"], categories=age_groups, ordered=True
                )
            )
            .astype({"KT_full": "category", "kt_name": "category", "pop": int})
        )
        dataframe = dataframe[
            ["KT_full", "kt_name", "age_group", "pop"]
        ]  # change column order

        return dataframe

    def _get_population_by_canton_age_and_nationality(self) -> pd.DataFrame:
        if self.scenario_name != "Referenzszenario AR-00-2020":
            raise ReferenceError(
                "Population by age is only available for the reference scenario!"
            )

        # Query data
        url = "https://www.bfs.admin.ch/bfsstatic/dam/assets/12947637/master"

        # Create DataFrame and format
        years = [f"{i} Jahre" for i in range(0, 101)]
        years[1] = years[1][:-1]  # 1 Jahre -> 1 Jahr
        years[-1] += " oder mehr"  # 100 Jahre -> 100 Jahre oder mehr

        col_filter = {
            "Jahr": [f"{self.year}"],
            "Kanton": cantons.all_canton_full_names(),
            "Alter": years,
            "Geschlecht": ["Geschlecht - Total"],
            "Staatsangehörigkeit (Kategorie)": ["Schweiz", "Ausland"],
            "Beobachtungseinheit": [
                "Bevölkerungsstand am 31. Dezember"
            ],  # SynPop is calibrated on this variable
        }

        dataframe = fetch_rawdata_from_bfs(url, col_filter)
        dataframe = (
            dataframe.drop(["Jahr", "Beobachtungseinheit", "Geschlecht"], axis=1)
            .rename(
                columns={
                    "Kanton": "KT_full",
                    "Alter": "age",
                    "Staatsangehörigkeit (Kategorie)": "is_swiss",
                    "DATA": "pop",
                }
            )
            .assign(
                kt_name=lambda x: x["KT_full"].map(cantons.full_names_to_abbreviations)
            )
            .assign(
                age=lambda x: x["age"]
                .str.replace("1 Jahr", "1 Jahre")
                .str.replace(" oder mehr", "")
                .str[:-6]
            )
            .assign(
                is_swiss=lambda x: x["is_swiss"].map(
                    {"Schweiz": True, "Ausland": False}
                )
            )
            .astype(
                {
                    "KT_full": "category",
                    "kt_name": "category",
                    "age": int,
                    "is_swiss": bool,
                    "pop": int,
                }
            )
        )
        dataframe = dataframe[
            ["KT_full", "kt_name", "age", "is_swiss", "pop"]
        ]  # change column order

        return dataframe


class ActivePopPredictionsClient:
    """Client to load active population data from the FSO (employed plus searching a job).

    Szenarien zur Entwicklung der Erwerbsbevölkerung ab 2020 -
    Erwerbsquote und Erwerbsbevölkerung nach Alter/Altersklasse, Geschlecht,
    Staatsangehörigkeit, pro Jahr und gemäss Szenario / Variante
    link 1: https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken/daten.assetdetail.12947685.html
    link 2: https://www.bfs.admin.ch/bfs/de/home/statistiken/arbeit-erwerb/erwerbstaetigkeit-arbeitszeit/erwerbspersonen/szenarien-erwerbsbevoelkerung.assetdetail.329286.html

    Only permanent residents are counted in theses predictions!
    For the FSO, active people includes job seekers!
    (Personnes active = personnes actives occupées + chômeurs)
    """

    def __init__(
        self,
        granularity: Union[
            Literal["age"], Literal["age_group"], Literal["global"]
        ] = "age",
    ):  # noqa: D107
        self.granularity = granularity
        # The following fields will be filled after the data is queried
        self.year = None
        self.scenario_name = None

        self._stats = None

    @property
    def stats(self) -> pd.DataFrame:  # noqa: D102
        assert self._stats is not None
        return self._stats

    def load(
        self, year: int, szenario_name: Optional[str] = None
    ) -> "ActivePopPredictionsClient":
        """Load active population data from the FSO."""
        self.year = year
        if szenario_name is None:
            szenario_name = (
                "A-00-2015"
                if self.granularity == "age_group"
                else "Referenzszenario A-00-2020"
            )
        self.scenario_name = szenario_name

        if self.granularity in ("age", "global"):
            self._stats = self._get_stats_age()
        else:
            raise ValueError(
                f'"{self.granularity}" is an invalid input for granularity. It must be "age", "age_group" or "global".'
            )

        return self  # returns an instance of this class with the data loaded and the variables set

    def _get_stats_age(self) -> pd.DataFrame:
        # Query data
        url = "https://www.bfs.admin.ch/bfsstatic/dam/assets/12947685/master"

        # Create DataFrame and format
        col_filter = {
            "Szenario-Variante": [self.scenario_name],
            "Jahr": [f"{self.year}"],
            "Geschlecht": ["Geschlecht - Total"],
            "Staatsangehörigkeit (Kategorie)": ["Staatsangehörigkeit - Total"],
            "Beobachtungseinheit": [
                "Erwerbsquote",
                "Erwerbsbevölkerung",
                "Erwerbsquote in VZÄ",
                "Erwerbsbevölkerung in VZÄ",
            ],
            "Alter": ["Alter - Total"],
        }
        if self.granularity == "age":
            col_filter["Alter"] = [f".... {age} Jahre" for age in range(15, 86)]

        dataframe = fetch_rawdata_from_bfs(url, col_filter).reset_index()
        dataframe = dataframe.pivot(
            index="Alter", values="DATA", columns="Beobachtungseinheit"
        ).reset_index()
        dataframe = (
            dataframe.rename(
                columns={
                    "Erwerbsbevölkerung": "active_people",
                    "Erwerbsbevölkerung in VZÄ": "active_people_fte",
                    "Erwerbsquote": "avg_active_people",
                    "Erwerbsquote in VZÄ": "avg_fte_per_person",
                    "Alter": "age",
                }
            )
            .astype(
                {
                    "active_people": float,
                    "active_people_fte": float,
                    "avg_active_people": float,
                    "avg_fte_per_person": float,
                }
            )
            .astype({"active_people": int, "active_people_fte": int})
        )
        dataframe.columns.name = None

        if self.granularity == "age":
            dataframe["age"] = dataframe["age"].str[5:-6].astype(int)
            dataframe = dataframe.sort_values("age").set_index("age")
        else:
            del dataframe["age"]

        return dataframe


class PopStatisticsClient:
    """Client to load population statistics data from the FSO.

    Ständige und nichtständige Wohnbevölkerung nach institutionellen Gliederungen,
    Staatsangehörigkeit, Geburtsort, Geschlecht und Altersklasse
    link: https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken/tabellen.assetdetail.14087564.html

    Actual data from 2010 to 2019
    """

    def __init__(self):  # noqa: D107
        # The following fields will be filled after the data is queried
        self.year = None
        self.scenario_name = None

        self._pop_by_canton_resident_status_and_age_group = None
        self._pop_by_canton_and_age_group = None
        self._pop_by_canton = None
        self._pop_by_age_group = None
        self._pop_total = None

    @property
    def pop_by_canton_resident_status_and_age_group(self) -> pd.DataFrame:  # noqa: D102
        assert self._pop_by_canton_resident_status_and_age_group is not None
        return self._pop_by_canton_resident_status_and_age_group

    @property
    def pop_by_canton_and_age_group(self) -> pd.DataFrame:  # noqa: D102
        assert self._pop_by_canton_and_age_group is not None
        return self._pop_by_canton_and_age_group

    @property
    def pop_by_age_group(self) -> pd.DataFrame:  # noqa: D102
        assert self._pop_by_age_group is not None
        return self._pop_by_age_group

    @property
    def pop_by_canton(self) -> pd.DataFrame:  # noqa: D102
        assert self._pop_by_canton is not None
        return self._pop_by_canton

    @property
    def pop_total(self) -> pd.DataFrame:  # noqa: D102
        assert self._pop_total is not None
        return self._pop_total

    def load(self, year: int) -> "PopStatisticsClient":
        """Load data from the FSO for a given year."""
        self.year = year

        self._pop_by_canton_resident_status_and_age_group = (
            self._get_pop_by_canton_resident_status_and_age_group()
        )

        self._pop_by_canton_and_age_group = (
            self.pop_by_canton_resident_status_and_age_group.groupby(
                ["KT_full", "kt_name", "age_group"], observed=True
            )
            .agg({"pop": sum})
            .reset_index()
        )

        self._pop_by_age_group = (
            self.pop_by_canton_and_age_group.groupby("age_group", observed=True)
            .agg({"pop": sum})
            .reset_index()
        )

        self._pop_by_canton = (
            self.pop_by_canton_and_age_group.groupby(
                ["KT_full", "kt_name"], observed=True
            )
            .agg({"pop": sum})
            .reset_index()
        )

        self._pop_total = self.pop_by_canton["pop"].sum()

        return self  # returns an instance of this class with the data loaded and the variables set

    def _get_pop_by_canton_resident_status_and_age_group(self) -> pd.DataFrame:
        # Query data
        url = "https://www.bfs.admin.ch/bfsstatic/dam/assets/14087564/master"

        # Create DataFrame and format
        all_cantons = cantons.all_canton_full_names()
        col_filter = {
            "Jahr": [f"{self.year}"],  # year in as a string
            "Kanton (-) / Bezirk (>>) / Gemeinde (......)": [
                f"- {c}" for c in all_cantons
            ],
            "Geschlecht": ["Geschlecht - Total"],
            "Staatsangehörigkeit (Kategorie)": [
                "Staatsangehörigkeit (Kategorie) - Total"
            ],
            "Geburtsort": ["Geburtsort - Total"],
            "Altersklasse": [
                "0-4 Jahre",
                "5-9 Jahre",
                "10-14 Jahre",
                "15-19 Jahre",
                "20-24 Jahre",
                "25-29 Jahre",
                "30-34 Jahre",
                "35-39 Jahre",
                "40-44 Jahre",
                "45-49 Jahre",
                "50-54 Jahre",
                "55-59 Jahre",
                "60-64 Jahre",
                "65-69 Jahre",
                "70-74 Jahre",
                "75-79 Jahre",
                "80-84 Jahre",
                "85-89 Jahre",
                "90-94 Jahre",
                "95-99 Jahre",
                "100 Jahre und mehr",
            ],
            "Bevölkerungstyp": [
                "Ständige Wohnbevölkerung",
                "Nichtständige Wohnbevölkerung",
            ],
        }

        age_groups = [
            "0-4",
            "5-9",
            "10-14",
            "15-19",
            "20-24",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            "60-64",
            "65-69",
            "70-74",
            "75-79",
            "80-84",
            "85-89",
            "90-94",
            "95-99",
            "100-120",
        ]

        dataframe = fetch_rawdata_from_bfs(url, col_filter)
        dataframe = (
            dataframe.drop(
                ["Jahr", "Staatsangehörigkeit (Kategorie)", "Geburtsort", "Geschlecht"],
                axis=1,
            )
            .rename(
                columns={
                    "Kanton (-) / Bezirk (>>) / Gemeinde (......)": "KT_full",
                    "Bevölkerungstyp": "resident",
                    "Altersklasse": "age_group",
                    "DATA": "pop",
                }
            )
            .assign(KT_full=lambda x: x["KT_full"].str[2:])  # remove ' - '
            .assign(
                kt_name=lambda x: x["KT_full"].map(cantons.full_names_to_abbreviations)
            )
            .assign(resident=lambda x: x["resident"] == "Ständige Wohnbevölkerung")
            .assign(
                age_group=lambda x: x["age_group"]
                .str[:-6]
                .str.replace("100 Jahre un", "100-120")
            )
            .assign(
                age_group=lambda x: pd.Categorical(
                    x["age_group"], categories=age_groups, ordered=True
                )
            )
            .astype({"KT_full": "category", "kt_name": "category", "pop": int})
        )
        dataframe = dataframe[
            ["KT_full", "kt_name", "age_group", "resident", "pop"]
        ]  # change column order
        return dataframe


def load_hh_bfs_pred(year):
    """Load the predicted number of households by canton and size from BFS."""
    bfs_url = (
        "https://dam-api.bfs.admin.ch/hub/api/dam/assets/3623353/master"
        if year < 2020
        else "https://dam-api.bfs.admin.ch/hub/api/dam/assets/16344855/master"
    )
    temp = Path(
        tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
            delete=False
        ).name
    )
    try:
        response = requests.get(bfs_url, timeout=10, proxies=PROXIES)
        temp.write_bytes(response.content)
        pc_households_bsf_pred = (
            pd.read_excel(temp, header=1 + (year < 2020), engine="openpyxl")
            .iloc[8 + 3 * (year < 2020) : 14 + 3 * (year < 2020)]
            .drop("Unnamed: 0", axis=1)
            .loc[:, year]
            .astype(float)
            .round(2)
        )
    except (IOError, requests.exceptions.RequestException) as err:
        logging.error("Error loading data: %s", err)
        raise
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    pc_households_bsf_pred.index = [1, 2, 3, 4, 5, 6]
    pc_households_bsf_pred.index.name = "members"
    return pc_households_bsf_pred


# Testing the classes declared above (The following code can be used a user-guide.)
def _testing_pop_predictions():
    print("\n\nTesting PopPredictionsClient().load(2030): ")
    pop_predictions_2030 = PopPredictionsClient().load(2030)

    print(
        "\npop_by_canton_age_and_nationality ==>\n",
        pop_predictions_2030.pop_by_canton_age_and_nationality.head(3).T,
    )
    print(
        "\npop_by_canton_and_age ==>\n",
        pop_predictions_2030.pop_by_canton_and_age.head(3).T,
    )
    print(
        "\npop_by_canton_and_age_group ==>\n",
        pop_predictions_2030.pop_by_canton_and_age_group.head(3).T,
    )
    print("\npop_by_canton ==>\n", pop_predictions_2030.pop_by_canton.head(3).T)
    print("\npop_total ==> ", pop_predictions_2030.pop_total)


def _testing_active_pop_predictions():
    # Global
    print("\n\nTesting ActivePopPredictionsClient(granularity='global').load(2019): ")
    active_pop_2017 = ActivePopPredictionsClient(granularity="global").load(2019)
    print(active_pop_2017.stats.head(3).T)

    # By Age
    print("\n\nTesting ActivePopPredictionsClient(granularity='age').load(2019): ")
    active_pop_2017 = ActivePopPredictionsClient(granularity="age").load(2019)
    print(active_pop_2017.stats.head(3).T)

    print("\n\nTesting ActivePopPredictionsClient(granularity='age').load(2040): ")
    active_pop_2040 = ActivePopPredictionsClient(granularity="age").load(2040)
    print(active_pop_2040.stats.head(3).T)

    # By Age Group
    # active_pop_2017 = ActivePopPredictionsClient(granularity="age_group").load(2019)
    # print(
    #     "\n\nTesting ActivePopPredictionsClient(granularity='age_group').load(2019): "
    # )
    # print(active_pop_2017.stats.head(3).T)


def _testing_stats():
    pop_stats_2017 = PopStatisticsClient().load(2017)

    print("\n\nTesting PopStatisticsClient().load(2017): ")
    print(
        "\npop_by_canton_resident_status_and_age_group ==>\n",
        pop_stats_2017.pop_by_canton_resident_status_and_age_group.head(3).T,
    )
    print(
        "\npop_by_canton_age_group ==>\n",
        pop_stats_2017.pop_by_canton_and_age_group.head(3).T,
    )
    print("\npop_by_canton ==>\n", pop_stats_2017.pop_by_canton.head(3).T)
    print("\npop_by_age_group ==>\n", pop_stats_2017.pop_by_age_group.head(3).T)
    print("\npop_total ==> ", pop_stats_2017.pop_total)


if __name__ == "__main__":
    _testing_pop_predictions()
    _testing_stats()
    _testing_active_pop_predictions()
