import json
import re
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from snowflake.snowpark import Session

from simba.cloud.portal.api.api import refreh_token
from simba.mobi.eap.snowflake import AZURE_SNOWFLAKE_APPID_MOBI
from simba.mobi.eap.snowflake import DEFAULT_CONNECTION_PARAMS
from synpop import DEFAULT_GEO_COLS
from synpop import mobi_zones_path
from synpop.visualization.zone_maps import DEFAULT_SHAPES
from synpop.widget import DEFAULT_ZONES

GEO_TABLES: dict[str, str] = {
    "mun_name": "GEMEINDE",
    "kt_name": "KANTONE",
    "KT_full": "KANTONE",
    "amr_name": "ARBEITSMARKTREGIONEN",
    "msr_name": "MSREGIONEN",
    "sl3_name": "SL3REGIONEN",
    "Lakes": "LAKES",
    "zone_id": "NPVMZONES",
}


@dataclass(frozen=True)
class DataSource(ABC):
    """Abstract base class for data sources."""

    session: Optional[Session] = None
    zone_options: Optional[dict[str, str]] = None

    @abstractmethod
    def load_shape(self, target: str):
        pass

    @abstractmethod
    def load_table(self, target: str):
        pass

    @abstractmethod
    def get_zone_options(self):
        pass

    @abstractmethod
    def get_base_zones(self):
        pass

    @abstractmethod
    def get_year(self, target: str) -> int:
        pass

    def load_zones(self, target: str):
        df = self.load_table(target)
        df = df.set_index("zone_id").drop(DEFAULT_GEO_COLS, axis=1, errors="ignore")
        base_zones = self.get_base_zones()
        return base_zones.join(
            df.drop("geometry", axis=1, errors="ignore")
        ).reset_index()


class LocalFileDataSource(DataSource):
    """Data source for loading local CSV or Geopackage files."""

    @staticmethod
    def build(
        additional_sources: Optional[Dict[str, str]] = None,
    ) -> "LocalFileDataSource":  # noqa: D107
        if additional_sources is None:
            return LocalFileDataSource()
        zone_options = additional_sources
        for key, value in DEFAULT_ZONES["paths"].items():
            # update while preferring original (and retain order)
            zone_options[key] = zone_options.get(key, value)

        return LocalFileDataSource(zone_options=zone_options)

    def load_shape(self, target: str):
        """Loads data from a CSV or Geopackage file."""
        target_path = DEFAULT_SHAPES[target]
        if target_path.lower().endswith(".gpkg"):
            return gpd.read_file(target_path, driver="GPKG")
        raise ValueError(target)

    def load_table(self, target: str):
        target_path = self.zone_options[target]
        if target_path.lower().endswith(".csv"):
            return pd.read_csv(target_path, sep=";")
        raise ValueError(target)

    def get_zone_options(self):
        return self.zone_options

    def get_base_zones(self):
        return gpd.read_file(mobi_zones_path(), driver="GPKG").set_index("zone_id")

    def get_year(self, target):
        year = 2099
        guess = re.findall(r"\d{4}", target)
        if len(guess):
            year = int(guess[0])
        else:
            target = DEFAULT_ZONES["paths"].get(target, target)
            guess = re.findall(r"\d{4}", target)
            if len(guess):
                year = int(guess[0])
        return year


class SnowflakeDataSource(DataSource):
    SCHEMA = "ZZWIDGET"
    DEFAULT_CRS = "EPSG:2056"
    GEOM_COL = "geometry"

    @staticmethod
    def build(
        custom_connection_params: Optional[Dict[str, str]] = None,
    ) -> "SnowflakeDataSource":  # noqa: D107
        connection_parameters = DEFAULT_CONNECTION_PARAMS.copy()
        connection_parameters["token"] = refreh_token(AZURE_SNOWFLAKE_APPID_MOBI)
        if custom_connection_params:
            connection_parameters.update(custom_connection_params)
        session = Session.builder.configs(connection_parameters).create()

        sdf = session.table("RUNS_PLANS")
        zone_options = (
            sdf.select(["MLFLOW_RUNID", "RUN_UNAME"])
            .to_pandas()
            .set_index("RUN_UNAME")["MLFLOW_RUNID"]
            .to_dict()
        )
        return SnowflakeDataSource(session=session, zone_options=zone_options)

    def load_shape(self, target: str):
        if target in GEO_TABLES:  # pylint: disable=consider-using-get
            target = GEO_TABLES[target]
        if target in GEO_TABLES.values():
            df = (
                self.session.table(f"{self.SCHEMA}.{target}")
                .to_pandas()
                .rename(columns=lambda c: c.lower())
            )
            df[self.GEOM_COL] = df[self.GEOM_COL].apply(lambda x: shape(json.loads(x)))
            return gpd.GeoDataFrame(df, geometry=self.GEOM_COL, crs=self.DEFAULT_CRS)
        raise ValueError(target)

    def load_table(self, target: str):
        target_runuid = self.zone_options[target]
        sdf = self.session.table("MOBI_ZONES")
        df = (
            sdf.filter(f"RUNUID = '{target_runuid}'")
            .to_pandas()
            .drop("RUNUID", axis=1)
            .rename(columns=lambda c: c.lower())
        )
        if len(df) == 0:
            raise ValueError(target)
        return df

    def get_zone_options(self):
        return self.zone_options

    def get_year(self, target: int):
        sdf = self.session.table("RUNS_PLANS")
        df = sdf.filter(f"MLFLOW_RUNID = {target}").select("YEAR").to_pandas()
        return df["YEAR"].iloc[0]

    def get_base_zones(self):
        return self.load_shape("NPVMZONES").set_index("zone_id")
