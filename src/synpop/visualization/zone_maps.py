"""Contains classes to generate geographical visualizations (maps)."""
import logging
import warnings

# Suppress pandas concat empty/all-NA warning
warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.",
    category=FutureWarning,
)

from collections import defaultdict
from typing import cast
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import colors

from synpop import DEFAULT_GEO_COLS
from synpop import mobi_zones_path, HOME_DIR

logger = logging.getLogger(__name__)

GEODATA_DIR = HOME_DIR / "assets/resources/geodata"
LAKE_COLOR = "#A0C8E6"

DEFAULT_SHAPES = {
    "mun_name": GEODATA_DIR / "Gemeinde.gpkg",
    "kt_name": GEODATA_DIR / "Kantone.gpkg",
    "KT_full": GEODATA_DIR / "Kantone.gpkg",
    "amr_name": GEODATA_DIR / "AMRegionen.gpkg",
    "msr_name": GEODATA_DIR / "MSRegionen.gpkg",
    "sl3_name": GEODATA_DIR / "SL3Regionen.gpkg",
    "Lakes": GEODATA_DIR / "BFS_CH14_Seen.gpkg",
}

# colormaps (one linear and one for two transitions in case of large growths)
ZERO_CENTERED_CMAP = "RdBu"
# Two transitions CMAP: non-sequential hsv (we want two transitions instead of one, but not a cycle)
CMAP_TWO_TRANSITIONS = colors.LinearSegmentedColormap.from_list(
    "trunc({n},{a:.2f},{b:.2f})".format(  # pylint: disable=consider-using-f-string
        n="hsv", a=0, b=0.8
    ),  # pylint: disable=consider-using-f-string
    cm.get_cmap("hsv")(np.linspace(0, 0.8, 100)),
)
CMAP_TWO_TRANSITIONS = colors.LinearSegmentedColormap.from_list(
    "CMAP_TWO_TRANSITIONS",
    np.vstack(
        (
            CMAP_TWO_TRANSITIONS(np.linspace(0, 0.22, 100)),
            CMAP_TWO_TRANSITIONS(np.linspace(0.23, 1.0, 100)),
        )
    ),
)


class SwissZoneMap:
    """Class to draw zonal maps of Switzerland."""

    def __init__(
        self,
        outline_communes: bool = False,
        outline_cantons: bool = True,
        draw_lakes: bool = True,
    ):
        """Draws a map of Switzerland to visualise zonal attributes.

        :param outline_communes: Draw commune outlines
        :param outline_cantons: Draw cantons outlies
        :param draw_lakes: Draw major water bodies
        :param commune_shp_path: use only to overwrite the default
            shapefile
        :param canton_shp_path: use only to overwrite the default
            shapefile
        :param lake_shp_path: use only to overwrite the default
            shapefile
        """
        self.outline_communes = outline_communes
        self.outline_cantons = outline_cantons
        self.draw_lakes = draw_lakes

        self.cantons = self.communes = self.lakes = None
        self._geo_cache = {}
        if self.outline_communes:
            self.communes = self.get_shape("mun_name")
        if self.outline_cantons:
            self.cantons = self.get_shape("kt_name")
        if self.draw_lakes:
            self.lakes = self.get_shape("Lakes")

    def get_shape(
        self, shape_id: str, zones: Optional[gpd.GeoDataFrame] = None
    ) -> gpd.GeoDataFrame:
        """Get geometries for a given shape_id.

        If not already loaded, it will be loaded from disk.
        """
        if shape_id not in self._geo_cache:
            if shape_id in DEFAULT_SHAPES:
                shape = DEFAULT_SHAPES[shape_id]
                if not shape.exists():
                    from simba.mobi.eap.util import (  # pylint: disable=import-outside-toplevel
                        download_dvc_file,
                    )

                    download_dvc_file(shape)
                geo = gpd.read_file(shape, driver="GPKG")
            elif zones is not None and shape_id in zones.columns:
                geo = zones[[shape_id, "geometry"]].dissolve(  # type: ignore
                    by=shape_id, as_index=False
                )
            else:
                raise ValueError(f"Unknown shape: {shape_id}")
            assert geo.index.is_unique
            self._geo_cache[shape_id] = geo
        return self._geo_cache[shape_id]

    def draw(  # pylint: disable=too-many-locals,too-many-arguments,too-many-statements,too-many-branches
        self,
        geodataframe: gpd.GeoDataFrame,
        column: str,
        cmap: Optional[Union[str, colors.Colormap]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        vcenter: Optional[float] = None,
        alpha: float = 1.0,
        legend: bool = True,
        title: Optional[str] = None,
    ) -> plt.Axes:
        """Draw a map of Switzerland with the given data.

        Adjusts colormap automatically.
        """
        if vmin is None:
            vmin = geodataframe[column].quantile(0.01)  # type: ignore
            if vmin == 0:
                vmin = -0.001
        if vmax is None:
            vmax = geodataframe[column].quantile(0.99)  # type: ignore
            if vmax == 0:
                vmax = 0.001
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        if vcenter is None:
            if vmin < 0 < vmax:
                vcenter = 0
                norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
                cmap = ZERO_CENTERED_CMAP
            else:
                cmap = "Reds_r" if vmax <= 0.001 else "Blues"
        elif round(abs(vcenter), 1) > 0:
            norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
            cmap = CMAP_TWO_TRANSITIONS

        # Preparing the figure and set copyright statement at extremes
        _, axis = plt.subplots(figsize=(15, 10))
        miny = geodataframe.geometry.bounds["miny"].min()
        maxx = geodataframe.geometry.bounds["maxx"].max()
        axis.text(maxx, miny, "SBB, MP-FV-APL-VPL\nSIMBA MOBi", fontsize=12)

        # ensure correct CRS
        if "2056" not in str(geodataframe.crs) and "CH1903+" not in str(
            geodataframe.crs
        ):
            try:
                import pyproj  # pylint: disable=import-outside-toplevel

                pyproj.Proj("+init=epsg:2056")
                geodataframe = geodataframe.to_crs(epsg=2056)  # type: ignore
            except RuntimeError:
                logging.error(
                    "CRS seems inconsistent and couldn't be transformed due to PyProj issue."
                    "Ignoring..."
                    "To fix PyProj in conda see: https://stackoverflow.com/a/58512331."
                )

        # Plotting the colored zones
        geodataframe.plot(
            ax=axis,
            column=column,
            linewidth=0.03,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            alpha=alpha,
        )
        axis.axis("off")
        axis.axis("tight")

        # Add Title
        if title:
            _ = axis.set_title(title, fontdict={"fontsize": 18, "fontweight": "bold"})

        if self.outline_communes:
            assert self.communes is not None
            self.communes.plot(ax=axis, facecolor="None", edgecolor="black", lw=0.5)

        if self.outline_cantons:
            assert self.cantons is not None
            self.cantons.plot(ax=axis, facecolor="None", edgecolor="black", lw=0.5)

        if self.draw_lakes:
            assert self.lakes is not None
            self.lakes.plot(ax=axis, facecolor=LAKE_COLOR, edgecolor="None")

        # Add single colorbar
        if legend:
            fig = axis.get_figure()
            cax = fig.add_axes([0.1, 0.9, 0.2, 0.01])
            scalable_cm = cm.ScalarMappable(cmap=str(cmap), norm=norm)
            scalable_cm.set_array([])
            cbar = fig.colorbar(scalable_cm, cax=cax, orientation="horizontal")
            cbar.ax.set_title(column, {"fontsize": 12})
            for cbar_xticklabel in cbar.ax.get_xticklabels():
                cbar_xticklabel.set(rotation=45)

        with warnings.catch_warnings(action="ignore"):
            plt.tight_layout()

        return axis

    def draw_cantons(  # pylint: disable=too-many-arguments
        self,
        dataframe: pd.DataFrame,
        column: str,
        cmap: Optional[Union[str, colors.Colormap]] = None,
        vmin: float = -10,
        vmax: float = 10,
        alpha: float = 1.0,
        legend: bool = True,
        title: Optional[str] = None,
    ) -> Tuple[plt.Axes, gpd.GeoDataFrame]:
        """Draw a map of cantons coloring each canton based on a given column attribute.

        The DataFrame has to have canton full names or abbreviations as
        index.
        """
        kt_name_col = dataframe.index.name
        assert self.cantons is not None and set(dataframe.index) == set(
            self.cantons[kt_name_col]
        ), f"DataFrame must have canton full names or abbreviations as index ({kt_name_col})."

        geo_df: gpd.GeoDataFrame = self.cantons.join(dataframe, on=kt_name_col)
        axis = self.draw(
            geo_df,
            column,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            legend=legend,
            title=title,
        )
        return axis, geo_df

    def draw_communes(  # pylint: disable=too-many-arguments
        self,
        dataframe: pd.DataFrame,
        column: str,
        cmap: Optional[str] = None,
        vmin: float = -10,
        vmax: float = 10,
        alpha: float = 1.0,
        legend: bool = True,
        title: Optional[str] = None,
    ) -> Tuple[plt.Axes, gpd.GeoDataFrame]:
        """Draw a map of communes coloring each commune based on a given column attribute.

        The DataFrame has to have all commune full names as index.
        """
        assert self.communes is not None and set(dataframe.index) == set(
            self.communes["mun_name"]
        ), "DataFrame must have all commune full names as index!"

        geo_df: gpd.GeoDataFrame = self.communes.join(dataframe, on="mun_name")
        axis = self.draw(
            geo_df,
            column,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            legend=legend,
            title=title,
        )
        return axis, geo_df


class ZonalViewer(SwissZoneMap):
    """A class to visualize zonal data on a map."""

    def __init__(
        self,
        zones1: gpd.GeoDataFrame,
        zones2: Optional[gpd.GeoDataFrame] = None,
        suffix1: str = "",
        suffix2: str = "",
        groupby_agg_cols: Optional[List[str]] = None,
        **kwargs,
    ):  # noqa: D107
        super().__init__(**kwargs)
        self.zones1 = zones1
        self.zones2 = zones2
        self.suffix1 = suffix1
        self.suffix2 = suffix2
        self.groupby_agg_cols = groupby_agg_cols

        if zones2 is not None:
            # ensure columns are the same
            cols = set.intersection(set(zones1.columns), zones2.columns)
            zones1 = zones1.drop(set(zones1.columns) - cols, axis=1)  # type: ignore
            zones2 = zones2.drop(set(zones2.columns) - cols, axis=1)  # type: ignore

        if not groupby_agg_cols:
            cols = set(zones1.columns.to_list())
            self.groupby_agg_cols = [c for c in cols if _col_is_numeric(zones1, c)]
            if zones2 is not None:
                self.groupby_agg_cols = set.intersection(
                    set(self.groupby_agg_cols), zones2.columns.to_list()
                )
                self.groupby_agg_cols = [
                    c for c in self.groupby_agg_cols if _col_is_numeric(zones2, c)
                ]

        assert self.groupby_agg_cols is not None
        self.groupby_agg_cols = [
            c for c in self.groupby_agg_cols if c not in DEFAULT_GEO_COLS
        ]
        self.dissolve_cache = defaultdict(dict)

        assert not set(self.groupby_agg_cols) - set(zones1.columns.to_list())
        if zones2 is not None:
            assert not set(self.groupby_agg_cols) - set(zones2.columns.to_list())

    def groupby_zones(
        self,
        zones: gpd.GeoDataFrame,
        groupby_agg: str,
        ref_var: str = "pop_total",
        aggregation: str = "sum",
    ) -> gpd.GeoDataFrame:
        """Group zones by a given column and aggregate all numeric columns."""
        assert self.groupby_agg_cols is not None
        cols = list(set(list(self.groupby_agg_cols) + [groupby_agg, ref_var]))
        df_agg = zones[cols].groupby(groupby_agg, observed=True).agg(aggregation)  # type: ignore
        geo = self.get_shape(groupby_agg, zones).set_index(groupby_agg)
        return cast(gpd.GeoDataFrame, geo.join(df_agg, how="inner"))

    def get_comparison(self, target_var, groupby_agg=None, geo_query=None):
        """Get a comparison of two zones."""
        # not implemented: add a further class for specific comparisons (i.e. variable)
        raise NotImplementedError

    def calc_zonal_stats(
        self,
        zones: gpd.GeoDataFrame,
        target_var: str,
        groupby_agg: Optional[str] = None,
        geo_query: Optional[str] = None,
        aggregation: str = "sum",
        ref_var: str = "pop_total",
    ) -> pd.DataFrame:
        """Calculate zonal statistics for a given variable and aggregation."""
        if geo_query is not None:
            zones = zones.query(geo_query)  # type: ignore
        if groupby_agg == "TOTAL":
            zones = zones[self.groupby_agg_cols].sum().to_frame("TOTAL").T  # type: ignore
            zones["geometry"] = None  # placeholder
        elif groupby_agg is not None:
            zones = self.groupby_zones(zones, groupby_agg, ref_var, aggregation)

        if ref_var != target_var:
            zones[f"{target_var} (%)"] = (
                (zones[target_var] / zones[ref_var]).astype(float).mul(100.0).round(1)
            )
            return zones[[target_var, ref_var, "geometry", f"{target_var} (%)"]]  # type: ignore
        return zones[[target_var, "geometry"]]  # type: ignore

    def calc_diff(
        self,
        zones1: Union[pd.DataFrame, gpd.GeoDataFrame],
        zones2: Union[pd.DataFrame, gpd.GeoDataFrame],
        target_var: str,
        ref_var: str,
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Calculate comparison statistics between two zones."""
        zones = zones1.join(
            zones2.drop("geometry", axis=1),
            lsuffix=f" {self.suffix1}",
            rsuffix=f" {self.suffix2}",
        )
        # reorder columns
        if ref_var != target_var:
            zones = zones.iloc[:, [0, 4, 1, 5, 3, 6, 2]]
        else:
            zones = zones.iloc[:, [0, 2, 1]]
        zones = zones.rename(
            columns=lambda c: c.replace(" (%)", "") + " (%)" if " (%)" in c else c
        )  # move suffix from percentage columns
        zones["Diff"] = (zones2[target_var] - zones1[target_var]).fillna(0.0)
        zones["Diff (%)"] = (
            (zones2[target_var] / zones1[target_var] - 1)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .mul(100.0)
            .round(1)
        )
        if ref_var != target_var:
            zones["Diff_Shares (%)"] = (
                (
                    zones[f"{target_var} {self.suffix2} (%)"]
                    - zones[f"{target_var} {self.suffix1} (%)"]
                )
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .mul(100.0)
                .round(1)
            )
        return zones

    def plot_zones(  # pylint: disable=too-many-locals,too-many-arguments
        self,
        target_var: str,
        plot_type: str = "rel",
        groupby_agg: Optional[str] = None,
        geo_query: Optional[str] = None,
        ref_var: str = "pop_total",
        aggregation: str = "sum",
        title: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        vcenter: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, gpd.GeoDataFrame, plt.Axes]:  # noqa: D202
        """Calculate zonal statistics and plot them. Optionally for comparison between two zones.

        :param plot_type: 'abs' for absolue, 'rel' for relative, 'share'
            for growth of shares (in relation to ref_var)
        :param groupby_agg: geographical aggregation
        :param geo_query: filter DF with pandas' query method
        :param ref_var: to which reference shall shares be calculated
        :param aggregation: when spatially aggregating, aggregate by sum or mean
        :param vmin, vmax and vcenter: plot bounds. If None, the .99 quantiles are taken
        :param colorscheme: not currently used
        :param label: 'id' for geographical ID or True for current variable
        """

        diff_var = {"rel": f"{target_var} (%)", "abs": target_var}
        zones = self.calc_zonal_stats(
            self.zones1, target_var, groupby_agg, geo_query, aggregation, ref_var
        )
        zones = zones.sort_values(ref_var, ascending=False)
        totals = self.calc_zonal_stats(
            self.zones1, target_var, "TOTAL", geo_query, aggregation, ref_var
        )
        if self.zones2 is not None:
            diff_var = {"rel": "Diff (%)", "abs": "Diff", "shares": "Diff_Shares (%)"}
            if ref_var == target_var and plot_type == "shares":
                raise ValueError(
                    "Cannot calculate difference of shares if reference==target"
                )
            zones2 = self.calc_zonal_stats(
                self.zones2, target_var, groupby_agg, geo_query, aggregation, ref_var
            )
            totals2 = self.calc_zonal_stats(
                self.zones2, target_var, "TOTAL", geo_query, aggregation, ref_var
            )
            zones = self.calc_diff(zones, zones2, target_var, ref_var)
            totals = self.calc_diff(totals, totals2, target_var, ref_var)

        if title is None:
            title_var = {
                "rel": "relative differences",
                "abs": "absolute differences",
                "shares": "relative difference of shares",
            }
            title = f'{target_var.replace("_", " ").title()}'
            if self.zones2 is None and plot_type == "rel":
                title = f'{title} (as % of {ref_var.replace("_", " ").title()})'
            if self.suffix1 and self.suffix2 and self.zones2 is not None:
                title = f"{title}: {title_var[plot_type]} ({self.suffix1} vs. {self.suffix2})"
            elif self.suffix1:
                title = f"{title}: {self.suffix1}"

        assert isinstance(zones, gpd.GeoDataFrame)
        axis = self.draw(
            zones,
            diff_var[plot_type],
            vmin=vmin,
            vmax=vmax,
            vcenter=vcenter,
            title=title,
        )

        stats_df = pd.concat([totals, zones]).drop("geometry", axis=1, errors="ignore").round(1)  # type: ignore

        return stats_df, zones, axis

    def get_rmse(self, target_var, groupby_agg=None, geo_query=None, aggregation="sum"):
        """Calculate the root mean squared error."""
        stats1 = self.calc_zonal_stats(
            self.zones1, target_var, groupby_agg, geo_query, aggregation=aggregation
        )
        assert self.zones2 is not None
        stats2 = self.calc_zonal_stats(
            self.zones2, target_var, groupby_agg, geo_query, aggregation=aggregation
        )
        return np.sqrt(
            np.sum(np.power(np.subtract(stats1[target_var], stats2[target_var]), 2))
            / len(stats1[target_var])
        )

    def get_mae(self, target_var, groupby_agg=None, geo_query=None, aggregation="sum"):
        """Calculate the mean absolute error."""
        stats1 = self.calc_zonal_stats(
            self.zones1, target_var, groupby_agg, geo_query, aggregation=aggregation
        )
        assert self.zones2 is not None
        stats2 = self.calc_zonal_stats(
            self.zones2, target_var, groupby_agg, geo_query, aggregation=aggregation
        )
        return np.sum(
            np.abs(np.subtract(stats1[target_var], stats2[target_var]))
        ) / len(stats1[target_var])


def aggregate_synpop(  # pylint: disable=too-many-locals
    persons: pd.DataFrame,
    businesses: Optional[pd.DataFrame] = None,
    aggregate_age_groups: bool = False,
    add_extra_variables: bool = False,
    only_existing_zones: bool = False,
) -> gpd.GeoDataFrame:
    """Aggregate synthetic population to mobi-zones and calculate aggregates."""
    # Prepare container
    logger.info("Aggregating synthetic population to zones.")
    zones = gpd.read_file(mobi_zones_path(), driver="GPKG")
    zone_ids = zones["zone_id"].to_list()  # type: ignore
    aggregates_df = pd.Series(zone_ids).to_frame("zone_id").set_index("zone_id")
    if only_existing_zones:
        existing_zone_ids = persons["zone_id"].unique().tolist()
        if businesses is not None:
            existing_zone_ids = set(existing_zone_ids + businesses["zone_id"].tolist())
        aggregates_df = aggregates_df.loc[aggregates_df.index.isin(existing_zone_ids)]

    # Mobility tools per Zone
    for var_to_agg, zonal_var in zip(
        ["car_available", "has_ga", "has_ht", "has_va"],
        ["pop_caravl", "pop_ga", "pop_ht", "pop_va"],
    ):
        if var_to_agg in persons:
            aggregates_df[zonal_var] = persons.groupby("zone_id", observed=True)[
                var_to_agg
            ].sum()
    if "has_ht" in persons.columns and "has_va" in persons.columns:
        aggregates_df["pop_va_ht"] = persons.loc[
            persons["has_ht"] & persons["has_va"], "zone_id"
        ].value_counts()

    # Students per Zone
    if "current_edu" in persons.columns:
        aggregates_df["pop_pup_1"] = (
            persons.loc[persons["current_edu"] == "pupil_primary", "zone_id"]
            .value_counts()
            .reindex(aggregates_df.index)
        )
        aggregates_df["pop_pup_2"] = (
            persons.loc[persons["current_edu"] == "pupil_secondary", "zone_id"]
            .value_counts()
            .reindex(aggregates_df.index)
        )
        aggregates_df["pop_stud_3"] = (
            persons.loc[persons["current_edu"] == "student", "zone_id"]
            .value_counts()
            .reindex(aggregates_df.index)
        )
        aggregates_df["pop_appr"] = (
            persons.loc[persons["current_edu"] == "apprentice", "zone_id"]
            .value_counts()
            .reindex(aggregates_df.index)
        )

    # Total population
    aggregates_df["pop_total"] = (
        persons["zone_id"].value_counts().reindex(aggregates_df.index)
    )
    if "level_of_employment" in persons.columns:
        aggregates_df["pop_empl"] = (
            persons.loc[persons["level_of_employment"] > 0, "zone_id"]
            .value_counts()
            .reindex(aggregates_df.index)
        )

    # Population totals per age groups per Zone
    if aggregate_age_groups:
        age_groups = [0, 18, 25, 45, 65, 75, 1000]
        persons["age_group"] = pd.cut(
            persons["age"],
            age_groups,
            right=False,
            labels=[
                "pop_0017",
                "pop_1824",
                "pop_2544",
                "pop_4564",
                "pop_6574",
                "pop_75xx",
            ],
        )
        person_age_group_df = (
            persons.groupby(["zone_id", "age_group"], observed=False)
            .size()
            .reset_index()
            .pivot(columns="age_group", index="zone_id")
        )
        person_age_group_df.columns = person_age_group_df.columns.droplevel(0).astype(
            str
        )
        aggregates_df[
            ["pop_0017", "pop_1824", "pop_2544", "pop_4564", "pop_6574", "pop_75xx"]
        ] = person_age_group_df[
            ["pop_0017", "pop_1824", "pop_2544", "pop_4564", "pop_6574", "pop_75xx"]
        ]

    # Businesses aggregates
    if businesses is not None:
        aggregates_df["jobs_endo"] = businesses.groupby("zone_id", observed=True)[
            "jobs_endo"
        ].sum()
        aggregates_df["jobs_total"] = (
            aggregates_df["jobs_endo"]
            + businesses.groupby("zone_id", observed=True)["jobs_exo"].sum()
        )
        aggregates_df["fte_endo"] = businesses.groupby("zone_id", observed=True)[
            "fte_endo"
        ].sum()
        aggregates_df["fte_total"] = (
            aggregates_df["fte_endo"]
            + businesses.groupby("zone_id", observed=True)["fte_exo"].sum()
        )

    if add_extra_variables:
        dfs = [aggregates_df]
        for col in ["current_job_rank", "language"]:
            col_counts = persons.pivot_table(
                index="zone_id",
                columns=col,
                aggfunc="size",
                fill_value=0,
                observed=True,
            )
            col_counts.columns = col_counts.columns.astype(str)
            col_counts = col_counts.drop(
                "apprentice", axis=1, errors="ignore"
            )  # apprentice added at current_edu
            dfs.append(col_counts)
        pop_active_age = (
            persons.query("17 < age < 66").groupby("zone_id", observed=True).size()
        )
        dfs.append(pop_active_age.to_frame("pop_active_age"))
        aggregates_df = pd.concat(dfs, axis=1, sort=False)

    join_type = "right" if only_existing_zones else "left"
    return cast(
        gpd.GeoDataFrame, zones.set_index("zone_id").join(aggregates_df, how=join_type)
    )


def _col_is_numeric(dataframe: pd.DataFrame, col: str) -> bool:
    return "int" in str(dataframe[col].dtype) or "float" in str(dataframe[col].dtype)


def get_zone_centroids():
    mobi_zones = gpd.read_file(mobi_zones_path(), driver="GPKG")
    zone_centroids = mobi_zones.set_index(DEFAULT_GEO_COLS).geometry.centroid
    zone_centroids = zone_centroids.to_frame("centroid").reset_index()
    zone_centroids["xcoord"] = zone_centroids["centroid"].apply(lambda c: c.x)
    zone_centroids["ycoord"] = zone_centroids["centroid"].apply(lambda c: c.y)
    return zone_centroids
