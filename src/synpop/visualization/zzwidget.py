import sys
from dataclasses import dataclass
from io import BytesIO
from typing import Dict
from typing import Optional

import geopandas as gpd
import streamlit as st

from simba.mobi.eap.cli import parse_key_value_from_cli
from synpop import DEFAULT_GEO_COLS
from synpop.visualization.zone_maps import ZonalViewer
from synpop.visualization.zones_datasource import DataSource
from synpop.visualization.zones_datasource import (
    LocalFileDataSource,
)
from synpop.visualization.zones_datasource import SnowflakeDataSource

KEY_VAL_SEPARATOR = ": "


@dataclass(frozen=True)
class Dashboard:
    """Class to manage the Streamlit dashboard for zone comparison."""

    zones1: gpd.GeoDataFrame
    datasource: DataSource
    zviewer: ZonalViewer
    zone_options: Dict[str, str]
    zones2: Optional[gpd.GeoDataFrame] = None

    def setup_sidebar(self):
        """Set up the sidebar widgets for configuration."""
        # Source & Target Zones
        st.sidebar.selectbox(
            "Reference", options=self.zone_options.keys(), key="reference_zones"
        )
        st.sidebar.selectbox(
            "Variant",
            options=[None] + list(self.zone_options.keys()),
            index=0,
            key="variant_zones",
        )

        # Spatial aggregation
        st.sidebar.selectbox(
            "Aggregation", options=self.get_geo_cols(), index=0, key="aggregation"
        )

        # Plot type selection
        plot_types = {"rel": "Relative", "abs": "Absolute", "shares": "Diff of shares"}
        st.sidebar.radio(
            "Plot Type:",
            options=plot_types.keys(),
            key="plot_type",
            format_func=plot_types.__getitem__,
            help="""
            - Absolute change: *scenario - reference*
            - Relative change: *(scenario/reference)-1*
            - Change of shares (percent-point difference): *(scenario/ref_variable) - (reference/ref_variable)*
            """,
        )

        # Variable selection
        zones1_columns = self.zones1.columns.tolist()
        zones2_columns = (
            self.zones2.columns.tolist() if self.zones2 is not None else None
        )
        target_var_options = self.get_variable_options(zones1_columns, zones2_columns)
        ref_var_options = self.get_variable_options(
            zones1_columns, zones2_columns, True
        )
        st.sidebar.selectbox(
            "Target Variable:",
            options=target_var_options,
            key="target_variable",
            index=target_var_options.index("pop_empl"),
        )
        st.sidebar.selectbox(
            "Reference Variable:",
            options=ref_var_options,
            key="reference_variable",
            index=ref_var_options.index("pop_total"),
        )

        # Aggregation function
        st.sidebar.radio(
            "Aggregation Function:",
            options=["sum", "mean"],
            key="agg_func",
            horizontal=True,
        )

        # Spatial filter
        filter_options = self.generate_filter_options()
        st.sidebar.selectbox(
            "Spatial Filter (Key: Value)",
            options=filter_options.keys(),
            key="geo_query",
            help="Select a key-value pair to filter the data.",
            format_func=filter_options.__getitem__,
        )

        # Form for updating colors
        with st.sidebar.form(key="color_form"):
            st.number_input("Color Scale Min (optional)", key="vmin", value=None)
            st.number_input("Color Scale Max (optional)", key="vmax", value=None)
            st.number_input("Color Scale Center (optional)", key="vcenter", value=None)
            st.form_submit_button(label="Update Color Scale")

    @staticmethod
    @st.cache_data
    def get_variable_options(zones1_columns, zones2_columns, ref_only: bool = False):
        """Get available variable options for plotting."""
        available_variables = set(zones1_columns) - set(DEFAULT_GEO_COLS + ["geometry"])
        if zones2_columns is not None:
            available_variables &= set(zones2_columns)
        if ref_only:
            ref_var_options = [
                "pop_total",
                "jobs_total",
                "pop_empl",
                "density",
                "area_land",
                "accsib_mul",
                "accsib_pt",
                "accsib_car",
            ]
            return [var for var in ref_var_options if var in available_variables]
        return sorted(available_variables)

    def get_geo_cols(self):
        return [
            c
            for c in DEFAULT_GEO_COLS
            if (((c == "zone_id") or c.endswith("_name")) and c in self.zones1.columns)
        ]

    def generate_filter_options(self):
        """Generate filter options formatted as 'key: value' for selection."""
        options = {None: ""}
        for col in self.get_geo_cols():
            unique_values = sorted(self.zones1[col].dropna().unique().tolist())
            for value in unique_values:
                val = value if isinstance(value, (int, float)) else f"'{value}'"
                options[f"{col} == {val}"] = f"{col}{KEY_VAL_SEPARATOR}{value}"
        return options

    def plot_data(self):
        """Plot the data based on the current configuration."""
        stats_df = self.cached_plot(
            st.session_state["reference_zones"],
            st.session_state["variant_zones"],
            self.zviewer,
            st.session_state["target_variable"],
            st.session_state["plot_type"],
            st.session_state.get("aggregation"),
            st.session_state.get("geo_query"),
            st.session_state["reference_variable"],
            st.session_state["agg_func"],
            None,
            st.session_state.get("vmin"),
            st.session_state.get("vmax"),
            st.session_state.get("vcenter"),
        )
        return stats_df

    @staticmethod
    @st.cache_data
    def cached_plot(  # pylint: disable=too-many-arguments
        tablename1: str,  # pylint: disable=unused-argument
        tablename2: str,  # pylint: disable=unused-argument
        _zviewer: ZonalViewer,
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
    ):
        """Plot the data based on the current configuration."""
        stats_df = _zviewer.plot_zones(
            target_var,
            plot_type,
            groupby_agg,
            geo_query,
            ref_var,
            aggregation,
            title,
            vmin,
            vmax,
            vcenter,
        )
        return stats_df

    def setup_data_display(self, stats_df):
        """Display original and filtered data."""
        st.subheader("Data")
        st.write(stats_df)

    def setup_download_buttons(self, geo_df):
        download_buffer = None
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Zones CSV",
                data=geo_df.drop("geometry", axis=1).to_csv().encode("utf-8"),
                file_name="zzwdiget.csv",
                mime="text/csv",
            )
        with col2:
            if st.button("Prepare Spatial for Download"):
                download_buffer = self.download_geopackage(geo_df)
            if download_buffer:
                st.download_button(
                    label="Download Spatial Data",
                    data=download_buffer,
                    file_name="zzwdiget.gpkg",
                    mime="application/octet-stream",
                )

    @staticmethod
    def download_geopackage(data):
        """Download the GeoDataFrame as a GeoPackage."""
        buffer = BytesIO()
        data.to_file(buffer, driver="GPKG", layer="filtered_data")
        buffer.seek(0)
        return buffer

    def render(self):
        """Render the entire dashboard."""
        st.title("Zonal Comparison Widget")
        self.setup_sidebar()
        stats_df, geo_df, axis = self.plot_data()
        fig = axis.get_figure()
        st.pyplot(fig)
        self.setup_data_display(stats_df)
        self.setup_download_buttons(geo_df)

    @staticmethod
    @st.cache_data
    def get_zones(_datasource, tablename):
        if tablename is None:
            return None
        zones = _datasource.load_zones(tablename)
        return zones

    @staticmethod
    def build(datasource: DataSource):
        st.set_page_config(page_title="Zonal Comparison Widget", layout="wide")
        zone_options = datasource.get_zone_options()
        if len(zone_options) == 0:
            raise RuntimeError("Zone options are empty")
        tablename1 = st.session_state.get(
            "reference_zones", list(zone_options.keys())[0]
        )
        tablename2 = st.session_state.get("variant_zones")
        zones1 = Dashboard.get_zones(datasource, tablename1)
        zones2 = Dashboard.get_zones(datasource, tablename2)
        zviewer = ZonalViewer(zones1, zones2, suffix1=tablename1, suffix2=tablename2)
        return Dashboard(zones1, datasource, zviewer, zone_options, zones2)


if __name__ == "__main__":
    datasource_ = SnowflakeDataSource.build()
    if sys.argv[1] == "True":
        add_zones = (
            parse_key_value_from_cli(None, "", sys.argv[2:])
            if len(sys.argv) > 2
            else {}
        )
        datasource_ = LocalFileDataSource.build(add_zones)
    dashboard = Dashboard.build(datasource_)
    dashboard.render()
