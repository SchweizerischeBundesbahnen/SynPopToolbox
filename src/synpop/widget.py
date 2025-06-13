"""Module containing logic and ipywidgets for the ZZWidget dashboard."""
import json
import logging
import os
import shutil
import socketserver
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from typing import cast
from typing import List
from typing import Optional
from typing import Union

import geopandas as gpd
import ipywidgets
import pandas as pd
import yaml
from IPython.display import clear_output
from IPython.display import display
from IPython.display import Javascript  # type: ignore
from ipywidgets import Layout

from synpop import mobi_zones_path, HOME_DIR
from synpop.synpop_tables import SynPop
from synpop.visualization import zone_maps

geo_cols = [
    "kt_name",
    "mun_name",
    "agglo_name",
    "amgr_name",
    "amr_name",
    "msr_name",
    "sl3_name",
    "zone_id",
]

SYNPOP_RESOURCES = HOME_DIR / "assets/resources"
EXAMPLE_ZONES = SYNPOP_RESOURCES / "dummy_zones.csv"


def copy_zonal_source(
    path: Union[str, Path],
    temp_dir: Path,
    suffix: Optional[str] = None,
) -> Path:
    """Copy zonal data for the ZZWidget. If input is a SynPop, loads and aggregates first."""
    path = Path(path)
    target_path = temp_dir / f"zones{suffix}.csv"
    if path.suffix == ".csv":
        shutil.copy(path, target_path)
    elif path.suffix in [".shp", ".gpkg"]:
        logging.info("Converting %s to CSV", path)
        dataframe = pd.DataFrame(gpd.read_file(path).drop("geometry", axis=1))
        dataframe.to_csv(target_path, sep=";")
    else:
        synpop = SynPop(0)
        synpop.load(path)
        dataframe = zone_maps.aggregate_synpop(
            synpop.persons.data,
            synpop.businesses.data,
            aggregate_age_groups=True,
            add_extra_variables=True,
        )
        dataframe.drop("geometry", axis=1, errors="ignore").to_csv(target_path, sep=";")
    return target_path


def _resolve_suffix(datasource: Path, suffix: Optional[str] = None) -> str:
    if suffix:
        return suffix
    suffix = datasource.stem
    while "mobi-zones" in suffix or "plans" in suffix:
        datasource = datasource.parent
        suffix = datasource.stem
    return suffix


def run_zone_diff_widget(  # pylint: disable=too-many-locals
    datasource1: Optional[Path] = None,
    datasource2: Optional[Path] = None,
    suffix1: Optional[str] = None,
    suffix2: Optional[str] = None,
    target_variable: Optional[str] = None,
    groupby_agg: Optional[str] = None,
    geoquery: Optional[str] = None,
) -> None:
    """Run the ZZWidget."""
    # find a free port for the server
    with socketserver.TCPServer(("localhost", 0), None) as srvr:  # type: ignore
        port = srvr.server_address[1]

    temp_dir = Path(tempfile.mkdtemp())

    if datasource1:
        suffix1 = _resolve_suffix(datasource1, suffix1)
        zones1 = copy_zonal_source(datasource1, temp_dir, suffix1)
    else:
        suffix1 = "dummy"
        zones1 = EXAMPLE_ZONES
    zone_options = {suffix1: str(zones1.resolve())}

    config = {
        "zones1": str(zones1.resolve()),
        "suffix1": suffix1,
        "codebase": str(HOME_DIR.parent.resolve()),
        "target_variable": target_variable or "pop_empl",
        "groupby_agg": groupby_agg or "kt_name",
        "geoquery": geoquery,
        "port": port,
        "zone_options": zone_options,
    }

    if datasource2:
        suffix2 = _resolve_suffix(datasource2, suffix2)
        zones2 = copy_zonal_source(datasource2, temp_dir, suffix2)
        zone_options.update({suffix2: str(zones2.resolve())})
        config.update({"zones2": str(zones2.resolve()), "suffix2": suffix2})

    with temp_dir.joinpath("config.yml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    widget_path = SYNPOP_RESOURCES / "zone_viewer_widget.md"
    shutil.copy(widget_path, temp_dir / widget_path.name)

    def start_widget():
        subprocess.run(
            [
                "jupyter",
                "server",
                "--port",
                str(port),
                "--no-browser",
                "--NotebookApp.token=''",
                "--NotebookApp.password=''",
                "--ExecutePreprocessor.timeout=300",
                "--VoilaConfiguration.show_tracebacks=true",
            ],
            cwd=str(temp_dir),
            check=True,
            shell=True,
        )

    try:
        # run voila/jupyter in a separate thread
        thread = threading.Thread(target=start_widget)
        thread.start()
        time.sleep(2)
        subprocess.run(
            ["jupytext", "--to", "notebook", widget_path.name],
            cwd=str(temp_dir),
            check=True,
        )
        subprocess.run(
            ["start", f"http://localhost:{port}/voila/render/{widget_path.stem}.ipynb"],
            cwd=str(temp_dir),
            shell=True,
            check=True,
        )
        while thread.is_alive():
            continue

    except RuntimeError as error:
        logging.error("Could not start Widget", exc_info=error)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_plot_type_widget(comparison=True):
    """Return a widget to select the type of plot to display."""
    plot_type_widget = ipywidgets.RadioButtons(
        options=(
            [("Relative", "rel"), ("Absolute", "abs")]
            + ([('Diff of pop-shares ("var/ref_var")', "shares")] if comparison else [])
        ),
        value="rel",
        layout=Layout(width="60%"),
    )
    return plot_type_widget


def get_vmin_widget():
    """Return a widget to select the Plot's colormap vmin."""
    vmin_widget = ipywidgets.Text(
        value=None,
        placeholder="Min",
        description="",
        disabled=False,
        layout=Layout(width="14%"),
    )
    return vmin_widget


def get_vmax_widget():
    """Return a widget to select the Plot's colormap vmax."""
    vmax_widget = ipywidgets.Text(
        value=None,
        placeholder="Max",
        description="",
        disabled=False,
        layout=Layout(width="14%"),
    )
    return vmax_widget


def get_vcenter_widget():
    """Return a widget to select the Plot's colormap vcenter."""
    vcenter_widget = ipywidgets.Text(
        value=None,
        placeholder="Center (optional)",
        description="",
        disabled=False,
        layout=Layout(width="14%"),
    )
    return vcenter_widget


def get_aggregation_type_widget():
    """Return a widget to select the aggregation type for the Plot."""
    return ipywidgets.RadioButtons(options=["sum", "mean"], value="sum")


def get_ref_var_widget(variables: Optional[List[str]] = None) -> ipywidgets.Dropdown:
    """Return a widget to select the reference variable."""
    options = [
        "pop_total",
        "jobs_total",
        "pop_empl",
        "density",
        "area_land",
        "accsib_mul",
        "accsib_pt",
        "accsib_car",
    ]
    if not variables:
        variables = options
    options = [o for o in options if o in variables]
    ref_var_widget = ipywidgets.Dropdown(
        value="pop_total",
        placeholder="Reference variable",
        options=options,
        description="",
        ensure_option=True,
    )
    return ref_var_widget


def get_target_value_widget(
    variables: List[str], target_variable: str = "pop_empl"
) -> ipywidgets.Dropdown:
    """Return a widget to select the target variable. Options depend on avaialble variables."""
    variables = list(set(variables) - set(zone_maps.DEFAULT_GEO_COLS + ["geometry"]))
    assert (
        target_variable in variables
    ), f"Invalid option for target_variable: '{target_variable}'. Chose one of {variables}."
    target_value_widget = ipywidgets.Dropdown(
        value=target_variable,
        placeholder="Choose target variable",
        options=sorted(list(variables)),
        description="",
        ensure_option=True,
        disabled=False,
    )
    return target_value_widget


def get_groupby_agg_widget(groupby_agg: str = "kt_name") -> ipywidgets.Dropdown:
    """Return a widget to select the aggregation variable. Options depend on avaialble variables."""
    assert (
        groupby_agg in geo_cols
    ), f"Invalid option for groupby_agg: '{groupby_agg}'. Chose one of {geo_cols}."
    groupby_agg_widget = ipywidgets.Dropdown(
        value=groupby_agg,
        placeholder="Choose aggregation variable (optional)",
        options=geo_cols,
        description="",
        ensure_option=True,
        disabled=False,
    )
    return groupby_agg_widget


class GeoQueryWidgets:
    """Class to hold widgets for geoquery selection."""

    def __init__(self, geoquery: Optional[str] = None):  # noqa: D107
        self.zones = gpd.read_file(mobi_zones_path(), driver="GPKG")
        self.query_key = geoquery.split("==")[0] if geoquery else None
        self.query_value = geoquery.split("==")[1] if geoquery else None
        self.query_targets = {
            c: sorted(self.zones[c].astype(str).unique().tolist()) for c in geo_cols  # type: ignore
        }

        self.geo_query_widget = self.get_geo_query_widget()
        self.query_target_widget = self.get_query_target_widget()

        self.current_value = self.geo_query_widget.value
        self.geo_query_widget.observe(self.on_query_key_change)

    def get_geo_query_widget(self) -> ipywidgets.Dropdown:
        """Get key widget for geoquery."""
        assert self.query_key in geo_cols + [
            None
        ], f"Invalid key for geoquery: '{self.query_key}'. Chose one of {geo_cols}."
        geo_query_widget = ipywidgets.Dropdown(
            value=self.query_key if self.query_key else "Aggregation Key",
            options=geo_cols + ["Aggregation Key"],
            description="",
            ensure_option=True,
            disabled=False,
            layout=Layout(width="20%"),
        )
        return geo_query_widget

    def get_query_target_widget(self) -> ipywidgets.Combobox:
        """Get target value widget for geoquery. Options depend on active Key."""
        query_target_widget = ipywidgets.Combobox(
            placeholder="Aggregation Value",
            value=self.query_value if self.query_value else None,
            options=self.query_targets[self.query_key] if self.query_key else [],
            description="",
            ensure_option=True,
            disabled=False,
            layout=Layout(width="20%"),
        )
        return query_target_widget

    def on_query_key_change(self, _: Any) -> None:
        """Refresh query target widget options when query key changes."""
        # adapted from https://stackoverflow.com/a/49187073/4187668
        if self.geo_query_widget.value != self.current_value:  # refresh combobox
            self.query_target_widget.index = None  # type: ignore  - mysterious workaround
            self.query_target_widget.index = 0  # type: ignore
            self.query_target_widget.options = self.query_targets[  # type: ignore
                self.geo_query_widget.value
            ]
            self.query_target_widget.value = ""

    @staticmethod
    def parse_query(key: str, value: str) -> Optional[str]:
        """Parse query from key and value."""
        if (key and value) and (key != "Aggregation Key"):
            try:
                if value.isdigit():
                    value_ = int(value)
                else:
                    value_ = float(value)
            except ValueError:
                pass
            if isinstance(value, str):
                value_ = f"'{value}'"
            geo_query = f"{key} == {value_}"
            return geo_query
        return None


def get_source_zone_widget(
    zone_options: List[str], suffix1: str
) -> ipywidgets.Dropdown:
    """Return widget to select source zones."""
    source_widget = ipywidgets.Dropdown(
        value=suffix1,
        options=zone_options,
        description="",
        ensure_option=True,
        disabled=False,
        layout=Layout(width="21.3%"),
    )
    return source_widget


def get_target_zone_widget(
    zone_options: List[str], suffix2: Optional[str] = None
) -> ipywidgets.Dropdown:
    """Return widget to select target zones."""
    kwargs = {
        "options": [None] + zone_options,
        "description": "",
        "ensure_option": True,
        "disabled": False,
        "layout": Layout(width="21.3%"),
    }
    if suffix2:
        kwargs["value"] = suffix2
    target_zone = ipywidgets.Dropdown(**kwargs)
    return target_zone


def get_shutdown_widget(port) -> ipywidgets.DOMWidget:
    """Return a widget to shutdown the server."""

    def shutdown_server():
        netprocesses = subprocess.check_output(["netstat", "-ano"])
        netprocesses = netprocesses.decode("utf-8").split("\r\n")
        header = ["Proto", "LocalAddress", "ForeignAddress", "State", "PID"]
        dataframe = pd.DataFrame(
            [[s for s in o.split(" ") if s != ""] for o in netprocesses[4:]],
            columns=header,
        )
        dataframe = dataframe.dropna()
        pid = dataframe.loc[dataframe.loc[:, "LocalAddress"].str.contains(port)].iloc[
            0, 4
        ]
        return subprocess.run(["taskkill", "/PID", cast(str, pid), "/F"], check=False)

    shutdown_button = ipywidgets.interactive(
        shutdown_server, {"manual": True, "manual_name": "Update"}
    )
    shutdown_button = shutdown_button.children[0]
    shutdown_button.description = "Shutdown Widget"  # type: ignore
    shutdown_button.button_style = "warning"  # type: ignore

    return shutdown_button


class DownloadButton(ipywidgets.Button):  # pylint: disable=abstract-method
    """Download button for currently displayed zones."""

    notify_output = ipywidgets.Output()
    display(notify_output)

    def __init__(
        self, filename: str, download_zones: gpd.GeoDataFrame, **kwargs
    ):  # noqa: D107
        super().__init__(**kwargs)
        self.filename = filename
        self.download_zones = download_zones
        self.on_click(self.on_click_callback)

    def on_click_callback(self, _: Any) -> None:
        """Download zones. Don't use the Browser function but save directly to Downloads folder."""
        download_path = (
            Path(cast(str, os.getenv("USERPROFILE"))) / f"Downloads/{self.filename}"
        )
        file_stem = download_path.stem
        file_index = 0
        while download_path.exists():
            file_index += 1
            download_path = download_path.with_name(
                f"{file_stem} ({file_index}){download_path.suffix}"
            )
        self.download_zones.to_file(download_path, driver="GPKG")
        self.popup(rf"Downloaded zones to {download_path}.")

    def popup(self, text: str) -> None:
        """Display a popup alert."""
        clear_output()
        display(Javascript(f"alert('{text}')"))


if __name__ == "__main__":
    run_zone_diff_widget()
