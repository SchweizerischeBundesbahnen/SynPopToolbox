---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
#########################
##### INITIAL SETUP #####
#########################
```

```{code-cell} ipython3
%matplotlib widget
%load_ext autoreload
%autoreload 2
```

```{code-cell} ipython3
import warnings

# Suppress Altair convert_dtype deprecation warning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="altair",
)

import logging
import ipywidgets as widgets
from ipywidgets import VBox, HBox, Layout, HTML
import pandas as pd
import geopandas as gpd
import sys
from matplotlib import pyplot as plt
from shapely.geometry import Point
from IPython.display import display, IFrame
import markdown
import subprocess
```

```{code-cell} ipython3
import yaml

# load config and synpop code
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

from synpop import mobi_zones_path
from synpop.visualization import zone_maps
from synpop.visualization.visualisations import generate_simple_grid_table
```

```{code-cell} ipython3
# setup altair
from synpop.visualization.visualisations import altair_scatter
from synpop import widget as synpop_widgets
import altair as alt
alt.data_transformers.disable_max_rows();
import altair_viewer
```

```{code-cell} ipython3
logger = logging.getLogger('SynPop')
logging.getLogger('numexpr.utils').setLevel(logging.CRITICAL)  # supress logging
logging.getLogger('tornado.access').setLevel(logging.CRITICAL)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)
logging.getLogger('pyproj').setLevel(logging.CRITICAL)
logging.getLogger('pandas').setLevel(logging.CRITICAL)

# not displayed
SUFFIX1 = config['suffix1']
SUFFIX2 = config.get('suffix2')

PORT = str(config['port'])
```

```{code-cell} ipython3
prettify = lambda c: c.replace('_', ' ').title()
kwargs = {}  # for debugging
def update_tooltip(event, tooltip_cols, gdf, tooltip, groupby_agg):
    global kwargs
    kwargs.update(locals())
    # get the point contained in the event
    point = gpd.GeoDataFrame(geometry=gpd.GeoSeries(
        [Point(event.xdata, event.ydata)], crs={'init': 'epsg:2056'}))
    intersection = gpd.sjoin(gdf, point)
    kwargs.update({'point': point, 'intersection': intersection})
    if len(intersection) > 0:
        tooltip.set_position((event.xdata, event.ydata))
        text_parts = [f"{prettify(groupby_agg)}: {intersection.index[0]}"]
        text_parts += [f"{prettify(c)}: {intersection[c].iloc[0]}" for c in tooltip_cols]
        tooltip.set_text('\n'.join(text_parts))
        tooltip.set_visible(True)
    else:
        tooltip.set_visible(False)
    plt.draw()
```

```{code-cell} ipython3
class MapPlotWidget:
    last_plot_kwargs: dict
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

    def __init__(self, suffix_zone_dict, download_button, target_value_widget, ref_var_widget, plot_type_widget, suffix1, suffix2=None):
        self.download_button = download_button
        self.target_value_widget = target_value_widget
        self.ref_var_widget = ref_var_widget
        self.plot_type_widget = plot_type_widget
        self.suffix_zone_dict = suffix_zone_dict
        self.ref = None
        self.var = None
        self.suffix1 = suffix1
        self.suffix2 = suffix2
        self.init_zviewer(self.suffix1, self.suffix2)

    def init_zviewer(self, suffix1, suffix2=None):
        zones1 = self.suffix_zone_dict.get(suffix1)
        zones2 = self.suffix_zone_dict.get(suffix2)
        self.ref = self.load_ref_zones(suffix1, zones1)
        self.var = self.load_var_zones(suffix2, zones2) if zones2 else None
        self.zviewer = zone_maps.ZonalViewer(self.ref, self.var, suffix1, suffix2)
        self.suffix1 = suffix1
        self.suffix2 = suffix2

        # update variable widget
        variables = set(self.ref.columns.tolist()) - set(zone_maps.DEFAULT_GEO_COLS + ['geometry'])
        if self.var is not None:
            variables = variables & set(self.var.columns.tolist())
        value = self.target_value_widget.value
        self.target_value_widget.options = sorted(list(variables))
        self.target_value_widget.value = value
        self.ref_var_widget.options = [o for o in self.ref_var_options if o in variables]

    def load_ref_zones(self, suffix, path):
        if suffix == self.suffix1 and self.ref is not None:
            return self.ref
        return self.load_zones(path)

    def load_var_zones(self, suffix, path):
        if suffix == self.suffix2 and self.var is not None:
            return self.var
        return self.load_zones(path)

    @staticmethod
    def load_zones(path):
        df = pd.read_csv(path, index_col='zone_id', sep=';').drop(zone_maps.DEFAULT_GEO_COLS, axis=1, errors='ignore')
        return zones.set_index('zone_id').join(df.drop('geometry', axis=1, errors='ignore')).reset_index()

    def update(self, Variable, Comparison, Aggregation, Aggregation_Function, Ref_variable,
               FilterKey, FilterValue, vmin, vmax, vcenter, suffix1, suffix2):
        if (suffix1 != self.suffix1) or (suffix2 != self.suffix2):
            self.init_zviewer(suffix1, suffix2)

        geo_query = synpop_widgets.GeoQueryWidgets.parse_query(FilterKey, FilterValue)
        if Ref_variable == Variable and suffix2 is None:
            self.plot_type_widget.value = Comparison = 'abs'
        self.last_plot_kwargs = dict(
            target_var=Variable,
            plot_type=Comparison,
            groupby_agg=Aggregation,
            aggregation=Aggregation_Function,
            ref_var=Ref_variable,
            geo_query=geo_query,
            vmin=float(vmin) if vmin != '' else None,
            vmax=float(vmax) if vmax != '' else None,
            vcenter=float(vcenter) if vcenter != '' else None
        )

        # output widgets
        plot_w = widgets.Output()
        grid_w = widgets.Output(layout=Layout(width='1000px'))
        altair_w = widgets.Output(layout=Layout(width='600px'))
        output = VBox([plot_w, HBox([grid_w, altair_w])], layout=Layout(width='1600px'))

        # plot and tooltip
        plt.close('all')
        with plot_w:
            df, plotted_zones, ax = self.zviewer.plot_zones(**self.last_plot_kwargs)
            self.download_button.download_zones = plotted_zones
            tooltip = ax.text(0, 0, "", bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", ec="black", lw=1.2, alpha=0.6),
                              fontsize=9, c='black')
            tooltip_cols = df.columns.to_list()[0:4] + (['Diff (%)'] if len(df.columns) > 5 else [])
            ax.get_figure().canvas.mpl_connect("button_press_event",
                                               lambda event: update_tooltip(event, tooltip_cols, plotted_zones, tooltip, Aggregation))
            plt.pause(0.1)

        # grid
        df.index.name = Aggregation
        df = df.round(1).reset_index().rename(columns=prettify)

        with grid_w:
            display(generate_simple_grid_table(df, header_height=80, width="100%"))

        # altair scatter
        if not (Ref_variable == Variable and suffix2 is None):
            cols = ([Variable] if self.var is None else [f'{Variable} {suffix1}', f'{Variable} {suffix2}'])
            cols = [prettify(c) for c in cols]
            aggregation = prettify(Aggregation)
            df_scatter = df.set_index(aggregation).drop('TOTAL')
            df_scatter.index.name = aggregation
            df_data = df_scatter[cols].copy()
            if Ref_variable != Variable:
                df_scatter = df_scatter.drop(cols, axis=1)
            df_scatter = df_scatter.stack().to_frame('y-value')
            df_scatter.index.names = [aggregation, 'y-axis']
            df_scatter = df_scatter.reset_index('y-axis')
            df_scatter = df_scatter.join(df_data).reset_index()
            if self.var is None:
                chart = altair_scatter(df_scatter, cols[0], 'y-value', table=False, title=cols[0],
                                   dropdown_col='y-axis', dropdown_name=f'Y-Axis {suffix1} ', dropdown_all_option=False)
            else:
                chart1 = altair_scatter(df_scatter, cols[0], 'y-value', table=False, title=cols[0],
                                    dropdown_col='y-axis', dropdown_name=f'Y-Axis {suffix1} ', dropdown_all_option=False)
                chart2 = altair_scatter(df_scatter, cols[1], 'y-value', table=False, title=cols[1],
                                    dropdown_col='y-axis', dropdown_name=f'Y-Axis {suffix2} ', dropdown_all_option=False)
                chart = alt.vconcat(chart1, chart2).resolve_legend(color="independent")

            #breakpoint()
            host = altair_viewer.display(chart, open_browser=False)
            with altair_w:
                display(IFrame(host.url, width=580, height=1000))

        display(output)
```

```{code-cell} ipython3
#####################
##### LOAD DATA #####
#####################
```

```{code-cell} ipython3
zones = gpd.read_file(mobi_zones_path(), driver="GPKG")
```

```{code-cell} ipython3
geo_cols = ['kt_name', 'mun_name', 'agglo_name', 'amgr_name', 'amr_name', 'msr_name', 'sl3_name', 'zone_id']
```

```{code-cell} ipython3
##########################
##### CREATE WIDGETS #####
##########################
```

```{code-cell} ipython3
plotted_zones = None
download_button = synpop_widgets.DownloadButton(filename='mobi-zones_zzwidget.geopkg',
                                                download_zones=None,
                                                description='Download Zones')
```

```{code-cell} ipython3
plot_type_widget = synpop_widgets.get_plot_type_widget(SUFFIX2 is not None)
```

```{code-cell} ipython3
target_value_widget = synpop_widgets.get_target_value_widget(['pop_empl'], config.get('target_variable', 'pop_empl'))
```

```{code-cell} ipython3
ref_var_widget = synpop_widgets.get_ref_var_widget()
```

```{code-cell} ipython3
map_plotter = MapPlotWidget(config["zone_options"], download_button, target_value_widget, ref_var_widget, plot_type_widget, SUFFIX1, SUFFIX2)
```

```{code-cell} ipython3
groupby_agg_widget = synpop_widgets.get_groupby_agg_widget(config.get('groupby_agg', 'kt_name'))
```

```{code-cell} ipython3
geoquery_widgets = synpop_widgets.GeoQueryWidgets(config.get('geoquery'))
```

```{code-cell} ipython3
zone_options = [str(key) for key in config["zone_options"].keys()]
source_zones_widget = synpop_widgets.get_source_zone_widget(zone_options, SUFFIX1)
target_zones_widget = synpop_widgets.get_target_zone_widget(zone_options, SUFFIX2)
```

```{code-cell} ipython3
shutdown_button = synpop_widgets.get_shutdown_widget(PORT)
```

```{code-cell} ipython3
#######################
##### MAIN WIDGET #####
#######################
```

```{code-cell} ipython3
aggregation_type_widget = synpop_widgets.get_aggregation_type_widget()
vmin_widget = synpop_widgets.get_vmin_widget()
vmax_widget = synpop_widgets.get_vmax_widget()
vcenter_widget = synpop_widgets.get_vcenter_widget()

w = widgets.interactive(map_plotter.update, {'manual': True, "manual_name": 'Update'}, **{
    'Variable': target_value_widget,
    'Comparison': plot_type_widget,
    'Aggregation': groupby_agg_widget,
    'Aggregation_Function': aggregation_type_widget,
    'Ref_variable': ref_var_widget,
    'FilterKey': geoquery_widgets.geo_query_widget,
    'FilterValue': geoquery_widgets.query_target_widget,
    'vmin': vmin_widget,
    'vmax': vmax_widget,
    'vcenter': vcenter_widget,
    'suffix1': source_zones_widget,
    'suffix2': target_zones_widget
})
```

```{code-cell} ipython3
#########################
##### WIDGET LAYOUT #####
#########################
```

```{code-cell} ipython3
output = w.children[-1]
update_button = w.children[-2]

for w_ in w.children:
    w_.description = ''
update_button.description = 'Update Plot'
```

```{code-cell} ipython3
html = markdown.markdown("""
This widget provides a simple yet flexible tool to:

- visualize mobi-zones (or a SynPop, aggregated in the background)
- run comparisons between two versions of mobi-zones (or two SynPops)

**Three kinds of comparison are available:**

- Absolute change: *scenario - reference*
- Relative change: *(scenario/reference)-1*
- Change of shares (only for comparisons and reference!=scenario): *(scenario/ref_variable) / (reference/ref_variable)*


**How to use it:**

1. Choose the variable to be analyzed
2. Choose type of comparison desired
3. Choose the level of aggregation
4. Filter results (optional): define the level, e.g. 'kt_name', and the filter, e.g. 'AG'
""")
info_w = HTML(html)
```

```{code-cell} ipython3
label_w = "30%"

widget_layout = VBox([
    HBox([
        VBox([
            HBox([widgets.Label('Source & Target Zones', layout=Layout(width=label_w)), source_zones_widget, target_zones_widget]),
            HBox([widgets.Label('Variable', layout=Layout(width=label_w)), target_value_widget]),
            HBox([widgets.Label('Aggregation', layout=Layout(width=label_w)), groupby_agg_widget]),
            HBox([widgets.Label('Comparison', layout=Layout(width=label_w)), plot_type_widget]),
            HBox([widgets.Label('Aggregation function', layout=Layout(width=label_w)), aggregation_type_widget]),
            HBox([widgets.Label('Reference variable', layout=Layout(width=label_w)), ref_var_widget]),
            HBox([widgets.Label('Filter (optional)', layout=Layout(width=label_w)),
                  geoquery_widgets.geo_query_widget, widgets.Label('=='), geoquery_widgets.query_target_widget]),
            HBox([widgets.Label('ColorScale (optional)', layout=Layout(width=label_w)),
                  vmin_widget, vmax_widget, vcenter_widget]),
            widgets.Label('  '),
            HBox([update_button, download_button, shutdown_button])]),
        info_w], layout=Layout(height="420px")),
    widgets.Label('  '),
    output
])
```

# Zonal Comparison Widget

```{code-cell} ipython3
widget_layout
```

```{code-cell} ipython3
w.update()
```
