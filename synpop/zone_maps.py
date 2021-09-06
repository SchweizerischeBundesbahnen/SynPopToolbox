"""
Contains classes to generate geographical visualizations (maps).
"""

import logging
import warnings
from pathlib import Path
from typing import Callable

import geopandas
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
LAKE_COLOR = '#A0C8E6'
DEFAULT_COMMUNE_SHP = HERE / 'shp/BFS_CH14_Gemeinden.shp'
DEFAULT_CANTON_SHP = HERE / 'shp/BFS_CH14_Kantone.shp'
DEFAULT_LAKE_SHP = HERE / 'shp/BFS_CH14_Seen.shp'
MOBI_ZONES = HERE / 'shp/zones.shp'
DEFAULT_GEO_COLS = ['mun_name', 'agglo_id', 'agglo_name', 'amgr_id', 'amgr_name', 'amr_id', 'amr_name',
                    'msr_id', 'msr_name', 'mun_id', 'sl3_id', 'sl3_name', 'kt_id', 'kt_name']

# colormaps (one linear and one for two transitions in case of large growths)
DEFAULT_CMAP = 'RdBu'
# Two transitions CMAP: non-sequential hsv (we want two transitions instead of one, but not a cycle)
CMAP_TWO_TRANSITIONS = colors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n='hsv', a=0, b=0.8),
    cm.get_cmap('hsv')(np.linspace(0, 0.8, 100)))
CMAP_TWO_TRANSITIONS = colors.LinearSegmentedColormap.from_list(
    'CMAP_TWO_TRANSITIONS', np.vstack((CMAP_TWO_TRANSITIONS(np.linspace(0, 0.22, 100)),
                                       CMAP_TWO_TRANSITIONS(np.linspace(0.23, 1.0, 100)))))


class SwissZoneMap:

    def __init__(self, outline_communes=False, outline_cantons=True, draw_lakes=True,
                 commune_shp_path=None, canton_shp_path=None, lake_shp_path=None):
        """
        This class is able to draw a map of Switzerland to visualise some zonal attribute.

        :param outline_communes: Draw commune outlines
        :param outline_cantons: Draw cantons outlies
        :param draw_lakes:  Draw major water bodies
        :param commune_shp_path: use only to overwrite the default shapefile
        :param canton_shp_path: use only to overwrite the default shapefile
        :param lake_shp_path: use only to overwrite the default shapefile
        """
        self.outline_communes = outline_communes
        self.outline_cantons = outline_cantons
        self.draw_lakes = draw_lakes

        self.commune_shp_path = commune_shp_path or DEFAULT_COMMUNE_SHP  # if None, use default
        self.canton_shp_path = canton_shp_path or DEFAULT_CANTON_SHP  # if None, use default
        self.lake_shp_path = lake_shp_path or DEFAULT_LAKE_SHP  # if None, use default

        self.cantons = self.communes = self.lakes = None
        if self.outline_communes:
            self.communes = geopandas.read_file(self.commune_shp_path)
        if self.outline_cantons:
            self.cantons = geopandas.read_file(self.canton_shp_path)
            self.cantons = self.cantons.drop([26, 27])  # drop LIE and Enclaves
        if self.draw_lakes:
            self.lakes = geopandas.read_file(self.lake_shp_path)

    def draw(self, geodataframe: geopandas.GeoDataFrame, column: str, cmap=None, vmin=None, vmax=None, vcenter=None,
             alpha=1.0, legend=True, title=None, labeler_func: Callable = None, labeler_filter: Callable = None):
        if not cmap and vcenter is not None:
            cmap = CMAP_TWO_TRANSITIONS
        elif not cmap:
            cmap = DEFAULT_CMAP
        if not vmin:
            vmin = geodataframe[column].quantile(0.01)
        if not vmax:
            vmax = geodataframe[column].quantile(0.99)

        # without vcenter: simple linear normalization (same as without norm)
        if vcenter is None:
            vcenter = (vmax+vmin)/2
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        # Preparing the figure and set copyright statement at extremes
        f, ax = plt.subplots(figsize=(15, 10))
        miny = geodataframe.geometry.bounds['miny'].min()
        maxx = geodataframe.geometry.bounds['maxx'].max()
        ax.text(maxx, miny, u"SBB, P-V-APL-MEP\nSIMBA.MOBi", fontsize=12)

        # ensure correct CRS
        if '2056' not in str(geodataframe.crs) and 'CH1903+' not in str(geodataframe.crs):
            geodataframe = geodataframe.to_crs(epsg=2056)

        # Plotting the colored zones
        geodataframe.plot(ax=ax, column=column, linewidth=0.03, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, alpha=alpha)
        ax.axis('off')
        ax.axis('tight')

        # Add Title
        if title:
            _ = ax.set_title(title, fontdict={'fontsize': 18, 'fontweight': 'bold'})

        if self.outline_communes:
            self.communes.plot(ax=ax, facecolor="None", edgecolor="black", lw=0.5)

        if self.outline_cantons:
            self.cantons.plot(ax=ax, facecolor="None", edgecolor="black", lw=0.5)

        if self.draw_lakes:
            self.lakes.plot(ax=ax, facecolor=LAKE_COLOR, edgecolor="None")

        # Apply annotations if labeler function is provided
        if labeler_func is not None:
            if labeler_filter is not None:
                labeler_filter(geodataframe).apply(labeler_func, axis=1, args=(ax,))
            else:
                geodataframe.apply(labeler_func, axis=1, args=(ax,))

        # Add single colorbar
        if legend:
            fig = ax.get_figure()
            cax = fig.add_axes([0.1, 0.9, 0.2, 0.01])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm._A = []
            cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
            cbar.ax.set_title(column, {'fontsize': 12})
            for cbar_xticklabel in cbar.ax.get_xticklabels():
                cbar_xticklabel.set(rotation=45)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()

        return ax

    def draw_cantons(self, df, column, cmap=None, vmin=-10, vmax=10, alpha=1.0, legend=True, title=None):
        """
        Draws a map of cantons coloring each canton based on a given column attribute. The DataFrame has to have all
        canton full names as index.
        """
        if self.cantons is None:
            self.cantons = geopandas.read_file(self.canton_shp_path)
            self.cantons = self.cantons.drop([26, 27])  # drop LIE and Enclaves
        assert set(df.index) == set(self.cantons['KTNAME']), 'DataFrame must have all canton full names as index!'

        geo_df = self.cantons.merge(df, how='left', left_on='KTNAME', right_index=True)
        ax = self.draw(geo_df, column, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, legend=legend, title=title)
        return ax, geo_df

    def draw_communes(self, df, column, cmap=None, vmin=-10, vmax=10, alpha=1.0, legend=True, title=None):
        """
        Draws a map of communes coloring each commune based on a given column attribute. The DataFrame has to have all
        commune full names as index.
        """
        if self.communes is None:
            self.communes = geopandas.read_file(self.commune_shp_path)
        assert set(df.index) == set(self.communes['N_Gem']), 'DataFrame must have all commune full names as index!'

        geo_df = self.communes.merge(df, how='left', left_on='N_Gem', right_index=True)
        ax = self.draw(geo_df, column, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, legend=legend, title=title)
        return ax, geo_df


class ZonalComparison(SwissZoneMap):

    def __init__(self, zones1, zones2, groupby_sum_cols=None, **kwargs):
        super().__init__(**kwargs)
        self.zones1 = zones1
        self.zones2 = zones2
        self.groupby_sum_cols = groupby_sum_cols
        if not groupby_sum_cols:
            cols = set(zones1.columns.to_list())
            cols = set.intersection(cols, zones2.columns.to_list())
            col_is_numeric = lambda df, col: 'int' in str(df[col].dtype) or 'float' in str(df[col].dtype)
            self.groupby_sum_cols = [c for c in cols if col_is_numeric(zones1, c) and col_is_numeric(zones2, c)]
        self.groupby_sum_cols = [c for c in self.groupby_sum_cols if c not in DEFAULT_GEO_COLS]
        self.dissolve_cache = {}

    def dissolve_zones(self, groupby_sum, zones1, zones2):
        if groupby_sum not in self.dissolve_cache.keys():
            cols = set(self.groupby_sum_cols + [groupby_sum, 'geometry'])
            cols = set.intersection(cols, zones1.columns.to_list(), zones2.columns.to_list())
            self.dissolve_cache[groupby_sum] = (
                zones1[cols].dissolve(by=groupby_sum, aggfunc='sum'),
                zones2[cols].dissolve(by=groupby_sum, aggfunc='sum')
            )
        return self.dissolve_cache[groupby_sum]

    def get_comparison(self, target_var, groupby_sum=None, geo_query=None):
        # not implemented: add a further class for specific comparisons (i.e. variable)
        raise NotImplementedError

    def calc_stats(self, target_var, groupby_sum=None, geo_query=None):
        if geo_query is not None:
            zones1 = self.zones1.query(geo_query)
            zones2 = self.zones2.query(geo_query)
        else:
            zones1, zones2 = self.zones1.copy(), self.zones2.copy()
        if groupby_sum == 'TOTAL':
            zones1, zones2 = zones1.sum().to_frame('TOTAL').T, zones2.sum().to_frame('TOTAL').T
        elif groupby_sum is not None:
            zones1, zones2 = self.dissolve_zones(groupby_sum, zones1, zones2)

        zones = zones1.copy()
        zones[target_var + '_other'] = zones2[target_var]
        zones[target_var + '_rel'] = zones[target_var] / zones['pop_total']
        zones[target_var + '_rel_other'] = zones[target_var + '_other'] / zones2['pop_total']
        zones['diff_abs'] = (zones2[target_var] - zones1[target_var]).fillna(0)
        zones['diff_rel'] = (zones2[target_var] / zones1[target_var] - 1).replace([np.inf, -np.inf], np.nan).fillna(0)
        zones['diff_rel_shares'] = (zones[target_var + '_rel_other'] / zones[target_var + '_rel'] - 1)
        zones['diff_rel_shares'] = zones['diff_rel_shares'].replace([np.inf, -np.inf], np.nan).fillna(0)
        return zones

    @staticmethod
    def comparison_df(df, target_var, top_display=None):
        df['diff_rel (%)'] = df['diff_rel'] * 100  # .apply(lambda s: "{0:.1%}".format(s))
        df[target_var + '_rel (%)'] = df[target_var + '_rel'] * 100  # .apply(lambda s: "{0:.1%}".format(s))
        df[target_var + '_rel_other (%)'] = df[target_var + '_rel_other'] * 100  # .apply(lambda s: "{0:.1%}".format(s))
        df['diff_rel_shares (%)'] = df['diff_rel_shares'] * 100  # .apply(lambda s: "{0:.1%}".format(s))
        top_df = df.sort_values('pop_total', ascending=False).round(1)
        if top_display:
            top_df = top_df.head(top_display)
        return top_df[[target_var, target_var + '_other', target_var + '_rel (%)',
                       target_var + '_rel_other (%)', 'diff_abs', 'diff_rel (%)', 'diff_rel_shares (%)']]

    def plot_diff(self, target_var, plot_type='rel', groupby_sum=None, geo_query=None, title=None,
                  vmin=None, vmax=None, vcenter=None, label=False, top_display=0, cmap=None):
        # plot_type: 'abs' for absolue, 'rel' for relative, 'share' for growth of shares (in relation to pop_total)
        # groupby_sum: geographical aggregation
        # geo_query: filter DF with pandas' query method
        # vmin, vmax and vcenter: plot bounds. If None, .99 quantiles taken
        # colorscheme: not currently used
        # label: 'id' for geographical ID or True for current variable
        # top_display: display DF with top zones based on pop_total

        zones = self.calc_stats(target_var, groupby_sum, geo_query)

        diff_var = {'rel': 'diff_rel', 'abs': 'diff_abs', 'shares': 'diff_rel_shares'}[plot_type]
        title_var = {'rel': 'relative', 'abs': 'absolute', 'shares': 'relative of shares'}[plot_type]
        if title is None:
            title = f'{target_var}: {title_var} differences'

        labeler_filter = None
        labeler_func = None
        if label:
            def labeler_func(row, ax):
                ax.annotate(fontsize=8, s=row.name if label == 'id' else round(row[diff_var], 1),
                            xy=row['geometry'].centroid.coords[0], ha='center')
            if top_display > 0:
                def labeler_filter(df):
                    return df.sort_values('pop_total', ascending=False).head(top_display)

        _ = self.draw(zones, diff_var, cmap=cmap, vmin=vmin, vmax=vmax, vcenter=vcenter, title=title,
                       labeler_func=labeler_func, labeler_filter=labeler_filter)

        if top_display:
            zones = zones.append(self.calc_stats(target_var, groupby_sum='TOTAL', geo_query=geo_query), sort=True)
            return self.comparison_df(zones, target_var, top_display + 1)

    def get_rmse(self, target_var, groupby_sum=None, geo_query=None):
        stats = self.calc_stats(target_var, groupby_sum, geo_query)
        cc = stats[target_var]
        mm = stats[target_var + '_other']
        return np.sqrt(np.sum(np.power(np.subtract(cc, mm), 2)) / len(cc))

    def get_mae(self, target_var, groupby_sum=None, geo_query=None):
        stats = self.calc_stats(target_var, groupby_sum, geo_query)
        cc = stats[target_var]
        mm = stats[target_var + '_other']
        return np.sum(np.abs(np.subtract(cc, mm))) / len(cc)


def aggregate_synpop(persons_df: pd.DataFrame,
                     businesses_df: pd.DataFrame = None, age_groups: bool = False) -> pd.DataFrame:
    # Prepare container
    zones = geopandas.read_file(MOBI_ZONES)
    zone_ids = zones['zone_id'].to_list()
    aggregates_df = pd.Series(zone_ids).to_frame('zone_id').set_index('zone_id')

    # Mobility tools per Zone
    aggregates_df[["pop_caravl", "pop_ga", "pop_ht", "pop_va"]] = (
        persons_df.groupby('zone_id')[['car_available', 'has_ga', 'has_ht', 'has_va']].sum())
    aggregates_df['pop_va_ht'] = persons_df.loc[persons_df['has_ht'] & persons_df['has_va'],
                                                'zone_id'].value_counts()

    # Students per Zone
    aggregates_df["pop_pup_1"] = persons_df.loc[persons_df['current_edu'] == 'pupil_primary',
                                                'zone_id'].value_counts().reindex(aggregates_df.index)
    aggregates_df["pop_pup_2"] = persons_df.loc[persons_df['current_edu'] == 'pupil_secondary',
                                                'zone_id'].value_counts().reindex(aggregates_df.index)
    aggregates_df["pop_stud_3"] = persons_df.loc[persons_df['current_edu'] == 'student',
                                                 'zone_id'].value_counts().reindex(aggregates_df.index)
    aggregates_df["pop_appr"] = persons_df.loc[persons_df['current_edu'] == 'apprentice',
                                               'zone_id'].value_counts().reindex(aggregates_df.index)

    # Total population
    aggregates_df['pop_total'] = persons_df['zone_id'].value_counts().reindex(aggregates_df.index)
    aggregates_df['pop_empl'] = persons_df.loc[persons_df['level_of_employment'] > 0,
                                               'zone_id'].value_counts().reindex(aggregates_df.index)

    # Population totals per age groups per Zone
    if age_groups:
        age_groups = [0, 18, 25, 45, 65, 75, 1000]
        AGE_GROUP = 'age_group'
        persons_df[AGE_GROUP] = pd.cut(persons_df['age'], age_groups, right=False,
                                       labels=['pop_0017', 'pop_1824', 'pop_2544', 'pop_4564', 'pop_6574', 'pop_75xx'])
        person_age_group_df = (persons_df.groupby(['zone_id', AGE_GROUP])['age'].count().
                               reset_index().pivot(columns=AGE_GROUP, index='zone_id'))
        person_age_group_df.columns = person_age_group_df.columns.droplevel(0).astype(str)
        aggregates_df[['pop_0017', 'pop_1824', 'pop_2544', 'pop_4564', 'pop_6574', 'pop_75xx']] = (
            person_age_group_df[['pop_0017', 'pop_1824', 'pop_2544', 'pop_4564', 'pop_6574', 'pop_75xx']])

    # Businesses aggregates
    if businesses_df is not None:
        aggregates_df['jobs_endo'] = businesses_df.groupby('zone_id')['jobs_endo'].sum()
        aggregates_df['jobs_total'] = aggregates_df['jobs_endo'] + businesses_df.groupby('zone_id')['jobs_exo'].sum()
        aggregates_df['fte_endo'] = businesses_df.groupby('zone_id')['fte_endo'].sum()
        aggregates_df['fte_total'] = aggregates_df['fte_endo'] + businesses_df.groupby('zone_id')['fte_exo'].sum()

    return zones.set_index('zone_id').join(aggregates_df)
