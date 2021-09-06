"""
Contains the functions that generate the plots and other visualisations in the Jupyter notebooks UI tools.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn

# Logger settings set in __init__.py
from synpop import marginal_fitting

logger = logging.getLogger(__name__)

COLOUR_DICTS = {
    'is_swiss': {'true': 'firebrick', 'false': 'grey'},
    'current_edu': {'kindergarten': 'k', 'pupil_primary': 'C0', 'pupil_secondary': 'C1', 'student': 'C2',
                    'apprentice': 'C3', 'null': 'grey'},
    'current_job_rank': {'employee': 'C0', 'management': 'C1', 'apprentice': 'C3',  'null': 'grey'},
    'is_employed': {'true': 'C0', 'false': 'C1'}
}


def plot_people_employed_by_age(synpop_persons, bfs_active_people_by_age, title=''):
    loe_by_age = (synpop_persons.groupby(['age', 'loe'])['person_id'].count()
                  .unstack(['loe'])
                  .fillna(0)
                  .astype(int)
                  )
    loe_by_age.columns = ['0%', 'part_time', '100%']  # columns names as strings not categories
    loe_by_age = loe_by_age[['100%', 'part_time', '0%']]  # reordering columns

    bfs_active_people = bfs_active_people_by_age.rename('FSO Ref-Scenario - "Erwerbsquote"  (including unemployed)')

    # Plotting
    ax = loe_by_age.plot.bar(stacked=True, figsize=(18, 6), width=1, color=['C2', 'C0', 'grey'], alpha=.65, rot=90)
    if bfs_active_people is not None:
        bfs_active_people = bfs_active_people.copy()  # copy needed to change index
        ax = bfs_active_people.plot(style=':', marker='x', color='k', rot=90, ax=ax)

    ax.set_ylabel('People')
    _ = ax.set_title(title, pad=25, fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.xlim(0, 100)
    plt.ylim(0, 150_000)
    ax.grid(axis='y')
    _ = ax.legend(loc='upper right')

    return ax


def plot_cross_table(cross_table, main_feature_order=None, sec_feature_order=None, rename_dicts=None, title='',
                     cmap='Blues', show_pc=False):
    cross_table = cross_table.copy(deep=True)
    if rename_dicts is None:
        rename_dicts = {}

    # Pre-Processing the Cross Table
    cross_table.columns = cross_table.columns.astype(str)

    original_index_columns = list(cross_table.index.names)
    cross_table = cross_table.reset_index()
    try:
        cross_table = cross_table.rename(columns=rename_dicts[cross_table.columns.name])
    except KeyError:
        pass  # Key not in dict

    for feature, rename_dict in rename_dicts.items():
        try:
            cross_table[feature] = cross_table[feature].replace(rename_dict)
        except KeyError:
            pass  # Key not in dict

    cross_table = cross_table.set_index(original_index_columns)

    new_index = [' & '.join(pandas.Series(i).dropna()) for i in cross_table.index]
    try:
        id_null = new_index.index('')
        new_index[id_null] = 'other'
    except ValueError:
        pass

    cross_table.index = new_index

    # Sorting columns
    if main_feature_order:
        cross_table = cross_table[main_feature_order]

    # Sorting the rows
    if not sec_feature_order:
        # Sort by row with the most total counts
        cross_table['_sum'] = cross_table.sum(axis=1)
        cross_table = cross_table.sort_values('_sum', ascending=False)
        cross_table = cross_table.drop('_sum', axis=1)
    else:
        # Sort by pre-defined order
        sec_feature_order = [x for x in sec_feature_order if x in cross_table.index]  # only order existing columns
        cross_table = cross_table.reindex(sec_feature_order)

    # Plotting
    fig, ax = plt.subplots(figsize=(cross_table.shape[1] * 3, cross_table.shape[0] / 2))
    pc_cross_table = (cross_table.div(cross_table.sum(axis=1), axis=0) * 100).fillna(0)
    if show_pc:
        annotations = (pc_cross_table.applymap(lambda x: f'{x:.0f}%') +
                       cross_table.applymap(lambda x: str(f' ({x:,d})').replace(',', "'"))
                       )
        fmt = 's'
    else:
        annotations = cross_table.applymap(lambda x: str(f'{x:,d}').replace(',', "'"))
        fmt = 's'

    ax = seaborn.heatmap(pc_cross_table, annot=annotations, fmt=fmt,
                         cbar=True, square=False, linewidths=1, vmin=0, vmax=100, ax=ax, cmap=cmap,
                         cbar_kws={"shrink": 0.6, 'fraction': 0.1, 'aspect': 5, 'ticks': [0, 50, 100],
                                   'format': '%d %%'}
                         )
    _ = plt.xlabel('')
    _ = ax.set_title(title, pad=25, fontdict={'fontsize': 12, 'fontweight': 'bold'})
    _ = ax.xaxis.set_ticks_position('top')
    _ = plt.yticks(rotation=0)

    return ax, cross_table


def plot_ct1(persons, title=''):
    ct = marginal_fitting.compute_cross_table(persons,
                                              main_feature='is_employed',
                                              secondary_features=['current_edu']
                                              )

    sec_feature_order = ['kindergarten', 'pupil_primary', 'pupil_secondary', 'apprentice', 'student', 'null']
    ax, ct_plotted = plot_cross_table(ct,
                                      main_feature_order=['Employed', 'Not Employed'],
                                      sec_feature_order=sec_feature_order,
                                      rename_dicts={'is_employed': {'True': 'Employed', 'False': 'Not Employed'}},
                                      title=title
                                      )
    return ax, ct_plotted


def plot_ct2(persons, title=''):
    ct = marginal_fitting.compute_cross_table(persons,
                                              main_feature='current_job_rank',
                                              secondary_features='current_edu'
                                              )

    sec_feature_order = ['kindergarten', 'pupil_primary', 'pupil_secondary', 'apprentice', 'student', 'null']
    ax, ct_plotted = plot_cross_table(ct,
                                      sec_feature_order=sec_feature_order,
                                      title=title
                                      )
    return ax, ct_plotted


def plot_ct3(persons, title=''):
    ct = marginal_fitting.compute_cross_table(persons,
                                              main_feature='is_employed',
                                              secondary_features='current_job_rank'
                                              )

    ax, ct_plotted = plot_cross_table(ct,
                                      main_feature_order=['Employed', 'Not Employed'],
                                      sec_feature_order=['apprentice', 'employee', 'management', 'null'],
                                      rename_dicts={'is_employed': {'True': 'Employed', 'False': 'Not Employed'}},
                                      title=title
                                      )
    return ax, ct_plotted


# Visualising Businesses
def get_businesses_comparison_summary_by_category(df_ist, df_scenario, year_ist, year_scenario, agg_column, query=None):
    if query:
        df_ist = df_ist.query(query)
        df_scenario = df_scenario.query(query)

    stats_list = []
    for df, year in zip((df_ist, df_scenario), (year_ist, year_scenario)):
        stats = (df.groupby(agg_column).agg({'business_id': 'count',
                                             'jobs_endo': sum,
                                             'jobs_exo': sum,
                                             'fte_endo': sum,
                                             'fte_exo': sum}
                                            )
                 .rename(columns={'business_id': 'total_businesses'})
                 .astype(int)
                 )
        stats['total_jobs'] = stats['jobs_endo'] + stats['jobs_exo']
        stats['total_fte'] = stats['fte_endo'] + stats['fte_exo']

        stats = (pandas.melt(stats.reset_index(), id_vars=agg_column, var_name='statistic')
                 .set_index([agg_column, 'statistic'])
                 .iloc[:, 0]
                 .rename(year)
                 )
        stats_list.append(stats)

    summary = pandas.concat(stats_list, axis=1).sort_index()
    summary['% change'] = (((summary[year_scenario] - summary[year_ist]) / summary[year_scenario] * 100)
                           .fillna(0)
                           .round(1)
                           )

    return summary


def plot_businesses_comparison_by_category(df_ist, df_scenario, year_ist, year_scenario, agg_column, statistic,
                                           query=None, title='', figsize=(12, 6)):
    stats_per_cat = get_businesses_comparison_summary_by_category(df_ist, df_scenario, year_ist, year_scenario,
                                                                  agg_column, query)

    df = stats_per_cat.query('statistic == @statistic').reset_index(level=1, drop=True).iloc[:, :2]
    ax = df.plot.bar(figsize=figsize, rot=45)
    _ = plt.grid(axis='y')
    _ = ax.set_ylabel(statistic.replace('_', ' '))
    _ = ax.set_xlabel('')
    _ = ax.set_title(title, pad=25, fontdict={'fontsize': 16, 'fontweight': 'bold'})

    return ax


# Plots for marginal fitting notebooks
def save_figure(save_outputs, name, output_dir, dpi=150, fig_format='png', bbox_inches='tight', pad_inches=0.2):
    if save_outputs:
        fig_file_path = os.path.join(output_dir, name)
        plt.savefig(fig_file_path, dpi=dpi, format=fig_format, bbox_inches=bbox_inches, pad_inches=pad_inches)
        logging.info('Figure saved to file : {}'.format(fig_file_path))
        plt.close()


def plot_multi_class_feature_per_age(people_per_cat_and_age, colour_dict, ymax=None, y_grid=None, title='', **kwargs):
    colors = [colour_dict[col] for col in people_per_cat_and_age.columns]

    # Trimming the long flat tail
    people_per_cat_and_age = people_per_cat_and_age[people_per_cat_and_age.index <= 100]

    # Plotting
    ax = people_per_cat_and_age.plot.bar(figsize=(16, 6), stacked=True, width=1, color=colors, alpha=.65, rot=90,
                                         **kwargs
                                         )

    ax.set_ylabel('People')
    ax.set_xlabel('Age')

    if ymax:
        ax.set_ylim([0, ymax])

    _ = ax.set_title(title, pad=25, fontdict={'fontsize': 16, 'fontweight': 'bold'})

    if y_grid:
        ax.grid(axis='y')

    # Reset ticks
    ticks = [i for i in people_per_cat_and_age.index if i % 5 == 0]
    _ = ax.set_xticks(ticks)
    _ = ax.set_xticklabels(ticks)
    return ax


def plot_binary_feature_per_age_with_marginals(people_per_cat_and_age, expected_true_counts, colour_dict, ymax=None,
                                               y_grid=None, title='', **kwargs):
    """ To use when the BSF marginals are available for example. Always True first! """

    # Expected values based on BFS-predictions
    ax = expected_true_counts.replace(0, numpy.nan).plot(style=':', marker='x', color='k', rot=0)
    _ = ax.legend(loc='upper right')

    ax = plot_multi_class_feature_per_age(people_per_cat_and_age,
                                          colour_dict=colour_dict,
                                          ymax=ymax,
                                          y_grid=y_grid,
                                          title=title,
                                          ax=ax,
                                          **kwargs)
    ax.grid(axis='y')

    return ax


def plot_synpop_vs_bfs_avg_fte_per_age(synpop_persons, bfs_avg_fte_by_age, year, title=''):
    ax = synpop_persons.groupby('age')['level_of_employment'].mean().loc[:100].plot(figsize=(10, 6), color='r')
    ax = bfs_avg_fte_by_age.loc[:100].plot(style=':', marker='x', color='k', rot=0, ax=ax)
    ax.grid(axis='y')
    ax.set_ylim([0, 100])
    _ = ax.legend(['SynPop_{}: avg. loe'.format(year), 'BFS: avg. fte'])
    _ = ax.set_title(title, pad=25, fontdict={'fontsize': 16, 'fontweight': 'bold'})

    return ax


def plot_level_of_employment_age_heatmap(synpop_persons, title=''):
    bin_labels = ['0', '1-19', '20-39', '40-59', '60-79', '80-99', '100']
    synpop_persons['loe'] = pandas.cut(synpop_persons['level_of_employment'], [0, 1, 20, 40, 60, 80, 100, 101],
                                       right=False, labels=bin_labels)

    age_loe_matrix_abs = (synpop_persons.groupby(['age', 'loe']).count().iloc[:, 0]
                          .fillna(0)
                          .astype(int)
                          .reset_index()
                          .pivot('loe', 'age', 'person_id')
                          )

    age_loe_matrix_abs = age_loe_matrix_abs.sort_index(ascending=False)
    age_loe_matrix_abs = age_loe_matrix_abs.iloc[:, :100]
    age_loe_matrix_rel = (age_loe_matrix_abs / age_loe_matrix_abs.sum())

    fig, ax = plt.subplots(figsize=(16, 6))

    # Making a palette that starts totally white
    my_palette = seaborn.color_palette("Reds", 100)
    my_palette = [(1, 1, 1), ] + my_palette

    ax = seaborn.heatmap(age_loe_matrix_rel, cmap=my_palette, ax=ax)
    _ = ax.set_yticklabels(age_loe_matrix_rel.index.values, rotation=0)
    ax.grid(axis='y')
    ax.set_ylim([0, age_loe_matrix_rel.shape[0]])
    plt.gca().invert_yaxis()

    _ = ax.set_title(title, pad=25, fontdict={'fontsize': 16, 'fontweight': 'bold'})
    return ax


def generate_simple_grid_table(df, row_length=30.5, table_height=None, index=True):
    # Produces a HTML-Grid from DF. Works only in a jupyter environment.
    from ipyaggrid import Grid

    column_definitions = [{'field': df.index.name}] if index else []
    column_definitions = column_definitions + [{'field': c} for c in df.columns]

    grid_options = {
        'columnDefs': column_definitions,
        'defaultColDef': {'sortable': 'true', 'filter': 'true', 'resizable': 'true'},
        'enableRangeSelection': 'false',  # premium feature
        'rowSelection': 'multiple'
    }

    if table_height is None:
        table_height = max(80, int(len(df) * row_length))

    return Grid(grid_data=df,
                grid_options=grid_options,
                height=table_height,
                quick_filter=True,
                export_csv=True,
                export_excel=False,  # premium feature
                show_toggle_edit=False,
                export_mode='auto',
                index=index,
                theme='ag-theme-fresh'
                )
