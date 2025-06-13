"""Contains functions to generate the plots and widgets for the Jupyter UI tools."""
import logging
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import altair as alt
import ipyaggrid
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import seaborn
from altair.utils.schemapi import Undefined
from altair.utils.schemapi import UndefinedType

from synpop.fitting import marginal_fitting


# Logger settings set in __init__.py

logger = logging.getLogger(__name__)

COLOUR_DICTS = {
    "is_swiss": {"true": "firebrick", "false": "grey"},
    "current_edu": {
        "kindergarten": "k",
        "pupil_primary": "C0",
        "pupil_secondary": "C1",
        "student": "C2",
        "apprentice": "C3",
        "null": "grey",
    },
    "current_job_rank": {
        "employee": "C0",
        "management": "C1",
        "apprentice": "C3",
        "null": "grey",
    },
    "is_employed": {"true": "C0", "false": "C1"},
}


def plot_people_employed_by_age(
    synpop_persons: pd.DataFrame,
    bfs_active_people_by_age: pd.DataFrame,
    title: str = "",
) -> plt.Axes:
    """Plot the number of people employed by age compared to BFS."""
    if "loe" not in synpop_persons.columns:
        synpop_persons["loe"] = pd.cut(
            synpop_persons["level_of_employment"],
            bins=(-0.01, 0, 40.0, 80.0, 100),
            labels=["0%", "1-40%", "40-80%", "80-100%"],
        )
    loe_by_age = (
        synpop_persons.groupby(["age", "loe"])
        .size()
        .unstack("loe")
        .fillna(0)
        .astype(int)
    )
    loe_by_age.columns = loe_by_age.columns.astype(
        str
    )  # columns names as strings not categories
    loe_by_age = loe_by_age[loe_by_age.columns.tolist()[::-1]]  # invert column order

    bfs_active_people = bfs_active_people_by_age.rename(
        'FSO Ref-Scenario - "Erwerbsquote"  (including unemployed)'  # type: ignore
    )

    # Plotting
    axis = loe_by_age.plot.bar(
        stacked=True,
        figsize=(18, 6),
        width=1,
        color=["C2", "C1", "C0", "grey"],
        alpha=0.65,
        rot=90,
    )
    if bfs_active_people is not None:
        bfs_active_people = bfs_active_people.copy()  # copy needed to change index
        axis = bfs_active_people.plot(style=":", marker="x", color="k", rot=90, ax=axis)

    axis.set_ylabel("People")
    _ = axis.set_title(title, pad=25, fontdict={"fontsize": 16, "fontweight": "bold"})
    plt.xlim(0, 100)
    plt.ylim(0, 150_000)
    axis.grid(axis="y")
    _ = axis.legend(loc="upper right")

    return axis


def plot_cross_table(  # pylint: disable=too-many-locals
    cross_table: pd.DataFrame,
    main_feature_order: Optional[List[str]] = None,
    sec_feature_order: Optional[List[str]] = None,
    rename_dicts: Optional[Dict[str, Dict[str, str]]] = None,
    title: str = "",
    cmap: str = "Blues",
    show_pc: bool = False,
) -> Tuple[plt.Axes, pd.DataFrame]:
    """Plot a cross table with the given order of the main and secondary features."""
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

    new_index = [" & ".join(pd.Series(i).dropna()) for i in cross_table.index]  # type: ignore
    try:
        id_null = new_index.index("")
        new_index[id_null] = "other"
    except ValueError:
        pass

    cross_table.index = pd.Index(new_index)

    # Sorting columns
    if main_feature_order:
        cross_table = cross_table[main_feature_order]

    # Sorting the rows
    if not sec_feature_order:
        # Sort by row with the most total counts
        cross_table["_sum"] = cross_table.sum(axis=1)
        cross_table = cross_table.sort_values("_sum", ascending=False)
        cross_table = cross_table.drop("_sum", axis=1)
    else:
        # Sort by pre-defined order
        sec_feature_order = [
            x for x in sec_feature_order if x in cross_table.index
        ]  # only order existing columns
        cross_table = cross_table.reindex(sec_feature_order)

    # Plotting
    _, axis = plt.subplots(figsize=(cross_table.shape[1] * 3, cross_table.shape[0] / 2))
    pc_cross_table = (cross_table.div(cross_table.sum(axis=1), axis=0) * 100).fillna(0)
    if show_pc:
        annotations = pc_cross_table.applymap(
            lambda x: f"{x:.0f}%"
        ) + cross_table.applymap(lambda x: str(f" ({x})").replace(",", "'"))
        fmt = "s"
    else:
        annotations = cross_table.applymap(lambda x: str(f"{x}").replace(",", "'"))
        fmt = "s"

    axis = seaborn.heatmap(
        pc_cross_table,
        annot=annotations,
        fmt=fmt,
        cbar=True,
        square=False,
        linewidths=1,
        vmin=0,
        vmax=100,
        ax=axis,
        cmap=cmap,
        cbar_kws={
            "shrink": 0.6,
            "fraction": 0.1,
            "aspect": 5,
            "ticks": [0, 50, 100],
            "format": "%d %%",
        },
    )
    _ = plt.xlabel("")
    _ = axis.set_title(title, pad=25, fontdict={"fontsize": 12, "fontweight": "bold"})
    _ = axis.xaxis.set_ticks_position("top")
    _ = plt.yticks(rotation=0)

    return axis, cross_table


def plot_ct1(persons: pd.DataFrame, title: str = "") -> Tuple[plt.Axes, pd.DataFrame]:
    """Plot the cross table of the employment status and the current education."""
    if "is_employed" not in persons.columns:
        persons["is_employed"] = persons["level_of_employment"] > 0
    crosstable = marginal_fitting.compute_cross_table(
        persons, main_feature="is_employed", secondary_features=["current_edu"]
    )

    sec_feature_order = [
        "kindergarten",
        "pupil_primary",
        "pupil_secondary",
        "apprentice",
        "student",
        "null",
    ]
    axis, ct_plotted = plot_cross_table(
        crosstable,
        main_feature_order=["Employed", "Not Employed"],
        sec_feature_order=sec_feature_order,
        rename_dicts={"is_employed": {"True": "Employed", "False": "Not Employed"}},
        title=title,
    )
    return axis, ct_plotted


def plot_ct2(persons: pd.DataFrame, title: str = "") -> Tuple[plt.Axes, pd.DataFrame]:
    """Plot the cross table of the current job rank and the current education."""
    crosstable = marginal_fitting.compute_cross_table(
        persons, main_feature="current_job_rank", secondary_features="current_edu"
    )

    sec_feature_order = [
        "kindergarten",
        "pupil_primary",
        "pupil_secondary",
        "apprentice",
        "student",
        "null",
    ]
    axis, ct_plotted = plot_cross_table(
        crosstable, sec_feature_order=sec_feature_order, title=title
    )
    return axis, ct_plotted


def plot_ct3(persons: pd.DataFrame, title: str = "") -> Tuple[plt.Axes, pd.DataFrame]:
    """Plot the cross table of the employment status and the current job rank."""
    if "is_employed" not in persons.columns:
        persons["is_employed"] = persons["level_of_employment"] > 0
    crosstable = marginal_fitting.compute_cross_table(
        persons, main_feature="is_employed", secondary_features="current_job_rank"
    )

    axis, ct_plotted = plot_cross_table(
        crosstable,
        main_feature_order=["Employed", "Not Employed"],
        sec_feature_order=["apprentice", "employee", "management", "null"],
        rename_dicts={"is_employed": {"True": "Employed", "False": "Not Employed"}},
        title=title,
    )
    return axis, ct_plotted


def get_businesses_comparison_summary_by_category(
    dataframe_ist: pd.DataFrame,
    dataframe_scenario: pd.DataFrame,
    year_ist: int,
    year_scenario: int,
    agg_column: str,
    query: Optional[str] = None,
) -> pd.DataFrame:
    """Get a summary of the comparison of businesses by category."""
    if query:
        dataframe_ist = dataframe_ist.query(query)
        dataframe_scenario = dataframe_scenario.query(query)

    stats_list = []
    for dataframe, year in zip(
        (dataframe_ist, dataframe_scenario), (year_ist, year_scenario)
    ):
        stats = (
            dataframe.groupby(agg_column)
            .agg(
                {
                    "sector": "count",
                    "jobs_endo": sum,
                    "jobs_exo": sum,
                    "fte_endo": sum,
                    "fte_exo": sum,
                }
            )
            .rename(columns={"sector": "total_businesses"})
            .astype(int)
        )
        stats["total_jobs"] = stats["jobs_endo"] + stats["jobs_exo"]
        stats["total_fte"] = stats["fte_endo"] + stats["fte_exo"]

        stats = (
            pd.melt(stats.reset_index(), id_vars=agg_column, var_name="statistic")  # type: ignore
            .set_index([agg_column, "statistic"])
            .iloc[:, 0]
            .rename(year)
        )
        stats_list.append(stats)

    summary = pd.concat(stats_list, axis=1).sort_index()
    summary["% change"] = (
        ((summary[year_scenario] - summary[year_ist]) / summary[year_scenario] * 100)
        .fillna(0)
        .round(1)
    )

    return summary


def plot_businesses_comparison_by_category(  # pylint: disable=too-many-arguments
    dataframe_ist: pd.DataFrame,
    dataframe_scenario: pd.DataFrame,
    year_ist: int,
    year_scenario: int,
    agg_column: str,
    statistic,
    query: Optional[str] = None,
    title: str = "",
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Axes:
    """Plot the comparison of businesses by category."""
    stats_per_cat = get_businesses_comparison_summary_by_category(
        dataframe_ist, dataframe_scenario, year_ist, year_scenario, agg_column, query
    )

    dataframe = (
        stats_per_cat.query("statistic == @statistic")
        .reset_index(level=1, drop=True)
        .iloc[:, :2]
    )
    axis = dataframe.plot.bar(figsize=figsize, rot=45)
    _ = plt.grid(axis="y")
    _ = axis.set_ylabel(statistic.replace("_", " "))
    _ = axis.set_xlabel("")
    _ = axis.set_title(title, pad=25, fontdict={"fontsize": 16, "fontweight": "bold"})

    return axis


def save_figure(
    save: bool,
    name: str,
    output_dir: Union[Path, str],
    dpi: int = 150,
    fig_format: str = "png",
    bbox_inches: str = "tight",
    pad_inches: float = 0.2,
    display: bool = True,
) -> None:
    """Util function in Notebook to save current figure to a file."""
    if save:
        fig_file_path = os.path.join(output_dir, name)
        plt.savefig(
            fig_file_path,
            dpi=dpi,
            format=fig_format,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )
        logging.info("Figure saved to file : %s", fig_file_path)
        if not display:
            plt.close()


def plot_multi_class_feature_per_age(
    people_per_cat_and_age: pd.DataFrame,
    colour_dict: Dict[str, str],
    ymax: Optional[int] = None,
    y_grid: Optional[int] = None,
    title: str = "",
    **kwargs: Any,
) -> plt.Axes:
    """Plot the distribution of a multi-class feature per age."""
    colors = [colour_dict[col] for col in people_per_cat_and_age.columns]

    # Trimming the long flat tail
    people_per_cat_and_age = people_per_cat_and_age[people_per_cat_and_age.index <= 100]

    # Plotting
    axis = people_per_cat_and_age.plot.bar(
        figsize=(16, 6),
        stacked=True,
        width=1,
        color=colors,  # type: ignore
        alpha=0.65,
        rot=90,
        **kwargs,
    )

    axis.set_ylabel("People")
    axis.set_xlabel("Age")

    if ymax:
        axis.set_ylim([0, ymax])

    _ = axis.set_title(title, pad=25, fontdict={"fontsize": 16, "fontweight": "bold"})

    if y_grid:
        axis.grid(axis="y")

    # Reset ticks
    ticks = [i for i in people_per_cat_and_age.index if i % 5 == 0]
    _ = axis.set_xticks(ticks)
    _ = axis.set_xticklabels(ticks)
    return axis


def plot_binary_feature_per_age_with_marginals(
    people_per_cat_and_age: pd.DataFrame,
    expected_true_counts: pd.DataFrame,
    colour_dict: Dict[str, str],
    ymax: Optional[int] = None,
    y_grid: Optional[bool] = None,
    title: str = "",
    **kwargs: Any,
) -> plt.Axes:
    """Use when the BSF marginals are available for example. Keeps "True"-Category always first."""
    # Expected values based on BFS-predictions
    axis = expected_true_counts.replace(0, numpy.nan).plot(
        style=":", marker="x", color="k", rot=0
    )
    _ = axis.legend(loc="upper right")

    axis = plot_multi_class_feature_per_age(
        people_per_cat_and_age,
        colour_dict=colour_dict,
        ymax=ymax,
        y_grid=y_grid,
        title=title,
        ax=axis,
        **kwargs,
    )
    axis.grid(axis="y")

    return axis


def plot_synpop_vs_bfs_avg_fte_per_age(
    synpop_persons: pd.DataFrame,
    bfs_avg_fte_by_age: pd.DataFrame,
    year: int,
    title: str = "",
) -> plt.Axes:
    """Plot the comparison of average full-time equivalent (FTE) per age between SynPop and BFS."""
    axis = (
        synpop_persons.groupby("age")["level_of_employment"]
        .mean()
        .loc[:100]
        .plot(figsize=(10, 6), color="r")
    )
    axis = bfs_avg_fte_by_age.loc[:100].plot(
        style=":", marker="x", color="k", rot=0, ax=axis
    )
    axis.grid(axis="y")
    axis.set_ylim([0, 100])  # type: ignore
    _ = axis.legend([f"SynPop_{year}: avg. loe", "BFS: avg. fte"])
    _ = axis.set_title(title, pad=25, fontdict={"fontsize": 16, "fontweight": "bold"})

    return axis


def plot_level_of_employment_age_heatmap(
    synpop_persons: pd.DataFrame, title: str = ""
) -> plt.Axes:
    """Plot the level of employment (LOE) per age as a heatmap."""
    bin_labels = ["0", "1-19", "20-39", "40-59", "60-79", "80-99", "100"]
    synpop_persons["loe"] = pd.cut(
        synpop_persons["level_of_employment"],
        [0, 1, 20, 40, 60, 80, 100, 101],
        right=False,
        labels=bin_labels,
    )

    age_loe_matrix_abs = (
        synpop_persons.groupby(["age", "loe"])
        .count()
        .iloc[:, 0]
        .fillna(0)
        .astype(int)
        .reset_index()
        .pivot(index="loe", columns="age", values="person_id")
    )

    age_loe_matrix_abs = age_loe_matrix_abs.sort_index(ascending=False)
    age_loe_matrix_abs = age_loe_matrix_abs.iloc[:, :100]
    age_loe_matrix_rel = age_loe_matrix_abs / age_loe_matrix_abs.sum()

    _, axis = plt.subplots(figsize=(16, 6))

    # Making a palette that starts totally white
    my_palette = [(1, 1, 1),] + seaborn.color_palette(
        "Reds", 100
    )  # type: ignore

    axis = seaborn.heatmap(age_loe_matrix_rel, cmap=my_palette, ax=axis)
    _ = axis.set_yticklabels(age_loe_matrix_rel.index.values, rotation=0)
    axis.grid(axis="y")
    axis.set_ylim([0, age_loe_matrix_rel.shape[0]])  # type: ignore
    plt.gca().invert_yaxis()

    _ = axis.set_title(title, pad=25, fontdict={"fontsize": 16, "fontweight": "bold"})
    return axis


def plot_level_of_employment_distribution(
    persons: pd.DataFrame, persons_ist: pd.DataFrame, year: int, year_ist: int
) -> plt.Axes:
    """Plot the level of employment distribution."""
    title = f"Level of Employment Distribution - SynPop {year_ist} vs SynPop {year}"

    bins = list(range(5, 101, 5))
    labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
    loe = (
        pd.cut(
            persons.query("is_employed")["level_of_employment"],
            bins=bins,
            labels=labels,
        )
        .value_counts()
        .div(sum(persons["is_employed"]))
        .mul(100)  # share of employed
        .sort_index()
    )
    loe_ist = (
        pd.cut(
            persons_ist.query("is_employed")["level_of_employment"],
            bins=bins,
            labels=labels,
        )
        .value_counts()
        .div(sum(persons_ist["is_employed"]))
        .mul(100)  # share of employed
        .sort_index()
    )
    axis = pd.concat([loe, loe_ist], axis=1, keys=[year_ist, year]).plot.bar(
        figsize=(10, 6)
    )
    plt.locator_params(axis="x", nbins=7)
    plt.xticks(rotation=0)
    plt.xlabel("Level of Employment ranges (%)", fontsize=12)
    plt.ylabel("Frequency (%)", fontsize=11)
    _ = axis.set_title(title, pad=25, fontdict={"fontsize": 16, "fontweight": "bold"})
    return axis


def generate_simple_grid_table(  # pylint: disable=too-many-locals
    dataframe: pd.DataFrame,
    colnames: Optional[List[str]] = None,
    header_height: int = 25,
    column_toggles: Optional[Union[List[str], Dict[str, List[str]]]] = None,
    col_widths: Optional[Dict[str, int]] = None,
    hidden_cols: Optional[List[str]] = None,
    width: Optional[int] = None,
) -> ipyaggrid.Grid:
    """Create a HTML-Grid (jupyter widget) from a DataFrame. Works only in a Jupyter environment."""
    dataframe = dataframe.copy()
    if isinstance(colnames, list):
        dataframe.columns = colnames
    elif isinstance(colnames, dict):
        dataframe = dataframe.rename(columns=colnames)
    column_definitions = {c: {"field": c} for c in dataframe.columns}
    if col_widths is not None:
        for col, col_width in col_widths.items():
            column_definitions[col] = {
                "field": col,
                "width": col_width,
                "suppressSizeToFit": True,
            }
    if hidden_cols is not None:
        for col in hidden_cols:
            column_definitions[col] = {**column_definitions[col], "hide": True}

    default_col_def = {
        "flex": 1,
        "sortable": "true",
        "filter": "true",
        "resizable": "true",
        "headerComponentParams": {  # wrappable header
            "template": (
                '<div class="ag-cell-label-container" role="presentation">'
                + '  <span ref="eMenu" class="ag-header-icon ag-header-cell-menu-button"></span>'
                + '  <div ref="eLabel" class="ag-header-cell-label" role="presentation">'
                + '    <span ref="eSortOrder" class="ag-header-icon ag-sort-order"></span>'
                + '    <span ref="eSortAsc" class="ag-header-icon ag-sort-ascending-icon"></span>'
                + '    <span ref="eSortDesc" class="ag-header-icon ag-sort-descending-icon"></span>'
                + '    <span ref="eSortNone" class="ag-header-icon ag-sort-none-icon"></span>'
                + '    <span ref="eText" class="ag-header-cell-text" '
                'role="columnheader" style="white-space: normal;"></span>'
                + '    <span ref="eFilter" class="ag-header-icon ag-filter-icon"></span>'
                + "  </div>"
                + "</div>"
            )
        },
    }

    grid_options = {
        "columnDefs": list(column_definitions.values()),
        "defaultColDef": default_col_def,
        "enableRangeSelection": "false",  # paid feature
        "rowSelection": "multiple",
        "headerHeight": header_height,
    }

    grid_kwargs: Dict[str, Any] = {
        "grid_data": dataframe,
        "grid_options": grid_options,
        "quick_filter": True,
        "export_csv": True,
        "export_excel": False,  # paid feature
        "show_toggle_edit": False,
        "export_mode": "auto",
        "index": False,
        "width": width or "90%",
        "theme": "ag-theme-fresh",
    }

    if len(dataframe) > 50:
        grid_kwargs["grid_options"] = {
            **grid_kwargs["grid_options"],
            "pagination": True,
            "paginationPageSize": 50,
        }
        grid_kwargs["height"] = header_height + 600
    else:
        grid_kwargs["height"] = header_height + int(len(dataframe) * 28.5)

    if column_toggles is not None:
        buttons = []
        if isinstance(column_toggles, list):
            column_toggles = {c: [c] for c in column_toggles}
        if isinstance(column_toggles, dict):
            for group, cols in column_toggles.items():
                action = f"""
                var colNames = "{','.join(cols)}".split(',');  // import to js via string
                var column = gridOptions.columnApi.getColumn(colNames[0]);
                if (!column.isVisible()){{  // field not present, include
                    colNames.forEach(c => gridOptions.columnApi.setColumnVisible(c, true));
                }} else {{  // field present, remove
                    colNames.forEach(c => gridOptions.columnApi.setColumnVisible(c, false));
                }};
                gridOptions.api.sizeColumnsToFit();
                """
                buttons.append({"name": group, "action": action})
        grid_kwargs["menu"] = {"buttons": buttons}

    return ipyaggrid.Grid(**grid_kwargs)


def altair_scatter(  # pylint: disable=too-many-locals,too-many-arguments
    dataframe: pd.DataFrame,
    x_col: str,
    y_col: str,
    table: bool = True,
    columns: Optional[List[str]] = None,
    title: UndefinedType = Undefined,
    dropdown_col: Optional[str] = None,
    dropdown_name: str = "Filter Canton ",
    dropdown_all_option: bool = True,
) -> Union[alt.HConcatChart, alt.Chart]:
    """Create interactive scatter plot with a dropdown filter. Works only in Jupyter environment.

    If the number of options in the dropdown is huge, this is still possible via HTML's <datalist>
    https://stackoverflow.com/questions/61393699/is-it-possible-to-create-an-altair-binding-to-a-datalist-element-instead-of-sele
    """
    if not columns:
        columns = dataframe.columns.tolist()
    assert columns is not None

    # Brush for selection
    brush = alt.selection(type="interval")  # type: ignore

    # Zoom interaction with ctrl-Key
    interaction = alt.selection(
        type="interval",  # type: ignore
        bind="scales",
        on="[mousedown[event.ctrlKey], mouseup] > mousemove",
        translate="[mousedown[event.ctrlKey], mouseup] > mousemove!",
        zoom="wheel![event.ctrlKey]",
    )

    # Scatter Plot
    chart = (
        alt.Chart(dataframe, title=title)  # type: ignore
        .mark_point()
        .encode(
            x=f"{x_col}:Q", y=f"{y_col}:Q", tooltip=columns, detail=f"{dropdown_col}:N"
        )
        .add_selection(brush, interaction)
    )
    if table:
        # Base chart for data tables
        ranked_text = (
            alt.Chart(dataframe)  # type: ignore
            .mark_text()
            .encode(y=alt.Y("row_number:O", axis=None))  # type: ignore
            .transform_window(row_number="row_number()")
            .transform_filter(brush)
            .transform_window(rank="rank(row_number)")
            .transform_filter(alt.datum.rank < 20)
        )

        # Data Tables
        cols = [ranked_text.encode(text=f"{c}:N").properties(title=c) for c in columns]
        text = alt.hconcat(*cols)  # Combine data tables

        # Build chart
        chart = alt.hconcat(chart, text).resolve_legend(color="independent")

    # Dropdown filter
    if dropdown_col:
        options = dataframe[dropdown_col].unique().tolist()

        if dropdown_all_option:
            options = ["All"] + options
            dropdown_selection = alt.selection_single(
                fields=[dropdown_col],
                bind=alt.binding_select(options=options, name=dropdown_name),
                init={dropdown_col: options[0]},
            )
            chart = chart.add_selection(dropdown_selection).transform_filter(
                f"({getattr(dropdown_selection, dropdown_col)}[0] == 'All') || \
                    ({getattr(dropdown_selection, dropdown_col)}[0] == datum.{dropdown_col})"
            )
        else:
            dropdown_selection = alt.selection_single(
                fields=[dropdown_col],
                bind=alt.binding_select(options=options, name=dropdown_name),
                init={dropdown_col: options[0]},
            )
            chart = chart.add_selection(dropdown_selection).transform_filter(
                dropdown_selection
            )

    # Display
    return chart
