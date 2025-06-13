# pylint: disable=too-many-lines
"""Analysis of the fitting results."""
import logging
import math
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import numpy
import pandas as pd

from synpop.visualization import visualisations
from synpop.visualization import zone_maps

# Logger settings set in __init__.py
logger = logging.getLogger(__name__)


# Building marginal tables ##
def compute_comparison_summary(
    persons: pd.DataFrame,
    persons_ist: pd.DataFrame,
    year: int,
    year_ist: int,
    groupby: str,
) -> pd.DataFrame:
    """Compute a comparison table of the two populations aggregated by features given."""
    counts = persons.groupby(groupby).size().rename(f"counts {year}")
    counts_ist = persons_ist.groupby(groupby).size().rename(f"counts {year_ist}")

    growth = ((counts / counts_ist) - 1).mul(100).round(1).rename("absolute growth (%)")

    prop = (counts / persons.shape[0]).mul(100).round(1).rename(f"proportion {year}")
    prop_ist = (
        (counts_ist / persons_ist.shape[0])
        .mul(100)
        .round(1)
        .rename(f"proportion {year_ist}")
    )

    summary_table = pd.concat(
        [counts_ist, counts, growth, prop_ist, prop], axis=1
    ).fillna(0)
    summary_table.index.name = groupby

    assert isinstance(summary_table, pd.DataFrame)
    return summary_table


def generate_fitting_analysis(  # pylint: disable=too-many-arguments
    persons: pd.DataFrame,
    persons_fixed: pd.DataFrame,
    persons_ist: pd.DataFrame,
    year: int,
    year_ist: int,
    target_variable: str,
    plots_output_path: Union[Path, str],
    fitting_segments: Optional[List[str]] = None,
    expected_counts: Optional[pd.DataFrame] = None,
) -> None:
    """Generate plots and summary tables for the fitting results."""
    # outputs location
    path = Path(plots_output_path) / target_variable
    path.mkdir(exist_ok=True, parents=True)

    # store summary tables as CSV
    compute_comparison_summary(
        persons, persons_ist, year, year_ist, target_variable
    ).to_csv(path / "summary_table_raw.csv")
    compute_comparison_summary(
        persons_fixed, persons_ist, year, year_ist, target_variable
    ).to_csv(path / "summary_table_fixed.csv")
    is_binary_feature = (
        persons[target_variable].drop_duplicates().isin([True, False]).all()
    )

    for _persons, _year in zip(  # type: ignore
        [persons, persons_fixed, persons_ist], [f"{year}-Raw", year, year_ist]
    ):
        is_prognose_year = int("".join(c for c in str(_year) if c.isnumeric())) > 2020
        if target_variable in ("current_edu", "current_job_rank", "is_employed"):
            # generate cross-tables plots
            generate_crosstable_plots(_persons, _year, path)

        if fitting_segments is not None and "age" in fitting_segments:
            # generate pyramid plots
            generate_agepyramid_plots(
                _persons, int(_year), target_variable, path, expected_counts
            )

        if fitting_segments is not None and all(
            [
                "KT_full" in fitting_segments,
                is_binary_feature,
                is_prognose_year,
                expected_counts is not None,
            ]
        ):
            # generate map plots
            generate_map_plots(
                _persons, int(_year), target_variable, expected_counts, path  # type: ignore
            )


def generate_crosstable_plots(
    persons: pd.DataFrame, year: Union[int, str], path: Union[Path, str]
) -> None:
    """Generate post-Fitting cross-table plots for the given persons table."""
    _ = visualisations.plot_ct1(persons, title=f"SynPop{year}: Education vs Employment")
    visualisations.save_figure(True, f"SynPop{year}_CT1.png", path, display=False)
    _ = visualisations.plot_ct2(persons, title=f"SynPop{year}: Education vs Job-Rank")
    visualisations.save_figure(True, f"SynPop{year}_CT2.png", path, display=False)
    _ = visualisations.plot_ct3(persons, title=f"SynPop{year}: Job-Rank vs Employment")
    visualisations.save_figure(True, f"SynPop{year}_CT3.png", path, display=False)


def generate_agepyramid_plots(
    persons: pd.DataFrame,
    year: int,
    target_variable: str,
    path: Union[Path, str],
    expected_counts: Optional[pd.DataFrame] = None,
) -> None:
    """Generate post-Fitting age-pyramid plots for the given persons table."""
    is_binary_feature = (
        persons[target_variable].drop_duplicates().isin([True, False]).all()
    )
    is_prognose_year = int("".join(c for c in str(year) if c.isnumeric())) > 2020

    counts_per_cat = (
        persons.groupby(["age", target_variable])["person_id"].count().unstack(level=1)
    )
    if is_binary_feature:
        counts_per_cat = counts_per_cat.rename(
            columns={True: "true", False: "false"}, errors="ignore"
        )
        counts_per_cat = counts_per_cat[["true", "false"]]

    axis = None
    if is_binary_feature and is_prognose_year and expected_counts is not None:
        # Expected values based on BFS-predictions
        axis = (
            expected_counts.groupby("age")[True]  # type: ignore
            .sum()
            .rename("Expected")
            .replace(0, numpy.nan)
            .plot(style=":", marker="x", color="k", rot=0)
        )
        _ = axis.legend(loc="upper right")

    _ = visualisations.plot_multi_class_feature_per_age(
        counts_per_cat,
        colour_dict=visualisations.COLOUR_DICTS[target_variable],
        ymax=150_000,
        ax=axis,
        y_grid=is_binary_feature,
        title=f"{target_variable} by Age - SynPop{year}",
    )
    visualisations.save_figure(
        True, f"SynPop{year}_{target_variable}_by_age.png", path, display=False
    )


def generate_map_plots(
    persons: pd.DataFrame,
    year: int,
    target_variable: str,
    expected_counts: pd.DataFrame,
    path: Union[Path, str],
) -> None:
    """Generate post-Fitting maps for persons table in comparison to reference values."""
    mapper = zone_maps.SwissZoneMap()
    canton = pd.concat(
        [
            expected_counts.groupby("KT_full")[True].sum(),  # type: ignore
            persons.groupby("KT_full")[target_variable].sum(),
        ],
        keys=["Expected", "SynPop"],
        axis=1,
        sort=False,
    ).pipe(pd.DataFrame)
    canton["delta_abs"] = canton["SynPop"] - canton["Expected"]
    canton["delta_pc"] = (((canton["SynPop"] / canton["Expected"]) - 1) * 100).round(1)

    # Absolute differences
    title = f"SynPop{year} vs. Expected: Absolute Diff of {target_variable}"
    scale_bound = _round_up(canton["delta_abs"].abs().max())
    _ = mapper.draw_cantons(
        canton, "delta_abs", vmin=-scale_bound, vmax=scale_bound, title=title
    )
    visualisations.save_figure(
        True, f"SynPop{year}_{target_variable}_by_Canton_absolute.png", path, False
    )

    # Relative differences
    title = f"SynPop{year} vs. Expected: Relative (%) Diff of {target_variable}"
    scale_bound = round(canton["delta_pc"].abs().max())
    _ = mapper.draw_cantons(
        canton, "delta_abs", vmin=-scale_bound, vmax=scale_bound, title=title
    )
    visualisations.save_figure(
        True, f"SynPop{year}_{target_variable}_by_Canton_relative.png", path, False
    )


def _round_up(val: float) -> int:
    return int(math.ceil(val / 1000.0)) * 1000
