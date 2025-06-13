"""Module with the most important functions to use in mobi-synpop."""
import logging
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import pandas as pd

from synpop.anonymization import blur_per_zone
from synpop.fitting import business_fitting
from synpop.fitting import fitting_analysis
from synpop.fitting import fitting_config
from synpop.fitting import marginal_fitting
from synpop.fitting import population_fitting
from synpop.fitting.fitting_config import FittingConfig
from synpop.preprocessing import SynPopPreprocessor
from synpop.synpop_tables import SynPop
from synpop.synpop_tables import SynPopTable

logger = logging.getLogger(__name__)


def parse_raw_synpop(
    year: int,
    synpop_folder: Union[Path, str],
    synpop_preprocessing_config: Optional[Union[Path, str]] = None,
    validate: bool = True,
) -> SynPop:
    """Parse a raw synpop folder (with .csv files) into a SynPop object."""
    synpop_folder = Path(synpop_folder)
    preprocessor = SynPopPreprocessor(
        synpop_folder, year=year, config_path=synpop_preprocessing_config
    )
    persons, households = preprocessor.preprocess_persons_and_households()
    businesses = preprocessor.preprocess_businesses()
    synpop = SynPop(year, persons, households, businesses)
    _ = synpop.validate() if validate else None
    return synpop


def anonymize(
    synpop: SynPop,
    blurring_target: str = "households",
    blurring_hh_min_size_threshold: int = 1,
) -> SynPop:
    """Anonymize a SynPop (single table, default is households)."""
    table: SynPopTable = getattr(synpop, blurring_target)
    df = table.data
    df_nofit = df.head(0)
    if "collective_hh_type" in df.columns:
        df_nofit = df.loc[
            df["collective_hh_type"].isin(
                ["retirement_home", "boarding_school", "other"]
            ),
            :,
        ]
        df = df.drop(index=df_nofit.index)

    anonymized_df = blur_per_zone(
        dataframe=df.reset_index(),
        id_var=table.index_name,
        hh_min_size_threshold=blurring_hh_min_size_threshold,
    ).set_index(table.index_name)

    anonymized_df_full = pd.concat([anonymized_df, df_nofit])

    anonymized_synpop = synpop.copy()
    anonymized_table: SynPopTable = getattr(anonymized_synpop, blurring_target)
    anonymized_table.update(anonymized_df_full)

    if blurring_target == "households":
        # Update household-based columns on persons table
        fixed_persons = anonymized_synpop.persons.data.drop(
            [c for c in anonymized_df.columns if c != "household_id"],
            axis=1,
            errors="ignore",
        ).join(anonymized_df_full, on="household_id")

        anonymized_synpop.persons.update(fixed_persons)

    return anonymized_synpop


def fit_population(
    synpop: SynPop,
    fitting_tables_path: Union[Path, str],
    fitting_configs: List[FittingConfig],
    fitting_output: Optional[Path] = None,
    write_stats: bool = False,
    entire_households: bool = False,
) -> SynPop:
    """Fit the population according to the fitting tables."""
    if not fitting_output:
        fitting_output = Path(fitting_tables_path).parent / "fitting_output"
    fitting_output = fitting_output / "pop_total"
    fitting_output.mkdir(exist_ok=True, parents=True)

    segments = []
    persons_fitted = synpop.persons.data.copy()
    if "person_id" not in persons_fitted.columns:
        persons_fitted = persons_fitted.reset_index()
    for config in fitting_configs:
        logger.info("Fitting population: %s", config.config)
        pop_segment_variables = config.config.fitting_segments
        segments.append(pop_segment_variables)
        assert pop_segment_variables is not None
        control_totals = config.expected_counts.iloc[:, 0]
        if config.config.fit_as_delta:
            original_counts = persons_fitted.groupby(
                pop_segment_variables, observed=False
            ).size()
            logging.info(
                "Fitting population as delta. Original values: %d. Delta: %d. Target: %d.",
                original_counts.sum(),
                control_totals.sum(),
                original_counts.sum() + control_totals.sum(),
            )
            control_totals = control_totals.add(
                original_counts.loc[original_counts.index.isin(control_totals.index)],
                fill_value=0,
            ).astype(int)
        persons_fitted, _ = population_fitting.fit_population(
            persons=persons_fitted,
            pop_segment_variables=pop_segment_variables,
            control_totals=control_totals,
            entire_households=entire_households,
            clone_pool=None,
        )

    # households need to be adjusted (removed/added)
    households_fitted = synpop.households.data.copy()
    if "household_id" not in households_fitted.columns:
        households_fitted = households_fitted.reset_index()
    households_fitted = households_fitted.query(
        "household_id in @persons_fitted.household_id"
    )
    new_hhs = persons_fitted.query(
        "household_id not in @synpop.persons.data.household_id"
    )
    households_fitted = pd.concat(
        [
            households_fitted,
            new_hhs[["household_id", "zone_id", "xcoord", "ycoord"]].drop_duplicates(),
        ]
    )
    if write_stats:
        for fitting_segments in segments:
            population_fitting.write_pop_fitting_stats(
                output_folder=fitting_output,
                persons=synpop.persons.data,
                persons_fitted=persons_fitted,
                households=synpop.households.data,
                households_fitted=households_fitted,
                fitting_segments=fitting_segments,
            )

    return SynPop(
        synpop.year,
        persons_fitted.set_index("person_id"),
        households_fitted.set_index("household_id"),
        synpop.businesses.data,
    )


def fit_businesses(
    synpop: SynPop,
    fitting_tables_path: Union[Path, str],
    fitting_configs: List[FittingConfig],
    fitting_output: Optional[Path] = None,
    write_stats: bool = False,
) -> SynPop:
    """Fit the businesses according to the fitting tables."""
    if not fitting_output:
        fitting_output = Path(fitting_tables_path).parent / "fitting_output"
    fitting_output = fitting_output / "jobs"
    fitting_output.mkdir(exist_ok=True, parents=True)

    businesses_fitted = synpop.businesses.data.copy()
    if "business_id" not in businesses_fitted.columns:
        businesses_fitted = businesses_fitted.reset_index()
    for config in fitting_configs:
        logger.info("Fitting businesses: %s", config.config)
        segment_variables = config.config.fitting_segments
        jobs_var = config.config.target_variable
        assert jobs_var.startswith("jobs")
        lower_jobs_limit = 1
        if jobs_var == "jobs_exo":
            lower_jobs_limit = 0
        assert segment_variables is not None
        expected_counts = config.expected_counts.iloc[:, 0]
        if config.config.fit_as_delta:
            original_counts = businesses_fitted.groupby(
                segment_variables, observed=False
            )[jobs_var].sum()
            logging.info(
                "Fitting %s as delta. Original values: %d. Delta: %d. Target: %d.",
                jobs_var,
                original_counts.sum(),
                expected_counts.sum(),
                original_counts.sum() + expected_counts.sum(),
            )
            expected_counts = expected_counts.add(
                original_counts.loc[original_counts.index.isin(expected_counts.index)],
                fill_value=0,
            ).astype(int)
        businesses_fitted = business_fitting.fit_businesses(
            businesses=businesses_fitted,
            segment_variables=segment_variables,
            expected_counts=expected_counts,
            jobs_var=jobs_var,
            lower_limit=lower_jobs_limit,
        )

    if write_stats:
        logging.warning(
            "Automatic fitting statistics for businesses not yet implemented."
        )

    return SynPop(
        synpop.year,
        synpop.persons.data,
        synpop.households.data,
        businesses_fitted.set_index("business_id"),
    )


def fit_marginals(
    synpop: SynPop,
    synpop_ref: Optional[SynPop],
    fitting_tables_path: Union[Path, str],
    fitting_output: Optional[Union[Path, str]] = None,
    write_stats: bool = False,
) -> SynPop:
    """Fit SynPop attributes according to given marginals table."""
    configs = fitting_config.parse_fitting_goals(fitting_tables_path)

    if not fitting_output:
        fitting_output = Path(fitting_tables_path).parent / "fitting_output"

    persons_fixed = synpop.persons.data.copy()
    if "person_id" not in persons_fixed:
        persons_fixed = persons_fixed.reset_index()
    for target_variable, configs_var in configs.items():
        if target_variable == "pop_total" or target_variable.startswith("jobs_"):
            continue
        for i, config in enumerate(configs_var):
            logger.info("Fitting population: %s", config.config)
            fitting_segments = config.config.fitting_segments
            assert fitting_segments is not None
            expected_counts = config.expected_counts
            if config.config.fit_as_delta:
                if len(expected_counts) > 1:
                    raise ValueError(
                        "Delta-based marginal fitting only supported with single target."
                    )
                original_counts = (
                    persons_fixed.query(
                        f"{target_variable} == {expected_counts.columns[0]}"
                    )
                    .groupby(fitting_segments, observed=False)
                    .size()
                )
                logging.info(
                    "Fitting %s as delta. Original values: %d. Delta: %d. Target: %d.",
                    target_variable,
                    original_counts.sum(),
                    expected_counts.sum(),
                    original_counts.sum() + expected_counts.sum(),
                )
                expected_counts = expected_counts.add(
                    original_counts.loc[
                        original_counts.index.isin(expected_counts.index)
                    ],
                    fill_value=0,
                ).astype(int)
            persons_fixed = marginal_fitting.fit_target_variable(
                persons_fixed,
                synpop_ref.persons.data if synpop_ref is not None else None,
                config.config,
                expected_counts,
            )
            if (i == len(configs_var) - 1) and write_stats:
                fitting_analysis.generate_fitting_analysis(
                    synpop.persons.data,
                    persons_fixed,
                    synpop_ref.persons.data,
                    synpop.year,
                    synpop_ref.year,
                    target_variable,
                    fitting_output,
                    fitting_segments,
                    config.expected_counts,
                )
    return SynPop(
        synpop.year,
        persons_fixed.set_index("person_id"),
        synpop.households.data,
        synpop.businesses.data,
    )
