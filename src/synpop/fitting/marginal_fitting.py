# pylint: disable=too-many-lines
"""This module contains the framework to modify agents and match population wide control totals."""
import logging
import math
import random
import warnings
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy
import pandas as pd

from synpop.fitting.fitting_config import VariableFittingConfig
from synpop.validation import get_categories

# fix the random seed for the marginal fitting process
numpy.random.seed(2020)
random.seed(2020)


# Logger settings set in __init__.py
logger = logging.getLogger(__name__)


def compute_ist_vs_scenario_marginal_counts(
    dataframe: pd.DataFrame,
    dataframe_ist: pd.DataFrame,
    year: int,
    year_ist: int,
    feature: str,
    control_level_list: List[str],
) -> pd.DataFrame:
    """Compute the marginal counts for a given feature and control variables.

    The scenario and ist population are aggregated by the feature and the control variables.
    --> The rows are counted in each aggregation group
    """
    # Copies to make sure original data is not touched
    dataframe = dataframe.copy(deep=True)
    dataframe_ist = dataframe_ist.copy(deep=True)

    group_by = control_level_list + [
        feature,
    ]

    # Changing boolean to string to avoid boolean as index and columns
    if dataframe[feature].dtype == bool:
        dataframe[feature] = dataframe[feature].replace({False: "false", True: "true"})
        dataframe_ist[feature] = dataframe_ist[feature].replace(
            {False: "false", True: "true"}
        )

    # If feature is not categorical, make it. This will keep the empty categories when grouping by
    if dataframe[feature].dtype.name != "category":
        dataframe[feature] = dataframe[feature].astype("category")
        dataframe_ist[feature] = dataframe_ist[feature].astype("category")

    counts_ist = (
        dataframe_ist.groupby(group_by)
        .count()
        .iloc[:, 0]
        .fillna(0)
        .astype(int)
        .rename(f"counts_{year_ist}")
    )

    counts = (
        dataframe.groupby(group_by)
        .count()
        .iloc[:, 0]
        .fillna(0)
        .astype(int)
        .rename(f"counts_{year}")
    )

    marginals = pd.concat((counts_ist, counts), axis=1, sort=True).fillna(0).astype(int)
    assert isinstance(marginals, pd.DataFrame)
    return marginals


def compute_ist_vs_scenario_marginal_summary_table(  # pylint: disable=too-many-locals
    dataframe: pd.DataFrame,
    dataframe_ist: pd.DataFrame,
    year: int,
    year_ist: int,
    feature: str,
    control_level_list: List[str],
) -> pd.DataFrame:
    """Aggregate the scenario and ist population by the feature and the control variables.

    --> The rows are counted in each aggregation group
    --> The ratio of counts per total are computed for the IST year
    --> The ratio are applied to the scenario to compute the expected counts
    --> The deltas are the differences between expected and actual counts
    """
    # Copies to make sure original data is not touched
    dataframe = dataframe.copy(deep=True)
    dataframe_ist = dataframe_ist.copy(deep=True)

    # Computing counts
    counts_df = compute_ist_vs_scenario_marginal_counts(
        dataframe, dataframe_ist, year, year_ist, feature, control_level_list
    )
    counts_ist = counts_df[f"counts_{year_ist}"]
    counts = counts_df[f"counts_{year}"]

    # Computing ratio in the IST year
    counts_per_pop_segment_ist = (
        counts_ist.reset_index().groupby(control_level_list)[f"counts_{year_ist}"].sum()
    )

    ratios_ist = (counts_ist / counts_per_pop_segment_ist).rename(f"ratios_{year_ist}")

    # Computing the expected counts in the scenario year
    counts_per_pop_segment = dataframe.groupby(control_level_list)["person_id"].count()
    expected_counts = (
        (ratios_ist * counts_per_pop_segment)
        .fillna(0)
        .astype(int)
        .rename(f"expected_counts_{year}")
    )
    # When there are zero counts in a population segment in IST, there can be no valid predictions.
    # In that case, the expected counts are set to the actual counts
    # to avoid correcting based on this artifact.
    segments_with_no_ist_data = counts_per_pop_segment_ist[
        counts_per_pop_segment_ist == 0
    ].index
    if len(segments_with_no_ist_data) > 0:
        expected_counts.reset_index(feature).loc[  # type: ignore
            segments_with_no_ist_data
        ] = counts.reset_index(feature).loc[segments_with_no_ist_data]
        logger.info(
            "The following segments have no data in ist and will stay untouched: %s",
            segments_with_no_ist_data,
        )

    deltas = (counts - expected_counts).rename(f"deltas_{year}")

    marginals = (
        pd.concat(
            (counts_ist, ratios_ist, counts, expected_counts, deltas), axis=1, sort=True
        )
        .fillna(0)
        .astype(
            {
                f"counts_{year}": int,
                f"counts_{year_ist}": int,
                f"expected_counts_{year}": int,
                f"deltas_{year}": int,
            }
        )
    )

    assert isinstance(marginals, pd.DataFrame)
    return marginals


def compute_bfs_vs_scenario_marginal_summary_table(
    dataframe: pd.DataFrame,
    bfs_ratio_prediction: pd.DataFrame,
    year: int,
    feature: str,
    control_level_list: Union[str, List[str]],
) -> pd.DataFrame:
    """Aggregate the scenario and ist population by the feature and the control variables.

    --> The rows are counted in each aggregation group
    --> The ratio of counts per total are computed for the IST year
    --> The ratio are applied to the scenario to compute the expected counts
    --> The deltas are the differences between expected and actual counts
    """
    # Copies to make sure original data is not touched
    dataframe = dataframe.copy(deep=True)

    if not isinstance(control_level_list, list):
        control_level_list = [control_level_list]  # type: ignore

    group_by = control_level_list + [
        feature,
    ]

    # Change types to category to get empty  options in the groupby as well
    original_types = dataframe[
        group_by
    ].dtypes.to_dict()  # keep this to set original types back after processing
    dataframe[group_by] = dataframe[group_by].astype("category")
    counts = (
        dataframe.groupby(group_by)
        .count()
        .iloc[:, 0]
        .rename(f"counts_{year}")
        .fillna(0)
        .astype(int)
    )

    bfs_ratio_prediction = (
        bfs_ratio_prediction.reindex(counts.index)
        .fillna(0)
        .rename(f"expected_ratios_{year}")  # type: ignore
    )

    counts_per_category = (
        counts.reset_index().groupby(control_level_list).sum()[f"counts_{year}"]
    )
    expected_counts = (
        (bfs_ratio_prediction * counts_per_category)
        .fillna(0)
        .astype(int)
        .rename(f"expected_counts_{year}")  # type: ignore
    )

    deltas = (counts - expected_counts).rename(f"deltas_{year}")

    marginals = pd.concat(
        (bfs_ratio_prediction, counts, expected_counts, deltas), axis=1, sort=True
    ).fillna(0)
    marginals = marginals.astype(
        {f"counts_{year}": int, f"expected_counts_{year}": int, f"deltas_{year}": int}
    )
    # Setting index types to original
    marginals = marginals.reset_index().astype(original_types).set_index(group_by)
    return marginals


def compute_cross_table(
    dataframe: pd.DataFrame,
    main_feature: Union[str, List[str]],
    secondary_features: Union[str, List[str]],
) -> pd.DataFrame:
    """Compute the cross table of a main feature and a list of secondary features."""
    # Copies to make sure original data is not touched
    dataframe = dataframe.copy(deep=True)

    if not isinstance(secondary_features, list):
        secondary_features = [secondary_features]  # type: ignore

    counts = (
        dataframe.groupby(secondary_features + [main_feature])
        .count()
        .iloc[:, 0]
        .fillna(0)
        .astype(int)
        .rename("counts")
        .reset_index()
    )

    cross_table = counts.pivot_table(
        index=secondary_features, columns=main_feature, values="counts", fill_value=0
    )

    # In case the index is boolean, it is converted to string
    cross_table = (
        cross_table.reset_index()
        .astype({c: str for c in secondary_features})
        .set_index(secondary_features)
    )

    return cross_table


def rebalance_counts_by_age(
    counts: pd.Series,
    variable: str,
    marginal_name: str,
    target_marginal_counts: pd.Series,
    query: Optional[str] = None,
    remove_from: str = "null",
) -> pd.Series:
    """Rebalance counts by age."""
    mask_target = counts.index.get_level_values(variable) == marginal_name
    mask_removal = counts.index.get_level_values(variable) == remove_from
    if query:
        filter_ = counts.index.isin(counts.to_frame().query(query).index)
        mask_target = mask_target & filter_
        mask_removal = mask_removal & filter_
        target_marginal_counts = (
            target_marginal_counts.to_frame().query(query).iloc[:, 0]
        )

    counts_current = counts.loc[mask_target].droplevel(variable)
    counts_removal = counts.loc[mask_removal].droplevel(variable)
    delta_counts = target_marginal_counts - counts_current  # size of shift
    target_counts_removal = counts_removal - delta_counts
    if not (target_counts_removal >= 0).all():
        logging.warning("Some counts turned negative, please adjust manually.")

    new_counts = counts.copy()
    new_counts = new_counts.mask(mask_target, target_marginal_counts)
    new_counts = new_counts.mask(mask_removal, target_counts_removal)
    return new_counts


def comparison_table_categories(
    persons: pd.DataFrame,
    persons_ist: pd.DataFrame,
    year: int,
    year_ist: int,
    col: str,
    groupby: str,
) -> pd.DataFrame:
    """Compute a comparison table for a categorical variable."""
    col_counts = persons.pivot_table(
        index=groupby, columns=col, aggfunc="size", fill_value=0
    )
    col_counts.columns = col_counts.columns.astype(str)
    col_counts_ist = persons_ist.pivot_table(
        index=groupby, columns=col, aggfunc="size", fill_value=0
    )
    col_counts_ist.columns = col_counts_ist.columns.astype(str)
    abs_growth = col_counts - col_counts_ist
    pc_growth = (
        (abs_growth / col_counts_ist)
        .mul(100)
        .round(1)
        .rename(columns=lambda c: f"RelGrowth {c}")
    )
    abs_growth = abs_growth.rename(columns=lambda c: f"AbsGrowth {c}")
    counts = pd.merge(
        col_counts,
        col_counts_ist,
        left_index=True,
        right_index=True,
        suffixes=(f" {year}", f" {year_ist}"),
    )
    table = pd.concat([counts, abs_growth, pc_growth], axis=1, sort=False)
    columns = [
        [f"{c} {year}", f"{c} {year_ist}", f"AbsGrowth {c}", f"RelGrowth {c}"]
        for c in col_counts.columns
    ]
    columns = [c for sl in columns for c in sl]
    return table[columns]


# Fixing marginals ##
def fix_categorical_feature(  # pylint: disable=too-many-locals
    persons: pd.DataFrame,
    feature: str,
    pop_segment_variables: Union[str, List[str]],
    control_totals: pd.DataFrame,
    person_proba: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Fix a categorical feature according to given goals and probabilities.

    The idea is not to pick the persons to change compleatly at random
    but based on a probabilistic model that gives
    and estimation of the likelihood that person is in the wrong category.
    This functionality is implemented for categorical features.
    persons.

    :param persons:
    :param feature:
    :param pop_segment_variables:
    :param control_totals:
    :param person_proba:
    """
    # Copies to make sure original data is not touched
    persons = persons.copy(deep=True)

    # If only a string is given, cast pop_segment_variables into a list
    if not isinstance(pop_segment_variables, list):
        pop_segment_variables = [pop_segment_variables]

    # If no probabilities are given, a uniform distribution is used.
    if person_proba is None:
        logger.info(
            "Since no probability model has been given, a uniform distribution will be used."
        )
        person_proba = build_person_proba_with_uniform_distribution(
            persons, persons[feature].unique().tolist()
        )
    assert person_proba is not None
    # Change boolean to text to avoid weird issues. If it is already text, nothing will happen.
    persons[feature] = persons[feature].replace({False: "false", True: "true"})
    control_totals = control_totals.rename(columns={False: "false", True: "true"})
    person_proba = person_proba.rename(columns={False: "false", True: "true"})
    assert person_proba is not None

    # Some sanity checks
    assert (
        person_proba.shape[0] == persons.shape[0]
    ), "Must have a probabilities for all persons"
    # Count persons in all population segments
    raw_pop_counts = (
        persons.groupby(pop_segment_variables + [feature])
        .size()
        .fillna(0)
        .astype(int)
        .rename("counts")
        .reset_index()
        .pivot_table(
            index=pop_segment_variables, columns=feature, values="counts", fill_value=0
        )
    )

    control_totals_missing = ~control_totals.index.isin(raw_pop_counts.index)
    if control_totals_missing.any():
        logger.warning(
            "For the following target '%s' no persons were found: %s",
            feature,
            control_totals.loc[control_totals_missing].index.to_list(),
        )
        logger.warning("These zones will be ignored.")
        control_totals = control_totals.loc[~control_totals_missing]

    # raw + delta = control total
    raw_pop_counts = raw_pop_counts.loc[
        raw_pop_counts.index.isin(control_totals.index)
    ]  # fit required segments only

    # if only one category is given, and a 'null' category is available,
    # use this as exchange basis (dynamic fitting)
    categories = get_categories("persons")
    if len(control_totals.columns) == 1 and "null" in categories[feature]:
        logger.info(
            'Fitting binary exchange between %s and "null".', control_totals.columns[0]
        )
        # fomula is: new_null = totals - (new_cat + sum(other_cats))
        totals = raw_pop_counts.sum(axis=1)
        control_totals["null"] = totals - (
            control_totals.iloc[:, 0]
            + raw_pop_counts.drop([control_totals.columns[0], "null"], axis=1).sum(1)
        )
    # ensure all categories are in control_totals (fill missing with current_counts)
    if len(control_totals.columns) < len(categories[feature]):
        control_totals = pd.concat(
            [control_totals, raw_pop_counts.drop(control_totals.columns, axis=1)],
            axis=1,
        )

    assert (
        (control_totals >= 0).all().all()
    ), f"Negative marginals found at \n{control_totals.loc[(control_totals < 0).any(axis=1)]}"
    delta_counts = (control_totals - raw_pop_counts).fillna(0).astype(int)
    assert (
        delta_counts.sum(axis=1).abs() < 5
    ).all(), "Sum of marginals per pop segment must be consistent to SynPop!"

    feature_fixed = persons.set_index("person_id")[feature].copy(deep=True)

    logger.info(
        'Fixing "%s" by population segments based on: %s',
        feature,
        pop_segment_variables,
    )
    tot_pop_segments = len(delta_counts)

    # Group by the population is important to efficiently segment the population with n variables
    i = 0
    for segment_name, persons_segment in persons.groupby(pop_segment_variables):
        if isinstance(segment_name, tuple) and len(segment_name) == 1:
            segment_name = segment_name[0]

        if segment_name in delta_counts.index:
            i += 1
            deltas = delta_counts.loc[segment_name]
            logger.debug(
                "Sampling changes for segment = %s (%s)", segment_name, deltas.to_dict()
            )

            sampled_changes = _pick_the_persons_to_change(
                persons_segment.set_index("person_id")[feature], person_proba, deltas
            )

            for cat, ids in sampled_changes.items():
                feature_fixed.loc[ids] = cat

            if (i % 20) == 0:
                logger.info("%d / %d population segments fixed...", i, tot_pop_segments)

    logger.info("All %d population segments fixed!", i)

    # fix boolean attribute
    if feature_fixed.drop_duplicates().isin(["true", "false"]).all():
        feature_fixed = feature_fixed.replace({"false": False, "true": True})

    return feature_fixed


def _pick_the_persons_to_change(  # pylint: disable=too-many-locals
    persons_raw: pd.Series, person_probabilities: pd.DataFrame, deltas: pd.DataFrame
) -> Dict[str, pd.Index]:
    """Pick persons to change given deltas and probabilities.

    Observation: this method takes ~1s and is the bottleneck in terms of computation time.

    To optimise computational time, the agents are not sampled and modified one by one.
    This function sample a pool of agents from all categories with too many people.
    Each person in this pool of people is then assigned one of the categories with too few people.
    The modifications are done in one go afterwards.
    """
    # raw + delta = control total
    cats_with_too_many_people = deltas[deltas < 0].index.values
    logger.debug(
        "The categories with too many people are: %s", cats_with_too_many_people
    )

    # Check if there are any people to change
    total_changes = abs(deltas.loc[cats_with_too_many_people].sum())
    if total_changes == 0:
        logger.debug("The total number of peoples to change is zero. Returning ... ")
        return {}  # no changes

    cats_with_too_few_people = deltas[deltas > 0].index.values
    logger.debug("The categories with too few people are: %s", cats_with_too_few_people)

    # The correct number of people are pre-sampled from each category with too many people.
    # This forms the pool of persons that can be modified.
    person_subpools = []
    sampled_changes = {}
    for origin_cat in cats_with_too_many_people:
        with warnings.catch_warnings():
            # This will hide one confusing FutureWarning message that should not be sent to the user
            # ref: https://stackoverflow.com/a/46721064
            warnings.simplefilter(action="ignore", category=FutureWarning)
            persons_in_origin_cat = persons_raw[persons_raw == origin_cat]

        logger.debug(
            'Cat. "%s" contains %d persons', origin_cat, len(persons_in_origin_cat)
        )
        logger.debug(
            '%d persons from "%s" will be sampled out', -deltas[origin_cat], origin_cat
        )

        # The probability of being chose for change from A
        # is the probability of not being A (1 - P(A))
        weights = 1 - person_probabilities.loc[persons_in_origin_cat.index, origin_cat]
        assert isinstance(
            weights, pd.Series
        ), f"weights should be a series, not a {type(weights)}"
        persons_to_change = persons_in_origin_cat.sample(
            n=-deltas[origin_cat], replace=False, weights=weights.tolist()
        )

        person_subpools.append(persons_to_change)
        logger.debug(
            '%d persons with original cat = "%s" have been sampled',
            len(persons_to_change),
            origin_cat,
        )

    person_pool = pd.concat(person_subpools, axis=0)
    logger.debug("Person-pool has been filled up with %d persons.", len(person_pool))

    for dest_cat in cats_with_too_few_people:
        # Pick randomly the persons from the pool weighted with their probability of being in dest
        persons_to_sample = min(
            deltas[dest_cat], person_pool.shape[0]
        )  # to avoid rounding induced bugs
        diff = abs(persons_to_sample - deltas[dest_cat])
        assert (
            diff < 10
        ), f"This difference should always be very small! (diff = {diff})."

        logger.debug('Picking %d persons for "%s"', persons_to_sample, dest_cat)

        # There is bug with sample happens when sampling a big proportion of values
        # with weights and without replacement.
        # Solution: if sampling more than 50%, the persons to be left out are sampled instead
        weights = person_probabilities.loc[person_pool.index, dest_cat]
        assert isinstance(
            weights, pd.Series
        ), f"weights should be a series, not a {type(weights)}"
        if persons_to_sample <= (person_pool.shape[0] * 0.5):
            logger.debug("Sampling directly %d to change", persons_to_sample)
            sampled_persons = person_pool.sample(
                n=persons_to_sample,
                replace=False,
                weights=weights,
            ).index
        else:
            persons_not_to_sample = person_pool.shape[0] - persons_to_sample
            logger.debug(
                "Sampling indirectly %d people not to change (to avoid a numpy bug...)",
                persons_not_to_sample,
            )
            persons_not_sampled = person_pool.sample(
                n=persons_not_to_sample,
                replace=False,
                weights=1 - weights,
            ).index

            sampled_persons = person_pool.index[
                ~person_pool.index.isin(persons_not_sampled)
            ]

        sampled_changes[dest_cat] = sampled_persons
        logger.debug(
            '%d people sampled to be converted to "%s"', len(sampled_persons), dest_cat
        )
        logger.debug(
            "Total number of people sampled for change: %d",
            len(sampled_changes[dest_cat]),
        )

        # Remove the sampled person from the person pool
        # (redefining is much faster than dropping rows)
        person_pool = person_pool[~person_pool.index.isin(sampled_persons)]
        logger.debug("Person-pool now contains %d persons", len(person_pool))
    final_report = {f'to "{v}"': len(k) for v, k in sampled_changes.items()}
    logger.debug("All sampling done: %s", final_report)
    return sampled_changes


def build_person_proba_with_uniform_distribution(
    persons: pd.DataFrame, categories: List[str]
) -> pd.DataFrame:
    """Build a person-probability table with a uniform distribution across given categories."""
    person_proba = pd.DataFrame(
        1, index=persons["person_id"], columns=categories
    ).fillna(1)
    person_proba = person_proba / len(categories)
    person_proba.index.name = "person_id"
    assert (person_proba.sum(axis=1) - 1.0 < 1e-8).all()
    return person_proba


def build_person_proba_with_from_cross_table(
    persons: pd.DataFrame, cross_table: pd.DataFrame, merge_on: Union[str, List[str]]
) -> pd.DataFrame:
    """Build a person-probability table from a cross table."""
    # Copies to make sure original data is not touched
    persons = persons.copy(deep=True)

    # If only a string is given, cast it into a list
    if not isinstance(merge_on, list):
        merge_on = [merge_on]  # type: ignore

    # In to facilitate joining, convert all to string
    persons[merge_on] = persons[merge_on].astype(str)

    # All probabilities can be zero for one line if there was no counts for
    # that category in the cross table. This should very rarely happen,
    # but when it does, a uniform probability is given to all possibilities
    uniform_proba = 1 / cross_table.shape[1]

    probabilities = cross_table.div(cross_table.sum(axis=1), axis=0).fillna(
        uniform_proba
    )

    person_proba = (
        pd.merge(
            persons[["person_id"] + merge_on],
            probabilities,
            left_on=merge_on,
            right_index=True,
        )
        .drop(merge_on, axis=1)
        .set_index("person_id")
    )

    assert person_proba.sum(axis=1).apply(lambda v: math.isclose(v, 1)).all()

    # There are some issues with sampling when some probabilities are exactly zero.
    person_proba = person_proba.mask(person_proba == 0, 1e-10)
    person_proba = person_proba.mask(person_proba == 1, 1 - 1e-10)

    return person_proba


def fit_target_variable(
    persons: pd.DataFrame,
    persons_ist: Optional[pd.DataFrame],
    fitting_config: VariableFittingConfig,
    expected_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Fit the target variable of the persons table.

    Wrapper function which does all the fitting-related work and returns
    the full transformed persons dataframe.
    """
    persons_fixed = persons.copy(deep=True)
    target_variable = fitting_config.target_variable
    assert isinstance(target_variable, str)
    if target_variable in ("level_of_employment", "is_employed"):
        raise ValueError('Fitting employment only supported for "current_job_rank".')

    # prepare person probabilities if required
    if fitting_config.probability_weights is not None:
        if persons_ist is None:
            raise ValueError(
                "For weighted fits (probability_weights != None), "
                "'persons_ist' must be provided for the cross-table"
            )
        cross_table = compute_cross_table(
            persons_ist,
            main_feature=target_variable,
            secondary_features=fitting_config.probability_weights,
        )
        person_proba = build_person_proba_with_from_cross_table(
            persons_fixed, cross_table, merge_on=fitting_config.probability_weights
        )
    else:
        # uniform probability is used
        person_proba = None

    # fix variable with regular marginal fitting
    fitting_segments = fitting_config.fitting_segments
    assert fitting_segments is not None
    fixed_variable = fix_categorical_feature(
        persons=persons_fixed,
        feature=target_variable,
        pop_segment_variables=fitting_segments,
        control_totals=expected_counts,
        person_proba=person_proba,
    )

    # update persons dataframe and return
    persons_fixed = persons_fixed.drop(target_variable, axis=1)
    persons_fixed = pd.merge(
        persons_fixed, fixed_variable, left_on="person_id", right_index=True
    )

    # fix other employment variables
    if target_variable == "current_job_rank":
        is_employed_fixed = persons_fixed[target_variable] != "null"
        persons_fixed.loc[~is_employed_fixed, "level_of_employment"] = 0
        # if loe is unknown, simply sample from the global distribution
        mask = is_employed_fixed & (persons_fixed["level_of_employment"] == 0)
        is_employed_orig = persons[target_variable] != "null"
        sampled = persons.loc[is_employed_orig, "level_of_employment"].sample(
            mask.sum(), replace=True
        )
        persons_fixed.loc[mask, "level_of_employment"] = sampled.astype(
            "float64"
        ).tolist()
        assert persons_fixed.loc[~is_employed_fixed, "level_of_employment"].max() == 0
        assert persons_fixed.loc[is_employed_fixed, "level_of_employment"].min() > 0
        if "is_employed" in persons_fixed.columns:
            persons_fixed["is_employed"] = is_employed_fixed

    return persons_fixed
