"""
This module contains the framework to modify agents and match population wide control totals.
"""
from typing import List
import logging
import warnings
import math

import pandas as pd

# fix the random seed for the marginal fitting process
import numpy
from synpop.config import RandomFittingConfig
numpy.random.seed(RandomFittingConfig().random_seed)

# Logger settings set in __init__.py
logger = logging.getLogger(__name__)


# Building marginal tables ##
def compute_comparison_summary(df_persons, df_persons_ist, year, year_ist, groupby):
    """
    Computes a comparison table of the two populations aggregated by the feature or features given.
    """
    counts = df_persons.groupby(groupby).count().iloc[:, 0].rename('counts_{}'.format(str(year)))
    counts_ist = df_persons_ist.groupby(groupby).count().iloc[:, 0].rename('counts_{}'.format(str(year_ist)))

    growth = (counts / counts_ist).round(2).rename('growth')

    prop = (counts / df_persons.shape[0]).round(2).rename('prop_{}'.format(str(year)))
    prop_ist = (counts_ist / df_persons_ist.shape[0]).round(2).rename('prop_{}'.format(str(year_ist)))

    return pd.concat([counts_ist, counts, growth, prop_ist, prop], axis=1).fillna(0).round(2)


def compute_ist_vs_scenario_marginal_counts(df, df_ist, year, year_ist, feature, control_level_list):
    """
    The scenario and ist population are aggregated by the feature and the control variables.
    --> The rows are counted in each aggregation group
    """
    # Copies to make sure original data is not touched
    df = df.copy(deep=True)
    df_ist = df_ist.copy(deep=True)

    group_by = control_level_list + [feature, ]

    # Changing boolean to string to avoid boolean as index and columns
    if df[feature].dtype == bool:
        df[feature] = df[feature].replace({False: 'false', True: 'true'})
        df_ist[feature] = df_ist[feature].replace({False: 'false', True: 'true'})

    # If feature is not categorical, make it. This will keep the empty categories when grouping by
    if df[feature].dtype.name != 'category':
        df[feature] = df[feature].astype('category')
        df_ist[feature] = df_ist[feature].astype('category')

    counts_ist = (df_ist.groupby(group_by).count().iloc[:, 0]
                  .fillna(0).astype(int)
                  .rename('counts_{}'.format(year_ist))
                  )

    counts = (df.groupby(group_by).count().iloc[:, 0]
              .fillna(0).astype(int)
              .rename('counts_{}'.format(year))
              )

    marginals = pd.concat((counts_ist, counts), axis=1, sort=True).fillna(0).astype(int)
    return marginals


def compute_ist_vs_scenario_marginal_summary_table(df, df_ist, year, year_ist, feature, control_level_list):
    """
    The scenario and ist population are aggregated by the feature and the control variables.
    --> The rows are counted in each aggregation group
    --> The ratio of counts per total are computed for the IST year
    --> The ratio are applied to the scenario to compute the expected counts
    --> The deltas are the differences between expected and actual counts
    """
    # Copies to make sure original data is not touched
    df = df.copy(deep=True)
    df_ist = df_ist.copy(deep=True)

    # Computing counts
    counts_df = compute_ist_vs_scenario_marginal_counts(df, df_ist, year, year_ist, feature, control_level_list)
    counts_ist = counts_df['counts_{}'.format(year_ist)]
    counts = counts_df['counts_{}'.format(year)]

    group_by = control_level_list + [feature, ]

    # Computing ratio in the IST year
    counts_per_pop_segment_ist = (counts_ist
                                  .reset_index()
                                  .groupby(control_level_list)['counts_{}'.format(year_ist)].sum()
                                  )

    ratios_ist = (counts_ist / counts_per_pop_segment_ist).rename('ratios_{}'.format(year_ist))

    # Computing the expected counts in the scenario year
    counts_per_pop_segment = df.groupby(control_level_list)['person_id'].count()
    expected_counts = ((ratios_ist * counts_per_pop_segment)
                       .fillna(0)
                       .astype(int)
                       .rename('expected_counts_{}'.format(year))
                       )
    # When there are zero counts in a population segment in IST, there can be no valid predictions.
    # In that case, the expected counts are set to the actual counts to avoid correcting based on this artifact.
    segments_with_no_ist_data = counts_per_pop_segment_ist[counts_per_pop_segment_ist == 0].index
    if len(segments_with_no_ist_data) > 0:
        expected_counts.loc[segments_with_no_ist_data] = counts.loc[segments_with_no_ist_data]
        logger.info('The following segments have no data in ist and will stay untouched: {}'
                       .format(segments_with_no_ist_data)
                       )

    deltas = (counts - expected_counts).rename('deltas_{}'.format(year))

    marginals = (pd.concat((counts_ist, ratios_ist, counts, expected_counts, deltas), axis=1, sort=True)
                 .fillna(0)
                 .astype({'counts_{}'.format(year): int, 'counts_{}'.format(year_ist): int,
                          'expected_counts_{}'.format(year): int, 'deltas_{}'.format(year): int
                          })
                 )

    return marginals


def compute_bfs_vs_scenario_marginal_summary_table(df, bfs_ratio_prediction, year, feature, control_level_list):
    """
    The scenario and ist population are aggregated by the feature and the control variables.
    --> The rows are counted in each aggregation group
    --> The ratio of counts per total are computed for the IST year
    --> The ratio are applied to the scenario to compute the expected counts
    --> The deltas are the differences between expected and actual counts
    """
    # Copies to make sure original data is not touched
    df = df.copy(deep=True)

    if type(control_level_list) is not list:
        control_level_list = [control_level_list]

    group_by = control_level_list + [feature, ]

    # Change types to category to get empty  options in the groupby as well
    original_types = df[group_by].dtypes.to_dict()  # keep this to set original types back after processing
    df[group_by] = df[group_by].astype('category')
    counts = df.groupby(group_by).count().iloc[:, 0].rename('counts_{}'.format(year)).fillna(0).astype(int)

    bfs_ratio_prediction = (bfs_ratio_prediction
                            .reindex(counts.index)
                            .fillna(0)
                            .rename('expected_ratios_{}'.format(year))
                            )

    counts_per_category = counts.reset_index().groupby(control_level_list).sum()['counts_{}'.format(year)]
    expected_counts = ((bfs_ratio_prediction * counts_per_category)
                       .fillna(0)
                       .astype(int)
                       .rename('expected_counts_{}'.format(year))
                       )

    deltas = (counts - expected_counts).rename('deltas_{}'.format(year))

    marginals = (pd.concat((bfs_ratio_prediction, counts, expected_counts, deltas), axis=1, sort=True).fillna(0))
    marginals = marginals.astype({'counts_{}'.format(year): int,
                                  'expected_counts_{}'.format(year): int,
                                  'deltas_{}'.format(year): int})
    # Setting index types to original
    marginals = marginals.reset_index().astype(original_types).set_index(group_by)
    return marginals


def compute_cross_table(df, main_feature, secondary_features):
    # Copies to make sure original data is not touched
    df = df.copy(deep=True)

    if type(secondary_features) is not list:
        secondary_features = [secondary_features]

    counts = (df.groupby(secondary_features + [main_feature]).count().iloc[:, 0]
              .fillna(0).astype(int)
              .rename('counts')
              .reset_index()
              )

    cross_table = counts.pivot_table(index=secondary_features, columns=main_feature, values='counts', fill_value=0)

    # In case the index is boolean, it is converted to string
    str_indices = []
    for i in range(cross_table.index.nlevels):
        str_indices.append(cross_table.index.get_level_values(i).astype(str))

    cross_table.index = str_indices

    return cross_table


# Fixing marginals ##
def fix_categorical_feature(persons: pd.DataFrame, feature, pop_segment_variables, control_totals: pd.DataFrame,
                            person_proba: pd.DataFrame = None):
    """
    The idea is not to pick the persons to change compleatly at random but based on a probabilistic model that gives
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
    if type(pop_segment_variables) is not list:
        pop_segment_variables = [pop_segment_variables]

    # If no probabilities are given, a uniform distribution is used.
    if person_proba is None:
        logger.info('Since no probability model has been given, a uniform distribution will be used.')
        person_proba = build_person_proba_with_uniform_distribution(persons, persons[feature].unique())

    # Change boolean to text to avoid weird issues. If it is already text, nothing will happen.
    persons[feature] = persons[feature].replace({False: 'false', True: 'true'})
    control_totals = control_totals.rename(columns={False: 'false', True: 'true'})
    person_proba = person_proba.rename(columns={False: 'false', True: 'true'})

    # Some sanity checks
    assert person_proba.shape[0] == persons.shape[0], 'Must have a probabilities for all persons'
    # Count persons in all population segments
    raw_pop_counts = (persons
                      .groupby(pop_segment_variables + [feature]).count()
                      .iloc[:, 0]
                      .fillna(0).astype(int)
                      .rename('counts')
                      .reset_index()
                      .pivot_table(index=pop_segment_variables, columns=feature, values='counts', fill_value=0)
                      )

    # raw + delta = control total
    delta_counts = (control_totals - raw_pop_counts).fillna(0).astype(int)

    feature_fixed = persons.set_index('person_id')[feature].copy(deep=True)

    logger.info('Fixing "{}" by population segments based on: {}'.format(feature, pop_segment_variables))
    tot_pop_segments = persons[pop_segment_variables].drop_duplicates().shape[0]

    # Group by the population is important to efficiently segment the population with n variables
    i = 0
    for segment_name, persons_segment in persons.groupby(pop_segment_variables):
        i += 1

        deltas = delta_counts.loc[segment_name]
        logger.debug('Sampling changes for segment = {} ({})'.format(segment_name, deltas.to_dict()))

        sampled_changes = _pick_the_persons_to_change(persons_segment.set_index('person_id')[feature],
                                                      person_proba,
                                                      deltas
                                                      )

        for cat, ids in sampled_changes.items():
            feature_fixed.loc[ids] = cat

        if (i % 20) == 0:
            logger.info('{} / {} population segments fixed...'.format(i, tot_pop_segments))

    logger.info('All {} population segments fixed!'.format(i))

    return feature_fixed


def _pick_the_persons_to_change(persons_raw, person_probabilities, deltas):
    """
    # Observation: this method takes close to 1 second and is the bottleneck in terms of computation time

    To optimise computational time, the agents are not sampled and modified one by one.
    This function sample a pool of agents from all categories with too many people.
    Each person in this pool of people is then assigned one of the categories with too few people.
    The modifications are done in one go afterwards.
    """
    # raw + delta = control total
    cats_with_too_many_people = deltas[deltas < 0].index.values
    logger.debug(f'The categories with too many people are: {cats_with_too_many_people}')

    # Check if there are any people to change
    total_changes = abs(deltas.loc[cats_with_too_many_people].sum())
    if total_changes == 0:
        logger.debug(f'The total number of peoples to change is zero. Returning ... ')
        return dict()  # no changes

    cats_with_too_few_people = deltas[deltas > 0].index.values
    logger.debug(f'The categories with too few people are: {cats_with_too_few_people}')

    # The correct number of people are pre-sampled from each category with too many people.
    # This forms the pool of persons that can be modified.
    person_subpools = []
    sampled_changes = dict()
    for origin_cat in cats_with_too_many_people:
        with warnings.catch_warnings():
            # This will hide one confusing FutureWarning message that should not be sent to the user
            # ref: https://stackoverflow.com/a/46721064
            warnings.simplefilter(action='ignore', category=FutureWarning)
            persons_in_origin_cat = persons_raw[persons_raw == origin_cat]

        logger.debug(f'Cat. "{origin_cat}" contains {len(persons_in_origin_cat)} persons')
        logger.debug(f'{-deltas[origin_cat]} persons from "{origin_cat}" will be sampled out')

        # The probability of being chose for change from A is the probability of not being A (1 - P(A))
        weights = 1 - person_probabilities.loc[persons_in_origin_cat.index, origin_cat]
        persons_to_change = persons_in_origin_cat.sample(n=-deltas[origin_cat],
                                                         replace=False,
                                                         weights=weights
                                                         )

        person_subpools.append(persons_to_change)
        logger.debug(f'{len(persons_to_change)} persons with original cat = "{origin_cat}" have been sampled')

    person_pool = pd.concat(person_subpools, axis=0)
    logger.debug(f'Person-pool has been filled up with {len(person_pool)} persons')

    for dest_cat in cats_with_too_few_people:
        # Pick randomly the persons from the pool weighted with their probability of being in dest
        persons_to_sample = min(deltas[dest_cat], person_pool.shape[0])  # to avoid rounding induced bugs
        diff = abs(persons_to_sample - deltas[dest_cat])
        assert diff < 10, 'This difference should always be very small! (diff = {})'.format(diff)

        logger.debug('Picking {} persons for "{}"'.format(persons_to_sample, dest_cat))

        # There is bug with sample happens when sampling a big proportion of values with weights and without replacement
        # Solution: if sampling more than 50%, the persons to be left out are sampled instead
        if persons_to_sample <= (person_pool.shape[0] * 0.5):
            logger.debug(f'Sampling directly {persons_to_sample} to change')
            sampled_persons = (person_pool
                               .sample(n=persons_to_sample,
                                       replace=False,
                                       weights=person_probabilities.loc[person_pool.index, dest_cat]
                                       )
                               .index
                               )
        else:
            persons_not_to_sample = person_pool.shape[0] - persons_to_sample
            logger.debug(f'Sampling indirectly {persons_not_to_sample} people not to change (to avoid a numpy bug...)')
            persons_not_sampled = (person_pool
                                   .sample(n=persons_not_to_sample,
                                           replace=False,
                                           weights=1 - person_probabilities.loc[person_pool.index, dest_cat]
                                           )
                                   .index
                                   )

            sampled_persons = person_pool.index[~person_pool.index.isin(persons_not_sampled)]

        sampled_changes[dest_cat] = sampled_persons
        logger.debug(f'{len(sampled_persons)} people sampled to be converted to "{dest_cat}"')
        logger.debug(f'Total number of people sampled for change: {len(sampled_changes[dest_cat])}')

        # Remove the sampled person from the person pool (redefining is much faster than dropping rows)
        person_pool = person_pool[~person_pool.index.isin(sampled_persons)]
        logger.debug(f'Person-pool now contains {len(person_pool)} persons')
    final_report = {f'to "{v}"': len(k) for v, k in sampled_changes.items()}
    logger.debug(f'All sampling done: {final_report}')
    return sampled_changes


def build_person_proba_with_uniform_distribution(persons, categories):
    person_proba = pd.DataFrame(1, index=persons['person_id'], columns=categories).fillna(1)
    person_proba = person_proba / len(categories)
    person_proba.index.name = 'person_id'
    assert (person_proba.sum(axis=1) - 1.0 < 1e-8).all()
    return person_proba


def build_person_proba_with_from_cross_table(persons, cross_table, merge_on):
    # Copies to make sure original data is not touched
    persons = persons.copy(deep=True)

    # If only a string is given, cast it into a list
    if type(merge_on) is not list:
        merge_on = [merge_on]

    # In to facilitate joining, convert all to string
    persons[merge_on] = persons[merge_on].astype(str)

    # All probabilities can be zero for one line if there was no counts for that category in the cross table.
    # This should very rarely happen, but when it does, a uniform probability is given to all possibilities
    uniform_proba = 1 / cross_table.shape[1]

    probabilities = cross_table.div(cross_table.sum(axis=1), axis=0).fillna(uniform_proba)

    person_proba = (pd.merge(persons[['person_id'] + merge_on], probabilities, left_on=merge_on, right_index=True)
                    .drop(merge_on, axis=1)
                    .set_index('person_id')
                    )

    assert person_proba.sum(axis=1).apply(lambda v: math.isclose(v, 1)).all()

    # There are some issues with sampling when some probabilities are exactly zero. This makes it work:
    person_proba = person_proba.mask(person_proba == 0, 1e-10)
    person_proba = person_proba.mask(person_proba == 1, 1 - 1e-10)

    return person_proba
