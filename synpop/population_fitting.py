"""
This module contains the framework to modify population control totals by cloning or removing agents.
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


# Fixing population totals (clone-kill)
def fix_population_totals(persons: pd.DataFrame, pop_segment_variables: List[str], control_totals: pd.Series,
                          clone_pool: pd.DataFrame = None):
    """
    This method takes a boolean feature which tells whether an agent should keep existing or not and fits it.
    Segments with too many agents are set to False and segments with too few receive clones.
    Clones receive a unique ID.
    Optionally a pool of agents may be provided to be sampled from instead of cloning.
    This can be useful to re-use agents previously removed instead of creating new clones.
    Returns the fitted exists_feature (including possible new cloned agents).
    This method has similarities to binary marginal fitting.
    """
    # Copies to make sure original data is not touched
    persons = persons.copy(deep=True)

    # Initialize helper variable.
    exists_feature = '__exists__'
    persons[exists_feature] = 'true'

    # Setup clone_pool if available
    if clone_pool is not None:
        clone_pool_ids = clone_pool['person_id'].copy(deep=True)
        clone_pool[exists_feature] = 'true'
        assert clone_pool_ids.isin(persons['person_id']).sum() == 0, "Person in clone_pool found in SynPop!"
        assert set(clone_pool.columns) == set(persons.columns), "clone_pool is incompatible with input SynPop!"

    # If only a string is given, cast pop_segment_variables into a list
    if type(pop_segment_variables) is not list:
        pop_segment_variables = [pop_segment_variables]

    # Count persons in all population segments
    raw_pop_counts = (persons
                      .groupby(pop_segment_variables).count()
                      .iloc[:, 0]
                      )
    assert control_totals.index.isin(raw_pop_counts.index).all(), "Control totals not found in SynPop!"

    # raw + delta = control total
    delta_counts = (control_totals - raw_pop_counts.loc[control_totals.index]).fillna(0).astype(int)

    keep_person = persons.set_index('person_id')[exists_feature].copy(deep=True)
    cloned_persons = []

    logger.debug(f'Initial population size is {len(persons)}.')
    logger.info(f'Total persons to clone: {delta_counts.loc[delta_counts > 0].sum()}.')
    logger.info(f'Total persons to remove: {delta_counts.loc[delta_counts < 0].abs().sum()}.')
    tot_pop_segments = persons[pop_segment_variables].drop_duplicates().shape[0]

    # Group by the population is important to efficiently segment the population with n variables
    i = 0
    added = 0
    removed = 0
    for segment_name, persons_segment in persons.groupby(pop_segment_variables):
        # we fit only cases where we have control_totals, otherwise skip
        if segment_name in delta_counts.index:
            i += 1

            delta = delta_counts.loc[segment_name]
            logger.debug('Sampling changes for segment = {} ({})'.format(segment_name, delta))
            if delta > 0:
                # positive delta means missing agents.
                # first try to sample from clone_pool if it exists
                n_clones = delta
                sampled_pool = []
                if clone_pool is not None and not clone_pool.empty:
                    compatible_pool = clone_pool.loc[(clone_pool[pop_segment_variables] == segment_name).all(axis=1)]
                    n_pool = min(len(compatible_pool), delta)
                    n_clones = delta - n_pool
                    sampled_pool = compatible_pool['person_id'].sample(n=n_pool, replace=False).to_list()
                    cloned_persons += sampled_pool
                    added += len(sampled_pool)
                    clone_pool = clone_pool.loc[~clone_pool['person_id'].isin(sampled_pool)]
                # clone agents
                to_clone = persons_segment['person_id'].sample(n=n_clones, replace=True).to_list()
                cloned_persons += to_clone
                added += len(to_clone)
                msg = f'{len(to_clone) + len(sampled_pool)} to add at {pop_segment_variables}="{segment_name}". '
                if clone_pool is not None:
                    msg += f'Of which {len(sampled_pool)} came from given clone_pool.'
                logger.debug(msg)
            else:
                # negative delta means too many agents. Remove.
                persons_to_remove = persons_segment.sample(n=-delta, replace=False)
                keep_person.loc[persons_to_remove['person_id']] = 'false'
                removed += len(persons_to_remove)
                logger.debug(f'{len(persons_to_remove)} to remove at {pop_segment_variables}="{segment_name}".')

            if (i % max(1, round(tot_pop_segments / 20, -1))) == 0:  # round to the nearest 10, avoid 0 division
                logger.info('{} / {} population segments fixed...'.format(i, tot_pop_segments))

    # convert back to boolean
    keep_person = keep_person == 'true'

    logger.debug('{} persons selected for removal.'.format(len(keep_person) - keep_person.sum()))
    logger.debug('{} persons were created through cloning.'.format(len(cloned_persons)))
    logger.debug(f'Final population size is {keep_person.sum() + len(cloned_persons)}.')
    logger.info('All {} population segments fixed!'.format(i))

    return keep_person, cloned_persons


# wrapper to fix_population_totals() for convenience and with some safety checks
def fit_population(persons: pd.DataFrame, pop_segment_variables: List[str], control_totals: pd.Series,
                   clone_pool: pd.DataFrame = None):
    exists_feature = '__exists__'  # helper
    belongs, cloned_persons = fix_population_totals(persons=persons,
                                                    pop_segment_variables=pop_segment_variables,
                                                    control_totals=control_totals,
                                                    clone_pool=clone_pool
                                                    )
    assert len(belongs) == len(persons)
    not_in_control_total = (len(persons) -
                            len(pd.merge(persons[pop_segment_variables], control_totals.reset_index(), how='inner')))
    assert (belongs.sum() + len(cloned_persons)) == control_totals.sum() + not_in_control_total, "Totals must match!"

    # removals
    persons_fitted = persons.copy(deep=True)
    persons_fitted = pd.merge(persons_fitted, belongs, left_on='person_id', right_index=True)

    # check if any new person comes from clone_pool
    clone_pool_person_ids = []
    if clone_pool is not None and not clone_pool.empty:
        clone_pool[exists_feature] = False
        clone_pool_persons = clone_pool.loc[clone_pool['person_id'].isin(cloned_persons)].copy(deep=True)
        clone_pool_persons[exists_feature] = True
        persons_fitted = pd.concat([persons_fitted, clone_pool_persons], ignore_index=True)
        clone_pool_person_ids = clone_pool_persons['person_id'].to_list()
        cloned_persons = [p for p in cloned_persons if p not in clone_pool_person_ids]

    # clones
    persons_to_clone = persons.set_index('person_id').loc[cloned_persons].reset_index().copy(deep=True)
    persons_to_clone['person_id'] = persons_to_clone.reset_index().index + persons['person_id'].max() + 1
    persons_to_clone[exists_feature] = True
    persons_fitted = pd.concat([persons_fitted, persons_to_clone], ignore_index=True)

    assert len(persons_fitted.query(exists_feature)) == control_totals.sum() + not_in_control_total, \
        "Results don't match control totals!"
    clones_match = (persons_fitted['person_id'].isin(persons['person_id']).sum() == len(persons) and
                    (~persons_fitted['person_id'].isin(persons['person_id'])).sum() == (len(cloned_persons) +
                                                                                        len(clone_pool_person_ids)))
    assert clones_match, "Number of clones don't match!"

    return (persons_fitted.loc[persons_fitted[exists_feature]].drop(exists_feature, axis=1),
            persons_fitted.loc[~persons_fitted[exists_feature]].drop(exists_feature, axis=1))


# iteratively fits the population to reach multiple targets.
# 'marginals' is a list of 2-tuples as ('pop_segment_variables', 'control_totals'), as in previous methods.
def iterative_pop_fit(persons: pd.DataFrame, marginals: dict, max_error=0.01, clone_pool: pd.DataFrame = None):
    error = numpy.inf
    persons = persons.copy(deep=True)
    if clone_pool is None:
        clone_pool = pd.DataFrame(columns=persons.columns)

    logger.info(f'Fitting population totals iteratively based on {[k for k, _ in marginals]}.')
    i = 1
    while error > max_error:
        logger.info(f'Iteration {i}')
        # fit population iteratively, IPF-like
        for pop_segment_variables, control_totals in marginals:
            logger.info(f'Fitting {pop_segment_variables}')
            clone_pool = clone_pool.loc[~clone_pool['person_id'].isin(persons['person_id'])]
            persons, new_removed = fit_population(persons, pop_segment_variables, control_totals, clone_pool)
            clone_pool = clone_pool.append(new_removed).drop_duplicates(ignore_index=True)
        # calc error
        it_max_error = 0.0
        for pop_segment_variables, control_totals in marginals[:-1]:  # skip last since error will be 0.0
            errors = (persons.groupby(pop_segment_variables).count().iloc[:, 0] - control_totals)
            if max_error < 1.0:  # relative error measure
                errors = errors / control_totals
            error = errors.abs().replace(numpy.inf, numpy.nan).dropna().max()
            logger.info(f'Maximum error of {pop_segment_variables}: {error}')
            if error > it_max_error:
                it_max_error = error
        logger.info(f'Finished iteration {i}. Error: {it_max_error}')
        error = it_max_error
        i += 1

    logger.info(f'Finished. Resulting population has {len(persons)} persons')
    return persons

