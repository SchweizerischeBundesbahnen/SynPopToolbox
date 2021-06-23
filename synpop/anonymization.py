"""
TODO: fix comment
Preprocessing:
    From raw "persons.csv", "households.csv", "businesses.csv", "locations.csv" to cleaned and compressed DataFrames.

All conventions are set in config/synpop_loading_config.json. This file can be modified to suite other pre-processing
 needs.
"""
import logging

import numpy
import pandas as pd

from synpop.config import SynPopConfig, SwissCantons

# Logger settings set in __init__.py
logger = logging.getLogger(__name__)


# Blurring households
class EmptyPoolException(Exception):
    pass


def blur_households_per_zone(households: pd.DataFrame, hh_min_size_threshold: int = 1) -> pd.DataFrame:
    """
    For privacy reasons, the coordinates of all households within a zone are shuffled randomly.
    No household is allowed to keep its origin coordinates unless there is no other way around.
    Small buildings may be additionally anonymized by allowing them to receive more households (hh_min_size_threshold).
    """
    original_households = households.copy(deep=True)

    blurred_households = (original_households.set_index('household_id')[['location_id', 'xcoord', 'ycoord']]
                          .copy(deep=True)
                          )

    # Shuffling all zones takes about 10-15 minutes!
    nbr_of_zones = blurred_households['location_id'].nunique()
    logger.info('Shuffling coordinates for all {} households in {} zones.'
                .format(blurred_households.shape[0], nbr_of_zones))

    shuffled_coordinates = dict()
    for i, (loc_id, households) in enumerate(blurred_households.groupby('location_id')):
        households = households[['xcoord', 'ycoord']]
        new_coordinates = _shuffle_coordinates_within_zone(
            households, loc_id, hh_min_size_threshold=hh_min_size_threshold)
        shuffled_coordinates.update(new_coordinates)

        if ((i + 1) % 100) == 0:
            logger.info('Household coordinates shuffled in {} / {} zones.'.format(i + 1, nbr_of_zones))

    ids = list(shuffled_coordinates.keys())
    xcoord = [shuffled_coordinates[i][0] for i in ids]
    ycoord = [shuffled_coordinates[i][1] for i in ids]

    shuffled_coords = pd.DataFrame({'xcoord': xcoord, 'ycoord': ycoord}, index=ids)

    blurred_households = pd.merge(blurred_households, shuffled_coords, how='left', right_index=True, left_index=True,
                                  suffixes=('_old', ''))
    hh_with_original_coord = blurred_households[(blurred_households['xcoord_old'] == blurred_households['xcoord']) &
                                                (blurred_households['ycoord_old'] == blurred_households['ycoord'])
                                                ]
    failed_zones = hh_with_original_coord['location_id'].unique()
    failed_households = hh_with_original_coord.index.values

    logger.info('Coordinates of {} households in {} zones have been blurred.'.
                format(blurred_households.shape[0], nbr_of_zones))
    if len(failed_zones) > 0:
        logger.warning('No solution found for {} zones where some households kept their original coordinates. '
                       'For example: {}'.format(len(failed_zones), failed_zones[:5]))
        logger.warning('In total, {} households kept their original coordinates.'
                       'For example: {}'.format(len(failed_households), failed_households[:5]))

    # Sanity check
    if hh_min_size_threshold == 1:
        hh_per_coord_old = (blurred_households
                            .groupby(['location_id', 'xcoord_old', 'ycoord_old']).count().iloc[:, 0]
                            .rename('hhs_old')
                            .reset_index()
                            .rename(columns={'xcoord_old': 'xcoord', 'ycoord_old': 'ycoord'})
                            )
        hh_per_coord_new = (blurred_households.groupby(['location_id', 'xcoord', 'ycoord']).count().iloc[:, 0].
                            rename('hhs_new')
                            .reset_index()
                            )
        hh_per_coord = pd.merge(hh_per_coord_old, hh_per_coord_new, how='left', on=['location_id', 'xcoord', 'ycoord'])
        assert (hh_per_coord['hhs_old'] == hh_per_coord['hhs_new']).all(),  'The number of hh per coord must match exactly!'
    assert blurred_households.shape[0] == original_households.shape[0], 'All households should have blurred coords!'

    # Joining the blurred coordinates onto the original household dataframe
    original_households = original_households.drop(['xcoord', 'ycoord'], axis=1)
    original_households = pd.merge(original_households, blurred_households[['xcoord', 'ycoord']], how='left',
                                   right_index=True, left_on='household_id')
    return original_households


def _shuffle_coordinates_within_zone(households: pd.DataFrame, location_id: int,
                                     max_attempts: int = 20, hh_min_size_threshold: int = 1) -> dict:
    """
    Every household must change coordinates, we do not allowed their original coordinates to be picked again.
    Sometimes, multiple households have the same coordinates (i.e. a building).
    Sometimes, because of random chance, the last household has not new coordinates to pick in the pool. When this
    happens, the shuffling is attempted again. After 20 attempts, shuffling correctly is given up for this zone. It
    may be impossible to change all coordinates (for example a zone with only one building).
    When it is impossible to change all coordinates, the rule is softened and the last households to be picked will
    keep their coordinates if the pool of available coordinates is empty.
    """
    attempt = 0
    while True:
        attempt += 1
        # It may happen that the random choices lead to an impossible solution (no more valid coordinates in pool).
        # Before declaring the zone impossible to blur, the shuffling is attempted up to "max_attempts" times.
        try:
            new_coord = _attempt_shuffling_coordinates_once(households, hh_min_size_threshold=hh_min_size_threshold)
            logger.debug('All {} household coordinates shuffled successfully in zone {} after {} attempt(s)'
                         .format(households.shape[0], location_id, attempt))
            break

        except EmptyPoolException:
            logger.debug('{} shuffling attempt failed with zone {} (max. attempts = {}) '
                         .format(attempt, location_id, max_attempts))

            if attempt >= max_attempts:
                new_coord = _attempt_shuffling_coordinates_once(households, allow_exceptions=True,
                                                                hh_min_size_threshold=hh_min_size_threshold)
                logger.debug(
                    'Max. attempts ({}) reached with zone {}! Some households will keep their original coordinates.'
                    .format(max_attempts, location_id))
                break

    return new_coord


def _attempt_shuffling_coordinates_once(households: pd.DataFrame, allow_exceptions: bool = False,
                                        hh_min_size_threshold: int = 1) -> dict:
    coordinate_pool = [tuple(coords) for coords in households.values]

    # Small buildings may be additionally anonymized by allowing them to receive more households
    if hh_min_size_threshold > 1:
        # Get number of HH each coordinate might additionally support
        building_sizes = households.reset_index().groupby(['xcoord', 'ycoord']).count().iloc[:, 0]
        extra_hh_slots = hh_min_size_threshold - building_sizes
        # Extra coordinates, or 'slots', are generated for all buildings below the threshold size
        for coords, extra_slots in extra_hh_slots.loc[extra_hh_slots > 0].iteritems():
            coordinate_pool.extend(
                [coords for _ in range(extra_slots)])

    new_coord = dict()
    for hh_id, hh_coord in households.iterrows():
        old_coord = tuple(hh_coord.values)

        # Every household must get new coordinates! We remove old_coord from the pool.
        valid_coordinate_pool = [coords for coords in coordinate_pool if coords != old_coord]

        if len(valid_coordinate_pool) > 0:
            new_coordinate_id = numpy.random.choice(len(valid_coordinate_pool))
            picked_coord = valid_coordinate_pool[new_coordinate_id]

            assert picked_coord != old_coord, 'The picked coordinates are the same as the old ones!'

            new_coord[hh_id] = picked_coord

            # Defining a new list is more than 10x faster than removing an element in list with .drop() or .pop()!
            id_to_drop = coordinate_pool.index(picked_coord)  # https://stackoverflow.com/a/9542768
            coordinate_pool = [coords for i, coords in enumerate(coordinate_pool) if i != id_to_drop]

        else:
            # In some cases it is hard to shuffle households while changing ALL ids.
            if allow_exceptions:
                # Exceptions are allowed when the max attempts has been reached with no success.
                # In this case, no correct solution is possible so we allow some households to keep their coordinates.
                logger.debug('It was impossible to find new coordinates for household {}!'.format(hh_id))
                new_coord[hh_id] = old_coord

            else:
                # If the valid_coordinate_pool is empty then we must try again or give up.
                raise EmptyPoolException('"valid_coordinate_pool" is empty for household {}'.format(hh_id))

    assert len(new_coord) == households.shape[0]

    return new_coord
