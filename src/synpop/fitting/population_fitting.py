"""Framework to modify population control totals by cloning/removing agents."""
import logging
import random
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import numpy
import pandas as pd

# fix the random seed for the marginal fitting process
numpy.random.seed(2020)
random.seed(2020)

# Logger settings set in __init__.py
logger = logging.getLogger(__name__)

SAMPLE_PERSONS = pd.DataFrame(
    {
        "position_in_bus": ["null", "employee", "employee"],
        "highest_education": ["primary", "secondary", "secondary"],
        "position_in_edu": ["null", "null", "null"],
        "nation": ["swiss", "swiss", "swiss"],
        "language": ["other", "other", "other"],
        "level_of_employment": [0, 85, 100],
        "year_of_birth": [2021, 1985, 1985],
        "age": [1, 37, 37],
        "is_swiss": [True, True, True],
        "analysis_subpopulation": ["regular", "regular", "regular"],
        "is_employed": [False, True, True],
        "loe_group": ["non-employed", "part-time", "full-time"],
        "current_edu": ["null", "null", "null"],
        "current_job_rank": ["null", "employee", "employee"],
        "is_apprentice": [False, False, False],
        "collective_hh_type": ["null", "null", "null"],
        "age_group": ["0-4", "35-39", "35-39"],
    }
)


# Fixing population totals (clone-kill)
def fix_population_totals(  # pylint: disable=too-many-locals,too-many-statements
    persons: pd.DataFrame,
    pop_segment_variables: List[str],
    control_totals: pd.Series,
    clone_pool: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, List[str]]:
    """Fit population totals by cloning or removing agents.

    This method takes a boolean feature which tells whether an agent
    should keep existing or not and fits it. Segments with too many
    agents are set to False and segments with too few receive clones.
    Clones receive a unique ID. Optionally a pool of agents may be
    provided to be sampled from instead of cloning. This can be useful
    to re-use agents previously removed instead of creating new clones.
    Returns the fitted exists_feature (including possible new cloned
    agents). This method has similarities to binary marginal fitting.
    """
    # Copies to make sure original data is not touched
    persons = persons.copy(deep=True)

    # Initialize helper variable.
    exists_feature = "__exists__"
    persons[exists_feature] = "true"

    # Setup clone_pool if available
    if clone_pool is not None:
        clone_pool_ids = clone_pool["person_id"].copy(deep=True)
        clone_pool[exists_feature] = "true"
        assert (
            clone_pool_ids.isin(persons["person_id"]).sum() == 0
        ), "Person in clone_pool found in SynPop!"
        assert set(clone_pool.columns) == set(
            persons.columns
        ), "clone_pool is incompatible with input SynPop!"

    # If only a string is given, cast pop_segment_variables into a list
    if not isinstance(pop_segment_variables, list):
        pop_segment_variables = [pop_segment_variables]

    # Count persons in all population segments
    raw_pop_counts = persons.groupby(pop_segment_variables).size()
    assert control_totals.index.isin(
        raw_pop_counts.index
    ).all(), "Control totals not found in SynPop!"

    # raw + delta = control total
    delta_counts = (
        (control_totals - raw_pop_counts.loc[control_totals.index])
        .fillna(0)
        .astype(int)
    )

    keep_person = persons.set_index("person_id")[exists_feature].copy(deep=True)
    cloned_persons = []

    logger.debug("Initial population size is %d.", len(persons))
    logger.info(
        "Total persons to clone: %d.", delta_counts.loc[delta_counts.gt(0)].sum()
    )
    logger.info(
        "Total persons to remove: %d.", delta_counts.loc[delta_counts.lt(0)].abs().sum()
    )
    tot_pop_segments = persons[pop_segment_variables].drop_duplicates().shape[0]

    # Group by the population is important to efficiently segment the population with n variables
    i = 0
    for segment_name, persons_segment in persons.groupby(pop_segment_variables):
        if isinstance(segment_name, tuple) and len(segment_name) == 1:
            segment_name = segment_name[0]

        # we fit only cases where we have control_totals, otherwise skip
        if segment_name in delta_counts.index:
            i += 1

            delta = delta_counts.loc[segment_name]
            logger.debug("Sampling changes for segment = %s (%d)", segment_name, delta)
            if delta > 0:
                # positive delta means missing agents.
                # first try to sample from clone_pool if it exists
                n_clones = delta
                sampled_pool = []
                if clone_pool is not None and not clone_pool.empty:
                    compatible_pool = clone_pool.loc[
                        (clone_pool[pop_segment_variables] == segment_name).all(axis=1)
                    ]
                    n_pool = min(len(compatible_pool), delta)
                    n_clones = delta - n_pool
                    sampled_pool = (
                        compatible_pool["person_id"]
                        .sample(n=n_pool, replace=False)
                        .to_list()
                    )
                    cloned_persons += sampled_pool
                    clone_pool = clone_pool.loc[
                        ~clone_pool["person_id"].isin(sampled_pool)
                    ]
                # clone agents
                to_clone = (
                    persons_segment["person_id"]
                    .sample(n=n_clones, replace=True)
                    .to_list()
                )
                cloned_persons += to_clone
                msg = f'{len(to_clone) + len(sampled_pool)} of {delta} \
                    to add at {pop_segment_variables}="{segment_name}". '
                if clone_pool is not None:
                    msg += f"Of which {len(sampled_pool)} came from given clone_pool."
                logger.debug(msg)
            else:
                # negative delta means too many agents. Remove.
                if len(persons_segment) < -delta:
                    logger.warning(
                        "Not enough persons in segment %s, %i persons, delta=%i",
                        segment_name,
                        len(persons_segment),
                        -delta,
                    )
                    logger.warning("Removing all persons instead.")
                    delta = -len(persons_segment)

                persons_to_remove = persons_segment.sample(n=-delta, replace=False)
                keep_person.loc[persons_to_remove["person_id"]] = "false"
                logger.debug(
                    '%d of %i to remove at %s="%s".',
                    len(persons_to_remove),
                    delta,
                    pop_segment_variables,
                    segment_name,
                )

            if (
                i % max(1, round(tot_pop_segments / 20, -1))
            ) == 0:  # round to the nearest 10, avoid 0 division
                logger.info("%d / %d population segments fixed...", i, tot_pop_segments)

    # convert back to boolean
    keep_person = keep_person == "true"

    logger.debug(
        "%d persons selected for removal.", len(keep_person) - keep_person.sum()
    )
    logger.debug("%d persons were created through cloning.", len(cloned_persons))
    logger.debug(
        "Final population size is %d.", (keep_person.sum() + len(cloned_persons))
    )
    logger.info("All %d population segments fixed!", i)

    return keep_person, cloned_persons


def get_sample_persons(persons, missing_ct):
    from synpop.visualization.zone_maps import (  # pylint: disable=import-outside-toplevel
        get_zone_centroids,
    )

    sample_persons = pd.concat(
        [SAMPLE_PERSONS.copy() for _ in range(len(missing_ct))],
        keys=range(len(missing_ct)),
        names=["key", "index"],
    ).reset_index("index", drop=True)
    sample_persons["household_id"] = (
        persons["household_id"].max() + 1 + sample_persons.index
    )
    sample_persons["person_id"] = (
        sample_persons.reset_index().index
        + persons.reset_index()["person_id"].max()
        + 1
    )
    zone_centroids = (
        get_zone_centroids()
        .query(f"zone_id in {missing_ct.get_level_values('zone_id').to_list()}")
        .reset_index(drop=True)
    )
    sample_persons = sample_persons.join(zone_centroids.drop("centroid", axis=1))
    cols = [c for c in sample_persons.columns if c in persons.columns]
    return sample_persons[cols]


def fit_population(  # pylint: disable=too-many-locals,too-many-statements
    persons: pd.DataFrame,
    pop_segment_variables: List[str],
    control_totals: pd.Series,
    entire_households=True,
    clone_pool: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Wrap fix_population_totals for convenience and additional safety checks.

    It adds three features:
    1) clone_pool (pick agents from pool before cloning new agents).
    2) adapt results to ensure full households are cloned/removed
    3) convenience: returns DF of final population and DF of those removed
    """
    raw_pop_counts = persons.groupby(pop_segment_variables).size()
    control_totals_missing = ~control_totals.index.isin(raw_pop_counts.index)
    if control_totals_missing.any():
        missing_ct = control_totals.loc[control_totals_missing].index
        logger.warning(
            "For the following target %s no persons were found: %s",
            pop_segment_variables,
            missing_ct.to_list(),
        )
        if "zone_id" in control_totals.index.names:
            logger.warning(
                "These zones will be fitted with a sample household at its centroid."
            )
            sample_persons = get_sample_persons(persons, missing_ct)
            persons = pd.concat([persons, sample_persons])
            raw_pop_counts = persons.groupby(pop_segment_variables).size()
        else:
            logger.warning("These zones will be ignored.")
            control_totals = control_totals.loc[~control_totals_missing]

    exists_feature = "__exists__"  # helper
    belongs, cloned_persons = fix_population_totals(
        persons=persons,
        pop_segment_variables=pop_segment_variables,
        control_totals=control_totals,
        clone_pool=clone_pool,
    )

    not_in_control_total = len(persons) - len(
        pd.merge(
            persons[pop_segment_variables], control_totals.reset_index(), how="inner"
        )
    )
    assert (
        belongs.sum() + len(cloned_persons)
    ) == control_totals.sum() + not_in_control_total, f"Unable to fit to control totals, {belongs.sum() + len(cloned_persons)} != {control_totals.sum() + not_in_control_total}"

    # adjust the sampled persons by picking everyone living in same HH and dropping the excess
    if entire_households:
        if clone_pool is not None:
            raise NotImplementedError(
                "Household consistency and clone_pool not yet compatible!"
            )
        delta_counts = (
            (control_totals - raw_pop_counts.loc[control_totals.index])
            .fillna(0)
            .astype(int)
        )
        # adjust cloned persons
        if len(cloned_persons) > 0:
            clones_hh_ids = (  # pylint: disable=unused-variable
                persons.set_index("person_id")
                .loc[cloned_persons, "household_id"]  # type: ignore
                .drop_duplicates()
            )
            clones_with_hh_members = persons.query(
                "household_id in @clones_hh_ids"
            ).sort_values("household_id")
            cloned_persons = (
                clones_with_hh_members.groupby(
                    pop_segment_variables, as_index=False
                ).apply(
                    lambda g: rolling_head(g, delta_counts.loc[g.name])  # type: ignore
                )[
                    "person_id"
                ]
            ).tolist()
        # adjust removed persons
        if (~belongs).sum() > 0:
            removed_hh_ids = (  # pylint: disable=unused-variable
                persons.set_index("person_id")
                .loc[~belongs, "household_id"]
                .drop_duplicates()
            )
            removed_with_hh_members = persons.query(
                "household_id in @removed_hh_ids"
            ).sort_values("household_id")
            new_removed_persons = removed_with_hh_members.groupby(
                pop_segment_variables, as_index=False
            ).apply(
                lambda g: g.head(max(0, -delta_counts.loc[g.name]))  # type: ignore
            )[
                "person_id"
            ]
            belongs = ~(
                persons.set_index("person_id")
                .index.to_series()
                .isin(new_removed_persons)
            )  # person_id-indexed
            belongs.name = exists_feature

        final_diff_after_hh = (belongs.sum() + len(cloned_persons)) - (
            control_totals.sum() + not_in_control_total
        )
        if final_diff_after_hh != 0:
            logger.warning(
                "Due to 'entire_households=True', a perfect fit is not always possible. Final diff: %i (%.4f%%)",
                final_diff_after_hh,
                (final_diff_after_hh / control_totals.sum()) * 100,
            )

    assert len(belongs) == len(persons)
    assert belongs.index.isin(persons["person_id"]).all()
    not_in_control_total = len(persons) - len(
        pd.merge(
            persons[pop_segment_variables], control_totals.reset_index(), how="inner"
        )
    )

    # removals
    persons_fitted = persons.copy(deep=True)
    persons_fitted = pd.merge(
        persons_fitted, belongs, left_on="person_id", right_index=True
    )

    # check if any new person comes from clone_pool
    clone_pool_person_ids = []
    if clone_pool is not None and not clone_pool.empty:
        clone_pool[exists_feature] = False
        clone_pool_persons = clone_pool.loc[
            clone_pool["person_id"].isin(cloned_persons)
        ].copy(deep=True)
        clone_pool_persons[exists_feature] = True
        persons_fitted = pd.concat(
            [persons_fitted, clone_pool_persons], ignore_index=True
        )
        clone_pool_person_ids = clone_pool_persons["person_id"].to_list()
        cloned_persons = [p for p in cloned_persons if p not in clone_pool_person_ids]

    # clones
    persons_to_clone = (
        persons.set_index("person_id").loc[cloned_persons].reset_index().copy(deep=True)
    )
    persons_to_clone["person_id"] = (
        persons_to_clone.reset_index().index + persons["person_id"].max() + 1
    )
    persons_to_clone[exists_feature] = True
    if entire_households:
        persons_to_clone = assign_new_hh_ids(persons_to_clone, persons)
        assert (
            not persons_to_clone["household_id"]
            .isin(persons_fitted["household_id"])
            .any()
        )

    persons_fitted = pd.concat([persons_fitted, persons_to_clone], ignore_index=True)
    assert persons_fitted["person_id"].is_unique

    clones_match = persons_fitted["person_id"].isin(persons["person_id"]).sum() == len(
        persons
    ) and (~persons_fitted["person_id"].isin(persons["person_id"])).sum() == (
        len(cloned_persons) + len(clone_pool_person_ids)
    )
    assert clones_match, "Number of clones don't match!"

    return (
        persons_fitted.loc[persons_fitted[exists_feature]].drop(exists_feature, axis=1),
        persons_fitted.loc[~persons_fitted[exists_feature]].drop(
            exists_feature, axis=1
        ),
    )


def assign_new_hh_ids(new_persons, persons):
    new_hh_ids = (
        new_persons.reset_index(drop=True)["household_id"]
        .drop_duplicates()
        .reset_index()
        .set_index("household_id")
        .iloc[:, 0]
        .add(persons["household_id"].max() + 1)
        .to_dict()
    )
    new_persons["household_id"] = new_persons["household_id"].apply(new_hh_ids.get)
    assert not new_persons["household_id"].isna().any()
    return new_persons


def rolling_head(df, n):
    # When collecting full households, it is possible,
    # due to picking each household only once (drop_duplicates), that too little clones are available.
    # Here they get cloned again.
    length = len(df)
    if n <= length:
        return df.head(n)
    repeats = n // length
    remainder = n % length
    rolled_df = pd.concat([df] * repeats + [df.head(remainder)], ignore_index=True)
    return rolled_df


def iterative_pop_fit(
    persons: pd.DataFrame,
    marginals: dict,
    max_error=0.01,
    clone_pool: Optional[pd.DataFrame] = None,
):
    """Fits the population iteratively to reach multiple targets.

    'marginals' is a list of 2-tuples as ('pop_segment_variables',
    'control_totals'), as in previous methods.
    """
    error = numpy.inf
    persons = persons.copy(deep=True)
    if clone_pool is None:
        clone_pool_ = pd.DataFrame(columns=persons.columns)
    else:
        clone_pool_ = clone_pool

    logger.info(
        "Fitting population totals iteratively based on %s.", [k for k, _ in marginals]
    )
    i = 1
    while error > max_error:
        logger.info("Iteration %d", i)
        # fit population iteratively, IPF-like
        for pop_segment_variables, control_totals in marginals:
            logger.info("Fitting %s", pop_segment_variables)
            clone_pool_ = clone_pool_.loc[
                ~clone_pool_["person_id"].isin(persons["person_id"])
            ]
            persons, new_removed = fit_population(
                persons, pop_segment_variables, control_totals, clone_pool=clone_pool_
            )
            clone_pool_ = pd.concat([clone_pool_, new_removed]).drop_duplicates(  # type: ignore
                ignore_index=True
            )
        # calc error
        it_max_error = 0.0
        for pop_segment_variables, control_totals in marginals[
            :-1
        ]:  # skip last since error will be 0.0
            errors = persons.groupby(pop_segment_variables).size() - control_totals
            if max_error < 1.0:  # relative error measure
                errors = errors / control_totals
            error = errors.abs().replace(numpy.inf, numpy.nan).dropna().max()
            logger.info("Maximum error of %s: %d.", pop_segment_variables, error)
            if error > it_max_error:
                it_max_error = error
        logger.info("Finished iteration %d. Error: %d.", i, it_max_error)
        error = it_max_error
        i += 1

    logger.info("Finished. Resulting population has %d persons.", len(persons))
    return persons


def write_pop_fitting_stats(
    output_folder: Path,
    persons: pd.DataFrame,
    persons_fitted: pd.DataFrame,
    households: pd.DataFrame,
    households_fitted: pd.DataFrame,
    fitting_segments: List[str],
) -> None:
    """Write statistics about the population fitting to a CSV file."""
    for dataframe_fitted, dataframe_orig in [
        (persons_fitted, persons),
        (households_fitted, households),
    ]:
        stats = pd.concat(
            [
                dataframe_orig[fitting_segments].value_counts(),
                dataframe_fitted[fitting_segments].value_counts(),
            ],
            axis=1,
            keys=["original", "fitted"],
        )
        stats["delta"] = stats["fitted"] - stats["original"]
        stats["delta (%)"] = (stats["delta"] / stats["original"]).mul(100).round(1)
        stats.query("delta != 0").to_csv(
            output_folder
            / f'{type(dataframe_orig).__name__}-stats-{"&".join(fitting_segments)}.csv'
        )
