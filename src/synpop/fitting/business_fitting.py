import logging
from typing import List
from typing import Union

import pandas as pd
from tqdm import tqdm

from synpop.visualization.zone_maps import get_zone_centroids

SAMPLE_BUSINESS = pd.DataFrame(
    {
        "jobs_endo": [2],
        "fte_endo": [1.4932936],
        "jobs_exo": [0],
        "fte_exo": [0.0],
        "school_type": ["no_school"],
        "noga_code": [960201],
        "sector": ["other services"],
    }
)


def get_sample_businesses(businesses, missing_ct):
    sample_businesses = pd.concat(
        [SAMPLE_BUSINESS.copy() for _ in range(len(missing_ct))]
    ).reset_index(drop=True)
    sample_businesses["business_id"] = (
        sample_businesses.index + businesses.reset_index()["business_id"].max() + 1
    )
    zone_centroids = (
        get_zone_centroids()
        .query(f"zone_id in {missing_ct.get_level_values('zone_id').to_list()}")
        .reset_index(drop=True)
    )
    sample_businesses = sample_businesses.join(zone_centroids.drop("centroid", axis=1))
    cols = [c for c in sample_businesses.columns if c in businesses.columns]
    return sample_businesses[cols]


def add_jobs_to_region(
    businesses_in_region: pd.DataFrame, delta: int, jobs_var: str = "jobs_endo"
) -> pd.DataFrame:
    """
    Add jobs to businesses in a region.

    Args:
        businesses_in_region (pd.DataFrame): DataFrame of businesses within a region.
        delta (int): Number of jobs to add.
        jobs_var (str): Column name representing the number of jobs.

    Returns:
        pd.DataFrame: Updated DataFrame with jobs added.
    """
    weights = businesses_in_region[jobs_var]
    jobs_to_add = (
        businesses_in_region.sample(
            delta, replace=True, weights=weights if weights.sum() > 0 else None
        )
        .reset_index()
        .groupby("business_id")
        .size()
        .rename(f"new_{jobs_var}")
    )

    businesses_in_region[jobs_var] = (
        businesses_in_region[jobs_var].add(jobs_to_add, fill_value=0).astype(int)
    )
    return businesses_in_region


def remove_jobs_from_region(
    businesses_in_region: pd.DataFrame,
    region: List[str],
    delta: int,
    jobs_var: str = "jobs_endo",
    lower_limit: int = 1,
) -> pd.DataFrame:
    """
    Remove jobs from businesses in a region while respecting a lower limit.

    Args:
        businesses_in_region (pd.DataFrame): DataFrame of businesses within a region.
        delta (int): Number of jobs to remove (as a negative number).
        jobs_var (str): Column name representing the number of jobs.
        lower_limit (int): Minimum number of jobs allowed for any business.
        region (list): Region name for logging purposes.

    Returns:
        pd.DataFrame: Updated DataFrame with jobs removed.
    """
    to_remove = -delta
    while to_remove > 0:
        target_idx = businesses_in_region.loc[
            businesses_in_region[jobs_var] > lower_limit
        ].index

        if (businesses_in_region.loc[target_idx, jobs_var] <= lower_limit).all():
            logging.warning(
                "Stopping removal of jobs from region "
                "while respecting the lower limit. Couldn't remove a total of %i jobs at region %s.",
                to_remove,
                region,
            )
            break

        weights = businesses_in_region.loc[target_idx, jobs_var]
        jobs_to_remove = (
            businesses_in_region.loc[target_idx]
            .sample(
                to_remove, replace=True, weights=weights if weights.sum() > 0 else None
            )
            .reset_index()
            .groupby("business_id")
            .size()
        )

        businesses_in_region.loc[
            jobs_to_remove.index, jobs_var
        ] = businesses_in_region.loc[jobs_to_remove.index, jobs_var].subtract(
            jobs_to_remove, fill_value=0
        )

        to_remove = -(
            businesses_in_region.loc[target_idx]
            .query(f"{jobs_var} < {lower_limit}")[jobs_var]
            .subtract(lower_limit)
            .sum()
        )

        businesses_in_region.loc[target_idx, jobs_var] = businesses_in_region.loc[
            target_idx, jobs_var
        ].clip(lower=lower_limit)
    return businesses_in_region


def scale_jobs_in_region(
    delta: int,
    businesses_in_region: pd.DataFrame,
    region: List[str],
    jobs_var: str = "jobs_endo",
    lower_limit: int = 1,
) -> pd.DataFrame:
    """
    Scale jobs in a specific region by adding or removing jobs.

    Args:
        delta (int): The number of jobs to add (positive) or remove (negative).
        businesses_in_region (pd.DataFrame): DataFrame of businesses within the region.
        jobs_var (str): Column name representing the number of jobs.
        lower_limit (int): Minimum number of jobs allowed for any business.
        region (list): Region name for logging purposes.

    Returns:
        pd.Series: Series representing the scaled jobs for each business.
    """
    if len(businesses_in_region) == 0:
        logging.warning(
            "Can't apply delta of %i jobs from region. There are no businesses!", delta
        )
        return businesses_in_region[jobs_var]

    if delta >= 0:
        businesses_in_region = add_jobs_to_region(businesses_in_region, delta, jobs_var)
    else:
        businesses_in_region = remove_jobs_from_region(
            businesses_in_region, region, delta, jobs_var, lower_limit
        )
    return businesses_in_region


def scale_jobs(
    deltas: pd.Series,
    businesses: pd.DataFrame,
    jobs_var: str = "jobs_endo",
    segment_variables: Union[str, List[str]] = "mun_id",
    lower_limit: int = 1,
) -> pd.Series:
    """
    Scale jobs across multiple regions by adjusting job numbers based on deltas.

    Args:
        deltas (pd.Series): Series mapping regions to job changes (positive for addition, negative for removal).
        businesses (pd.DataFrame): DataFrame of all businesses, including region and job information.
        jobs_var (str): Column name representing the number of jobs.
        segment_variables (str): Column name representing the region identifier.
        lower_limit (int): Minimum number of jobs allowed for any business.

    Returns:
        pd.Series: Series with the new job counts across all businesses.
    """
    if not isinstance(segment_variables, list):
        segment_variables = [segment_variables]

    assert (businesses[jobs_var] >= 0).all()
    businesses = businesses.set_index("business_id")
    grouped = businesses.groupby(segment_variables)

    dfs = []
    for segment_name, businesses_in_region in tqdm(grouped, total=len(grouped)):
        if len(deltas.index.names) == 1:
            segment_name = segment_name[0]
        if segment_name in deltas.index:
            delta = deltas.loc[segment_name]
            logging.debug("Sampling changes for segment = %s (%s)", segment_name, delta)
            businesses_in_region_scaled = scale_jobs_in_region(
                delta, businesses_in_region, segment_name, jobs_var, lower_limit
            )
            dfs.append(businesses_in_region_scaled.assign(scaled=True))
        else:
            dfs.append(businesses_in_region.assign(scaled=False))

    return pd.concat(dfs).reset_index()


def fit_businesses(  # pylint: disable=too-many-locals
    businesses,
    expected_counts,
    jobs_var,
    segment_variables,
    lower_limit=1,
):
    if businesses[jobs_var].isna().any():
        logging.warning(
            "There are %d businesses with unknown '%s' (NaN). These businesses will be dropped.",
            businesses[jobs_var].isna().sum(),
            jobs_var,
        )
        businesses = businesses.dropna(subset=jobs_var)

    current_counts = businesses.groupby(segment_variables)[jobs_var].sum()
    current_counts_local = current_counts.loc[
        current_counts.index.isin(expected_counts.index)
    ]

    deltas = expected_counts - current_counts_local

    if deltas.isna().any():
        missing_ct = deltas.loc[deltas.isna()].index
        logging.warning(
            "For the following target %s no businesses were found: %s",
            segment_variables,
            missing_ct.to_list(),
        )
        if "zone_id" in deltas.index.names:
            logging.warning(
                "These zones will be fitted with a sample business at its centroid."
            )
            sample_businesses = get_sample_businesses(businesses, missing_ct)
            businesses = pd.concat([businesses, sample_businesses])
            current_counts = businesses.groupby(segment_variables)[jobs_var].sum()
            deltas = (
                expected_counts
                - current_counts.loc[current_counts.index.isin(expected_counts.index)]
            )
        else:
            logging.warning("These zones will be ignored.")
            deltas = deltas.dropna()

    deltas = deltas.round(0).astype(int)

    scaled_businesses = scale_jobs(
        deltas=deltas,
        businesses=businesses,
        jobs_var=jobs_var,
        segment_variables=segment_variables,
        lower_limit=lower_limit,
    )

    not_in_scope = current_counts.loc[
        ~current_counts.index.isin(expected_counts.index)
    ].sum()
    logging.info("Original %s: %i", jobs_var, businesses[jobs_var].sum())
    logging.info("New %s: %i", jobs_var, scaled_businesses[jobs_var].sum())
    logging.info("Target was: %i", expected_counts.sum() + not_in_scope)
    logging.info(
        "Diff to targets: %i",
        expected_counts.sum() + not_in_scope - scaled_businesses[jobs_var].sum(),
    )

    # Calculate FTE change
    fte_var = jobs_var.replace("jobs", "fte")
    scaled_bs_ids = scaled_businesses.query("scaled")["business_id"].values
    mask = businesses["business_id"].isin(scaled_bs_ids)
    avg_fte = (
        (businesses.loc[mask, fte_var] / businesses.loc[mask, jobs_var])
        .fillna(1)
        .clip(upper=1, lower=0.01)
    )

    mask = scaled_businesses["business_id"].isin(scaled_bs_ids)
    assert len(avg_fte) == mask.sum()

    scaled_businesses.loc[mask, fte_var] = (
        scaled_businesses.loc[mask, jobs_var] * avg_fte.values
    )

    return scaled_businesses
