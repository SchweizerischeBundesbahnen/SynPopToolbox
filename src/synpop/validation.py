"""Module for SynPop validation functions."""
import logging
from typing import Optional

from synpop import NEW_CATEGORIES
from synpop.config import SynPopConfig


def validate_persons(persons):
    """Validate persons with some consistency assertions."""
    if "is_employed" in persons.columns:
        assert (
            persons.query("~is_employed")["level_of_employment"].max() == 0
        ), 'Issue with persons "is_employed"!'
        assert (
            persons.query("is_employed")["level_of_employment"].min() > 0
        ), 'Issue with persons "is_employed"!'
    assert (persons["age"] >= 0).all(), 'Issue with persons "age"!'
    assert not (
        persons["level_of_employment"] > 100
    ).any(), 'Some people have "level_of_employment" larger than 100!'


def get_categories(table_name, synpop_loading_config: Optional[str] = None):
    """Get expected categories for categorical columns."""
    synpop_config = SynPopConfig(synpop_loading_config)
    table_config = synpop_config.categorical_keys.get(table_name, {})
    categories = {var: list(cats.values()) for var, cats in table_config.items()}
    if table_name == "persons":
        categories.update(NEW_CATEGORIES)
    return categories


def validate_categories(table, synpop_loading_config: Optional[str] = None):
    """Validate that categorical columns contain only known categories."""
    to_ignore = ["mobility", "household_model", "hcoord_type"]  # not critical
    for var, cats in get_categories(table.table_name, synpop_loading_config).items():
        if var in table.data.columns:
            mask = table.data[var].isin(cats)
            if not mask.all():
                logging.getLogger(__name__).warning(
                    "%s contains unknown categories: %s",
                    var,
                    table.data.loc[~mask, var].unique(),
                )
                if var not in to_ignore:
                    raise AssertionError
