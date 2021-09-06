"""
This module contains the high level objects to work with a synthetic population.
"""
import logging
from pathlib import Path

import pandas as pd
from pyarrow import feather
import numpy

# Logger settings set in __init__.py
logger = logging.getLogger(__name__)


class SynPopTable(object):
    """
    Parent class of Persons and Businesses
    """
    def __init__(self, year, data=None):
        self.year = year
        if data is not None:
            self.data = data.copy()
        else:
            self.data = data
        self.table_name = None
        self.total_rows = None
        self.source_file = None  # For documentation only
        self.modified_columns = []  # For documentation only

    def update(self, data):
        self.data = data
        self.total_rows = self.data.shape[0]

    def load(self, path, force_update_data=False):
        if self.data is not None and not force_update_data:
            raise RuntimeError("Data already exists. Re-run with 'force_update_data=True' to force updating.")

        logger.info('Loading {} ...'.format(path))
        path = Path(path)
        if not path.suffixes:  # backwards compatibility to allow reading whatever format is available
            first = list(path.parent.glob(f'{path.stem}.*'))[0]
            path = path.with_suffix(''.join(first.suffixes))
        if path.suffixes == ['.feather']:
            data = pd.read_feather(path)
        elif path.suffixes == ['.pickle', '.gzip']:
            data = pd.read_pickle(path, compression='gzip')
        else:
            raise IOError(f'Unsupported file format {path.split(".")[-1]}.')

        # backwards compatibility: attempt rename columns of old synpops
        data = data.rename(
            errors='ignore',
            columns={'location_id': 'zone_id', 'ID': 'zone_id',
                     'N_KT': 'kt_name', 'KT': 'kt_id', 'N_Gem': 'mun_name', 'ID_Gem': 'mun_id',
                     'jobs_ch': 'jobs_endo', 'jobs_cb': 'jobs_exo', 'fte_ch': 'fte_endo', 'fte_cb': 'fte_exo'})

        self.update(data)
        self.source_file = path
        logger.info('Table {} loaded with {} rows.'.format(self.table_name, self.total_rows))

    def load_pickle(self, path, compression='gzip'):
        logger.warning("Pickle format is deprecated. Please switch to feather format.")
        self.load(path)

    def drop_column(self, column_name):
        try:
            self.data = self.data.drop(column_name, axis=1)
            logger.debug('Columns "{}" has been dropped.'.format(column_name))
        except KeyError:
            logger.warning('No columns "{}" in the {} table. It will be created.'.format(column_name, self.table_name))

    def write(self, path):
        logger.info('Writing to {} ...'.format(path))
        path = str(path)
        if path.endswith('.feather'):
            feather.write_feather(self.data, path, compression_level=9, compression='zstd')
        elif path.endswith('.pickle.gzip'):
            logger.warning("Pickle format is deprecated. Please switch to feather format.")
            self.data.to_pickle(path, compression='gzip')
        else:
            raise IOError(f'Unsupported file format {path.split(".")[-1]}.')
        logger.info('{} table of {} saved to file: {}'.format(self.table_name, self.year, path))

    def to_pickle(self, file_path, compression='gzip'):
        logger.warning("Pickle format is deprecated. Please switch to feather format.")
        self.write(file_path)


class Persons(SynPopTable):
    """
    All persons and their attributes.
    """
    def __init__(self, year, data=None):
        super().__init__(year, data)
        self.table_name = 'persons'

    def overwrite_column(self, attribute_name, new_values):
        """
        This attribute in the persons table will be overwritten using person_id as joining key. This function only
        works if all the person_ids have a matching new id. It may be extended later to overwrite only when exists.

        :param attribute_name: name of the attribute
        :param new_values: Must be a series with the new values and with  person_id as index
        """
        # Sanity checking first
        logger.info('Overwriting "{}" with new values ...'.format(attribute_name))
        assert new_values.index.nunique() == len(new_values.index), 'New values have duplicated indices!'
        assert set(self.data['person_id']) == set(new_values.index)

        self.drop_column(attribute_name)

        self.data = pd.merge(self.data, new_values.rename(attribute_name),
                             how='left', left_on='person_id', right_index=True
                             )
        self.modified_columns.append(attribute_name)
        logger.info('"{}" has been overwritten.'.format(attribute_name))

    def overwrite_apprentices(self, new_values, keep_is_apprentice_column=False):
        """
        Apprentices is a little special since we want to change only the values in the "position_in_bus" column that
        do not correspond to our new values. We keep all values the same.

        :param keep_is_apprentice_column: if False, the "is_apprentice" column is removed at the end of the operation
        :param new_values: Must be a series with the new is_apprentice values and with  person_id as index
        """
        # Sanity checking first
        logger.info('Overwriting "pos_in_business" where new "is_apprentice" values do not match...')
        assert "position_in_bus" in self.data.columns, 'Attribute name is not in the persons table!'

        # Adding "is_apprentice" to data
        self.overwrite_column('is_apprentice', new_values)

        # Adding a new category to a categorical column if necessary
        try:
            categories = list(self.data['position_in_bus'].cat.categories)
            if 'null' not in categories:
                categories.append('null')
            self.data['position_in_bus'] = self.data['position_in_bus'].cat.set_categories(categories)
        except AttributeError:
            pass  # not a categorical column

        # Changing  "position_in_bus" to apprentice for all people which is_apprentice = True
        self.data['position_in_bus'] = self.data['position_in_bus'].mask(self.data['is_apprentice'], 'apprentice')

        # Changing "position_in_bus" from "apprentice" to "null" if is_apprentice = False
        mask = (~self.data['is_apprentice']) & (self.data['position_in_bus'] == 'apprentice')
        self.data['position_in_bus'] = self.data['position_in_bus'].mask(mask, 'null')
        self.modified_columns.append('position_in_bus')

        # Remove 'is_apprentice'
        if not keep_is_apprentice_column:
            self.drop_column('is_apprentice')
            self.modified_columns.remove('is_apprentice')

        logger.info('"position_in_bus" has been overwritten to match new "is_apprentice" input.')

    def overwrite_is_swiss(self, new_values):
        """
        When fixing is_swiss column, the nationality column must also be fixed. Since we do not know which
        non-swiss nationality the people are, the nationality is chosen randomly with wights according to
        the proportions in the global population.

        :param new_values: Must be a series with the new is_swiss values and with person_id as index
        """
        # Sanity checking first
        logger.info('Overwriting "nation" where new "is_swiss" values do not match...')
        assert "nation" in self.data.columns, 'Attribute name is not in the persons table!'

        # Adding "is_swiss" to data
        self.overwrite_column('is_swiss', new_values)

        # Changing  "nation" to "swiss" for all people where "is_swiss" is "True"
        self.data['nation'] = self.data['nation'].mask(self.data['is_swiss'], 'swiss')

        # # Changing "nation" something other than "swiss" for all people where "is_swiss" is "False"
        is_new_foreigner = ((~self.data['is_swiss']) & (self.data['nation'] == 'swiss'))

        global_counts_per_nation = self.data.groupby(['nation']).count().iloc[:, 0].drop('swiss')

        foreign_nations = numpy.array(global_counts_per_nation.index)
        p = (global_counts_per_nation / global_counts_per_nation.sum()).values
        random_foreign_nations_as_replacements = numpy.random.choice(foreign_nations, self.total_rows, replace=True, p=p)

        self.data['nation'] = self.data['nation'].mask(is_new_foreigner, random_foreign_nations_as_replacements)

        logger.info('"nation" has been overwritten to match new "is_swiss" input.')


class Businesses(SynPopTable):
    """
    All businesses and their attributes.
    """
    def __init__(self, year, data=None):
        super().__init__(year, data)
        self.total_businesses = None
        self.table_name = 'businesses'


class Households(SynPopTable):
    """
    All households and their attributes.
    """
    def __init__(self, year, data=None):
        super().__init__(year, data)
        self.total_households = None
        self.table_name = 'households'


class SynPop:
    def __init__(self, year, persons=None, households=None, businesses=None):
        self.year = year
        self.persons = Persons(year, persons)
        self.households = Households(year, households)
        self.businesses = Businesses(year, businesses)

    def load(self, folder):
        # Look for files which look like SynPop-files and try to load them directly
        folder = Path(folder)
        self.persons.load(list(folder.glob('*persons*'))[0])
        self.households.load(list(folder.glob('*households*'))[0])
        self.businesses.load(list(folder.glob('*businesses*'))[0])

    def update(self, persons, households, businesses):
        self.persons.update(persons)
        self.households.update(households)
        self.businesses.update(businesses)

    def write(self, folder_path, suffix=''):
        folder_path = Path(folder_path)
        self.persons.write(folder_path / f'persons_{self.year}{suffix}.feather')
        self.households.write(folder_path / f'households_{self.year}{suffix}.feather')
        self.businesses.write(folder_path / f'businesses_{self.year}{suffix}.feather')
