"""
This module contains the functionality to get marginals distributions of the Swiss population.

See testing function at the bottom for example of how to use this module.
"""
import os
from hashlib import sha1
import logging
import time
import requests
import pandas
from pathlib import Path
import tempfile
from pyaxis import pyaxis
from synpop.config import SwissCantons

# Logger settings set in __init__.py
logger = logging.getLogger(__name__)

cantons = SwissCantons()

MAX_HTTP_CALL_ATTEMPTS = 5
PROXIES = {'http': 'http://zscaler.sbb.ch:9400', 'https': 'http://zscaler.sbb.ch:9400'}


def fetch_rawdata_from_bfs(url, col_filter):
    # get temporary directory (ensure it exists)
    temp_dir = Path(tempfile.gettempdir()).joinpath("BFS_rawData")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # check if cached results are available
    hash_params = sorted({**col_filter, 'url': url}.items())
    cached_file = (temp_dir / Path(sha1(repr(hash_params).encode('utf-8')).hexdigest()).with_suffix(".pickle"))
    if cached_file.exists():
        df = pandas.read_pickle(cached_file)
    else:
        logger.info(f" Cache not available. "
                    f"Parsing data from {url}. This could take a couple of minutes...")
        attempt = 0
        os.environ['http_proxy'] = ''  # make sure no environment proxy are set (happens in notebooks)
        os.environ['https_proxy'] = ''
        while True:  # Sometimes the HTTP call fails for no apparent reason -> try several times
            attempt += 1
            try:
                # monkey-patch requests in pyaxis to include proxy
                requests_session = requests.Session()
                requests_session.proxies = PROXIES
                pyaxis.requests = requests_session
                df = pyaxis.parse(url, encoding='latin1')['DATA']
                break
            except Exception as e:
                # HTTP calls sometimes fail, it usually is enough to wait a few seconds and start again.
                logger.warning(f'HTTP call attempt #{attempt} failed, attempting again... '
                               f'(max. attempts: {MAX_HTTP_CALL_ATTEMPTS})')
                time.sleep(3)
                if attempt == MAX_HTTP_CALL_ATTEMPTS:
                    logger.error('The maximum number of HTTP call attempts has been reached!')
                    raise e

        # apply filters
        ind = [True] * len(df)
        for col, vals in col_filter.items():  # Loop through filters, updating index
            if col in df.columns:
                new_ind = df[col].isin(vals)
                if not any(new_ind):
                    raise AttributeError(f'Values are not available: {vals}. Options are: {df[col].unique().tolist()}.')
                ind = ind & new_ind
            else:
                raise AttributeError(f'Column not available: {col}. Options are: {df.columns.tolist()}.')
        df = df.loc[ind]

        # cache results in disk
        df.to_pickle(cached_file)

    return df


class FSO_PopPredictionsClient:  # FSO if "BSF" in english
    """
    Client to load data from the FSO:
    Kantonale Bevölkerungsszenarien 2020-2050 - Zukünftige Bevölkerungsentwicklung nach Szenario, Staatsangehörigkeit
    (Kategorie), Geschlecht, Altersklasse / Alter und Jahr.
    links:
    - https://www.bfs.admin.ch/asset/de/px-x-0104020000_106
    - https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken/tabellen.assetdetail.12947637.html

    Only permanent residents are counted in theses predictions!
    """

    def __init__(self):
        # The following fields will be filled after the data is queried
        self.year = None
        self.scenario_name = None

        self.pop_by_canton_age_and_nationality = None
        self.pop_by_canton_and_age = None
        self.pop_by_canton_and_age_group = None
        self.pop_by_age_group = None
        self.pop_by_canton = None
        self.pop_total = None

    def load(self, year, szenario_name='Referenzszenario AR-00-2020'):
        self.year = year
        self.scenario_name = szenario_name

        self.pop_by_canton_age_and_nationality = self._get_population_by_canton_age_and_nationality()

        self.pop_by_canton_and_age = (self.pop_by_canton_age_and_nationality
                                      .groupby(['KT_full', 'kt_name', 'age'], observed=True)['pop'].sum()
                                      .reset_index()
                                      )

        self.pop_by_canton_and_age_group = self._get_population_by_canton_and_age_group()

        self.pop_by_age_group = (self.pop_by_canton_and_age_group.groupby('age_group', observed=True)
                                     .sum()
                                     .reset_index()
                                 )

        self.pop_by_canton = (self.pop_by_canton_and_age_group.groupby(['KT_full', 'kt_name'], observed=True)
                              .sum()
                              .reset_index()
                              )

        self.pop_total = self.pop_by_canton['pop'].sum()

        return self  # returns an instance of this class with the data loaded and the variables set

    def _get_population_by_canton_and_age_group(self):
        # Query data
        url = 'https://www.bfs.admin.ch/bfsstatic/dam/assets/12947640/master'

        # Create DataFrame and format
        col_filter = {'Szenario-Variante': [self.scenario_name],
                      'Jahr': ['{}'.format(self.year)],
                      'Kanton': cantons.all_canton_full_names(),
                      'Geschlecht': ['Geschlecht - Total'],
                      'Staatsangehörigkeit (Kategorie)': ['Schweiz', 'Ausland'],
                      'Altersklasse': ['0-4 Jahre', '5-9 Jahre', '10-14 Jahre', '15-19 Jahre', '20-24 Jahre',
                                       '25-29 Jahre', '30-34 Jahre', '35-39 Jahre', '40-44 Jahre', '45-49 Jahre',
                                       '50-54 Jahre', '55-59 Jahre', '60-64 Jahre', '65-69 Jahre', '70-74 Jahre',
                                       '75-79 Jahre', '80-84 Jahre', '85-89 Jahre', '90-94 Jahre', '95-99 Jahre',
                                       '100 Jahre oder mehr'],
                      'Beobachtungseinheit': ['Bevölkerungsstand am 31. Dezember']  # SynPop is calibrated on this variable
                      }

        age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
                      '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100-150']

        df = fetch_rawdata_from_bfs(url, col_filter)

        df = (df.drop(['Szenario-Variante', 'Jahr', 'Beobachtungseinheit'], axis=1)
              .rename(columns={'Kanton': 'KT_full',
                               'Altersklasse': 'age_group',
                               'DATA': 'pop'})
              .assign(kt_name=lambda x: x['KT_full'].map(cantons.full_names_to_abbreviations))
              .assign(age_group=lambda x: x['age_group'].str[:-6].str.replace('100 Jahre ode', '100-150'))
              .assign(age_group=lambda x: pandas.Categorical(x['age_group'], categories=age_groups, ordered=True))
              .astype({'KT_full': 'category', 'kt_name': 'category', 'pop': int})
              )
        df = df[['KT_full', 'kt_name', 'age_group', 'pop']]  # change column order

        return df

    def _get_population_by_canton_age_and_nationality(self):
        if self.scenario_name != 'Referenzszenario AR-00-2020':
            logger.warning('Population by age is only available for the reference scenario!')
            return None

        # Query data
        url = 'https://www.bfs.admin.ch/bfsstatic/dam/assets/12947637/master'

        # Create DataFrame and format
        years = ['{} Jahre'.format(i) for i in range(0, 101)]
        years[1] = years[1][:-1]  # 1 Jahre -> 1 Jahr
        years[-1] += ' oder mehr'  # 100 Jahre -> 100 Jahre oder mehr

        col_filter = {'Jahr': ['{}'.format(self.year)],
                      'Kanton': cantons.all_canton_full_names(),
                      'Alter': years,
                      'Geschlecht': ['Geschlecht - Total'],
                      'Staatsangehörigkeit (Kategorie)': ['Schweiz', 'Ausland'],
                      'Beobachtungseinheit': ['Bevölkerungsstand am 31. Dezember']  # SynPop is calibrated on this variable
                      }

        df = fetch_rawdata_from_bfs(url, col_filter)
        df = (df.drop(['Jahr', 'Beobachtungseinheit', 'Geschlecht'], axis=1)
              .rename(columns={'Kanton': 'KT_full',
                               'Alter': 'age',
                               'Staatsangehörigkeit (Kategorie)': 'is_swiss',
                               'DATA': 'pop'})
              .assign(kt_name=lambda x: x['KT_full'].map(cantons.full_names_to_abbreviations))
              .assign(age=lambda x: x['age'].str.replace('1 Jahr', '1 Jahre').str.replace(' oder mehr', '').str[:-6])
              .assign(is_swiss=lambda x: x['is_swiss'].map({'Schweiz': True, 'Ausland': False}))
              .astype({'KT_full': 'category', 'kt_name': 'category', 'age': int, 'is_swiss': bool, 'pop': int})
              )
        df = df[['KT_full', 'kt_name', 'age', 'is_swiss', 'pop']]  # change column order

        return df


class FSO_ActivePopPredictionsClient:  # FSO if "BSF" in english
    """
    Client to load data from the FSO:
    Szenarien zur Entwicklung der Erwerbsbevölkerung ab 2020 - Erwerbsquote und Erwerbsbevölkerung nach
    Alter/Altersklasse, Geschlecht, Staatsangehörigkeit, pro Jahr und gemäss Szenario / Variante
    link 1: https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken/daten.assetdetail.12947685.html
    link 2: https://www.bfs.admin.ch/bfs/de/home/statistiken/arbeit-erwerb/erwerbstaetigkeit-arbeitszeit/erwerbspersonen/szenarien-erwerbsbevoelkerung.assetdetail.329286.html

    Only permanent residents are counted in theses predictions!
    For the FSO, active people includes job seekers! (Personnes active = personnes actives occupées + chômeurs)
    """

    def __init__(self, granularity='age'):
        """
        :param granularity: 'age' or 'age_group'
        """
        self.granularity = granularity
        # The following fields will be filled after the data is queried
        self.year = None
        self.scenario_name = None

        self.stats = None

    def load(self, year, szenario_name=None):
        self.year = year
        if szenario_name is None:
            szenario_name = 'A-00-2015' if self.granularity == 'age_group' else 'Referenzszenario A-00-2020'
        self.scenario_name = szenario_name

        if self.granularity in ('age', 'global'):
            self.stats = self._get_stats_age()
        elif self.granularity == 'age_group':
            self.stats = self._get_stats_age_group()
        else:
            raise ValueError('"{}" is an invalid input for granularity. It must be "age", "age_group" or "global".')

        return self  # returns an instance of this class with the data loaded and the variables set

    def _get_stats_age(self):
        # Query data
        url = 'https://www.bfs.admin.ch/bfsstatic/dam/assets/12947685/master'

        # Create DataFrame and format
        col_filter = {'Szenario-Variante': [self.scenario_name],
                      'Jahr': ['{}'.format(self.year)],
                      'Geschlecht': ['Geschlecht - Total'],
                      'Staatsangehörigkeit (Kategorie)': ['Staatsangehörigkeit - Total'],
                      'Beobachtungseinheit': ['Erwerbsquote', 'Erwerbsbevölkerung',
                                              'Erwerbsquote in VZÄ', 'Erwerbsbevölkerung in VZÄ'],
                      'Alter': ['Alter - Total']
                      }
        if self.granularity == 'age':
            col_filter['Alter'] = ['.... {} Jahre'.format(age) for age in range(15, 86)]

        df = fetch_rawdata_from_bfs(url, col_filter).reset_index()
        df = df.pivot(index='Alter', values='DATA', columns='Beobachtungseinheit').reset_index()
        df = (df.rename(columns={'Erwerbsbevölkerung': 'active_people',
                                 'Erwerbsbevölkerung in VZÄ': 'active_people_fte',
                                 'Erwerbsquote': 'avg_active_people',
                                 'Erwerbsquote in VZÄ': 'avg_fte_per_person',
                                 'Alter': 'age'
                                 })
              .astype({'active_people': float, 'active_people_fte': float, 'avg_active_people': float,
                       'avg_fte_per_person': float})
              .astype({'active_people': int, 'active_people_fte': int}))
        df.columns.name = None

        if self.granularity == 'age':
            df['age'] = df['age'].str[5:-6].astype(int)
            df = df.sort_values('age').set_index('age')
        else:
            del df['age']

        return df

    def _get_stats_age_group(self):
        url = 'https://www.bfs.admin.ch/bfsstatic/dam/assets/329286/master'

        # Create DataFrame and format
        col_filter = {'Szenario-Variante': [self.scenario_name],
                      'Jahr': ['{}'.format(self.year)],
                      'Geschlecht': ['Geschlecht - Total'],
                      'Variable': ['Erwerbsquote', 'Erwerbsbevölkerung',
                                   'Erwerbsquote in VZÄ', 'Erwerbsbevölkerung in VZÄ'],
                      'Staatsangehörigkeit': ['Staatsangehörigkeit - Total'],
                      'Altersklasse': ['15-19 Jahre', '20-24 Jahre', '25-29 Jahre', '30-34 Jahre',
                                       '35-39 Jahre', '40-44 Jahre', '45-49 Jahre', '50-54 Jahre',
                                       '55-59 Jahre', '60-64 Jahre', '65-69 Jahre', '70-74 Jahre',
                                       '75-79 Jahre', '80-84 Jahre', '85-99 Jahre']
                      }

        df = fetch_rawdata_from_bfs(url, col_filter).reset_index()
        df = df.pivot(index='Altersklasse', values='DATA', columns='Variable').reset_index()
        df = (df.rename(columns={'Erwerbsbevölkerung': 'active_people',
                                 'Erwerbsbevölkerung in VZÄ': 'active_people_fte',
                                 'Erwerbsquote': 'avg_active_people',
                                 'Erwerbsquote in VZÄ': 'avg_fte_per_person',
                                 'Altersklasse': 'age_group',
                                 })
              .astype({'active_people': float, 'active_people_fte': float, 'avg_active_people': float,
                       'avg_fte_per_person': float})
              .astype({'active_people': int, 'active_people_fte': int}))
        df.columns.name = None

        df['age_group'] = df['age_group'].str[:-6]
        age_groups = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64',
                      '65-69', '70-74', '75-79', '80-84', '85-99']
        df['age_group'] = pandas.Categorical(df['age_group'], categories=age_groups, ordered=True)
        df = df.set_index('age_group')

        return df


class FSO_PopStatisticsClient:  # FSO if "BSF" in english
    """
    Client to load data from the FSO:
    Ständige und nichtständige Wohnbevölkerung nach institutionellen Gliederungen, Staatsangehörigkeit, Geburtsort,
    Geschlecht und Altersklasse
    link: https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken/tabellen.assetdetail.14087564.html

    Actual data from 2010 to 2019
    """

    def __init__(self):
        # The following fields will be filled after the data is queried
        self.year = None
        self.scenario_name = None

        self.pop_by_canton_resident_status_and_age_group = None
        self.pop_by_canton_age_group = None
        self.pop_by_canton = None
        self.pop_by_age_group = None
        self.pop_total = None

    def load(self, year):
        self.year = year

        self.pop_by_canton_resident_status_and_age_group = self._get_pop_by_canton_resident_status_and_age_group()

        self.pop_by_canton_age_group = (self.pop_by_canton_resident_status_and_age_group
                                        .groupby(['KT_full', 'kt_name', 'age_group'], observed=True).agg({'pop': sum})
                                        .reset_index()
                                        )

        self.pop_by_age_group = (self.pop_by_canton_age_group
                                 .groupby('age_group', observed=True).agg({'pop': sum})
                                 .reset_index()
                                 )

        self.pop_by_canton = (self.pop_by_canton_age_group
                              .groupby(['KT_full', 'kt_name'], observed=True).agg({'pop': sum})
                              .reset_index()
                              )

        self.pop_total = self.pop_by_canton['pop'].sum()

        return self  # returns an instance of this class with the data loaded and the variables set

    def _get_pop_by_canton_resident_status_and_age_group(self):
        # Query data
        url = 'https://www.bfs.admin.ch/bfsstatic/dam/assets/14087564/master'

        # Create DataFrame and format
        all_cantons = cantons.all_canton_full_names()
        col_filter = {'Jahr': ['{}'.format(self.year)],  # year in as a string
                      'Kanton (-) / Bezirk (>>) / Gemeinde (......)': ['- {}'.format(c) for c in all_cantons],
                      'Geschlecht': ['Geschlecht - Total'],
                      'Staatsangehörigkeit (Kategorie)': ['Staatsangehörigkeit (Kategorie) - Total'],
                      'Geburtsort': ['Geburtsort - Total'],
                      'Altersklasse': ['0-4 Jahre', '5-9 Jahre', '10-14 Jahre', '15-19 Jahre', '20-24 Jahre',
                                       '25-29 Jahre', '30-34 Jahre', '35-39 Jahre', '40-44 Jahre', '45-49 Jahre',
                                       '50-54 Jahre', '55-59 Jahre', '60-64 Jahre', '65-69 Jahre', '70-74 Jahre',
                                       '75-79 Jahre', '80-84 Jahre', '85-89 Jahre', '90-94 Jahre', '95-99 Jahre',
                                       '100 Jahre und mehr'],
                      'Bevölkerungstyp': ['Ständige Wohnbevölkerung', 'Nichtständige Wohnbevölkerung']
                      }

        age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
                      '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100-120']

        df = fetch_rawdata_from_bfs(url, col_filter)
        df = (df.drop(['Jahr', 'Staatsangehörigkeit (Kategorie)', 'Geburtsort', 'Geschlecht'], axis=1)
              .rename(columns={'Kanton (-) / Bezirk (>>) / Gemeinde (......)': 'KT_full',
                               'Bevölkerungstyp': 'resident',
                               'Altersklasse': 'age_group',
                               'DATA': 'pop'})
              .assign(KT_full=lambda x: x['KT_full'].str[2:])  # remove ' - '
              .assign(kt_name=lambda x: x['KT_full'].map(cantons.full_names_to_abbreviations))
              .assign(resident=lambda x: x['resident'] == 'Ständige Wohnbevölkerung')
              .assign(age_group=lambda x: x['age_group'].str[:-6].str.replace('100 Jahre un', '100-120'))
              .assign(age_group=lambda x: pandas.Categorical(x['age_group'], categories=age_groups, ordered=True))
              .astype({'KT_full': 'category', 'kt_name': 'category', 'pop': int})
              )
        df = df[['KT_full', 'kt_name', 'age_group', 'resident', 'pop']]  # change column order
        return df


# Testing the classes declared above (The following code can be used a user-guide.)
def _testing_pop_predictions():
    print('\n\nTesting FSO_PopPredictionsClient().load(2030): ')
    pop_predictions_2030 = FSO_PopPredictionsClient().load(2030)

    print("\npop_by_canton_age_and_nationality ==>\n", pop_predictions_2030.pop_by_canton_age_and_nationality.head(3).T)
    print("\npop_by_canton_and_age ==>\n", pop_predictions_2030.pop_by_canton_and_age.head(3).T)
    print("\npop_by_canton_and_age_group ==>\n", pop_predictions_2030.pop_by_canton_and_age_group.head(3).T)
    print("\npop_by_canton ==>\n", pop_predictions_2030.pop_by_canton.head(3).T)
    print("\npop_total ==> ", pop_predictions_2030.pop_total)


def _testing_active_pop_predictions():
    # Global
    active_pop_2017 = FSO_ActivePopPredictionsClient(granularity='global').load(2019)
    print("\n\nTesting FSO_ActivePopPredictionsClient(granularity='global').load(2019): ")
    print(active_pop_2017.stats.head(3).T)

    # By Age
    active_pop_2017 = FSO_ActivePopPredictionsClient(granularity='age').load(2019)
    print("\n\nTesting FSO_ActivePopPredictionsClient(granularity='age').load(2019): ")
    print(active_pop_2017.stats.head(3).T)

    # By Age Group
    active_pop_2017 = FSO_ActivePopPredictionsClient(granularity='age_group').load(2019)
    print("\n\nTesting FSO_ActivePopPredictionsClient(granularity='age_group').load(2019): ")
    print(active_pop_2017.stats.head(3).T)


def _testing_stats():
    pop_stats_2017 = FSO_PopStatisticsClient().load(2017)

    print('\n\nTesting FSO_PopStatisticsClient().load(2017): ')
    print("\npop_by_canton_resident_status_and_age_group ==>\n",
          pop_stats_2017.pop_by_canton_resident_status_and_age_group.head(3).T)
    print("\npop_by_canton_age_group ==>\n", pop_stats_2017.pop_by_canton_age_group.head(3).T)
    print("\npop_by_canton ==>\n", pop_stats_2017.pop_by_canton.head(3).T)
    print("\npop_by_age_group ==>\n", pop_stats_2017.pop_by_age_group.head(3).T)
    print("\npop_total ==> ", pop_stats_2017.pop_total)


if __name__ == '__main__':
    _testing_pop_predictions()
    _testing_stats()
    _testing_active_pop_predictions()
