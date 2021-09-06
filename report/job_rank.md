---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [remove-cell]

%matplotlib inline
```

+++ 

# Job Rank

+++ {"tags": ["remove-cell"]}

**Comments**: 

* *current_job_rank* is fixed to match the proportions per age in the 2017 SynPop
* The choice of persons to fix is done with a probability model based on their *is_employed* and *current_edu*

```{code-cell} ipython3
:tags: [remove-cell]

import logging
import os
import sys

from matplotlib import pyplot as plt
import pandas as pd
```

```{code-cell} ipython3
:tags: [remove-cell]

import yaml
from myst_nb import glue

# load config and synpop code
with open('_config.yml', 'r') as f:
    config = yaml.safe_load(f)['synpop_report_config']

sys.path.append(config['codebase'])

from synpop.visualisations import save_figure, plot_multi_class_feature_per_age, plot_ct1, plot_ct2, plot_ct3
from synpop import marginal_fitting, utils
from synpop.synpop_tables import Persons
from synpop.marginals import FSO_ActivePopPredictionsClient
```

+++ {"tags": ["remove-cell"]}

##  Settings

```{code-cell} ipython3
:tags: [remove-cell]

# not displayed
YEAR_IST = config['reference_year']
YEAR = config['target_year']



DATA_DIR_IST = config['reference_synpop']
SYNPOP_PERSONS_FILE_IST = os.path.join(DATA_DIR_IST, 'persons_{}'.format(YEAR_IST))

DATA_DIR = config['target_synpop']
SYNPOP_PERSONS_FILE = os.path.join(DATA_DIR, 'persons_{}'.format(YEAR))

SAVE_FIGURES = False
```

+++ {"tags": ["remove-cell"]}

# Loading Data

+++ {"tags": ["remove-cell"]}

## SynPop

```{code-cell} ipython3
:tags: [remove-cell]

COLUMNS_OF_INTEREST = ['person_id', 'level_of_employment', 'position_in_bus', 'position_in_edu', 'age', 'zone_id', 
                       'is_employed', 'loe_group', 'current_edu', 'current_job_rank', 'is_apprentice', 
                       'kt_id', 'kt_name', 'mun_name', 'KT_full' ]
```

+++ {"tags": ["remove-cell"]}

**Scenario Year**

```{code-cell} ipython3
:tags: [remove-cell]

SYNPOP_PERSONS_FILE
```

```{code-cell} ipython3
:tags: [remove-cell]

%%time
synpop_persons = Persons(YEAR)
synpop_persons.load(SYNPOP_PERSONS_FILE)

persons = synpop_persons.data[COLUMNS_OF_INTEREST].copy(deep=True)

del synpop_persons
```

```{code-cell} ipython3
:tags: [remove-cell]

persons.head()
```

```{code-cell} ipython3
:tags: [remove-cell]

persons['is_employed'].sum()/0.98
```

+++ {"tags": ["remove-cell"]}

**Year IST**

```{code-cell} ipython3
:tags: [remove-cell]

SYNPOP_PERSONS_FILE_IST
```

```{code-cell} ipython3
:tags: [remove-cell]

%%time
synpop_persons_ist = Persons(YEAR_IST)
synpop_persons_ist.load(SYNPOP_PERSONS_FILE_IST)

persons_ist = synpop_persons_ist.data[COLUMNS_OF_INTEREST].copy(deep=True)

del synpop_persons_ist

if YEAR == YEAR_IST:
    YEAR_IST = f'{YEAR_IST} (Ref)'
```

```{code-cell} ipython3
:tags: [remove-cell]

print('Imported data:')
print('persons DataFrame for {}: {}'.format(YEAR_IST, persons_ist.shape))
print('persons DataFrame for {}: {}'.format(YEAR, persons.shape))
```

+++ {"tags": ["remove-cell"]}

## FSO Data

+++ {"tags": ["remove-cell"]}

* In order for the cross-tables to work, the sum of *current_job_rank* that are either *employee*, *management* or *apprentice* must match the number of employed people.

```{code-cell} ipython3
:tags: [remove-cell]

+++ {"tags": ["remove-cell"]}

# This is a function that is only used for this Notebook so stays here
def prepare_bfs_level_of_activity_marginal_table(bfs_avg_active_people_by_age, all_ages_set, year):
    
    prop_active = (bfs_avg_active_people_by_age.reindex(all_ages_set, fill_value=0) / 100).to_frame()
    prop_inactive = 1 - prop_active

    prop_active['is_employed'] = True
    prop_active = prop_active.reset_index().set_index(['age', 'is_employed'])
    prop_inactive['is_employed'] = False
    prop_inactive = prop_inactive.reset_index().set_index(['age', 'is_employed'])

    bfs_prop_predictions = pd.concat([prop_active, prop_inactive]).sort_index()

    bfs_prop_predictions = bfs_prop_predictions.iloc[:, 0].rename(f'expected_ratios_{year}')
    return bfs_prop_predictions
```

+++ {"tags": ["remove-cell"]}

*Scenario Year*

```{code-cell} ipython3
:tags: [remove-cell]

active_pop_client = FSO_ActivePopPredictionsClient(granularity='age').load(YEAR)
active_pop_stats_by_age = active_pop_client.stats

all_ages = set(persons['age'].unique())
bfs_predicted_ratio = prepare_bfs_level_of_activity_marginal_table(active_pop_stats_by_age['avg_active_people'], all_ages, YEAR)
```

+++ {"tags": ["remove-cell"]}

### FSO without job seekers

+++ {"tags": ["remove-cell"]}

* BFS predicts active people which includes job seekers.

* In MOBi, all working people go to a job, job seekers have level_of_employment = 0.

* Based on the BFS own definition, the unemployement rate in 2017 is about 4.5%. 

References: 

* https://www.seco.admin.ch/seco/fr/home/wirtschaftslage---wirtschaftspolitik/Wirtschaftslage/Arbeitslosenzahlen.html

* https://www.bfs.admin.ch/bfs/fr/home/statistiques/travail-remuneration/activite-professionnelle-temps-travail/personnes-actives/scenarios-population-active.assetdetail.9687328.html

```{code-cell} ipython3
:tags: [remove-cell]

+++ {"tags": ["remove-cell"]}

# The very conservative assuption of 2% unemployement has been made for 2040.
UNEMPLOYMENT_RATE = 0.02

bfs_without_unemployed = bfs_predicted_ratio.copy(deep=True)
bfs_without_unemployed.loc[:, True] = bfs_without_unemployed.loc[:, True].values * (1 - UNEMPLOYMENT_RATE)
bfs_without_unemployed.loc[:, False] = (1 - bfs_without_unemployed.loc[:, True]).values
```

+++ {"tags": ["remove-cell"]}

# Examining SynPop-raw

```{code-cell} ipython3
:tags: [remove-cell]

COLOUR_DICT = {'employee': 'C0', 'management': 'C1', 'apprentice': 'C3',  'null': 'grey'}
```

```{code-cell} ipython3
:tags: [remove-cell]

ORDERED_CATEGORIES = list(persons['current_job_rank'].cat.categories)
```

+++ 

## Global Counts

```{code-cell} ipython3
:tags: [remove-cell]

%%time
summary_table = marginal_fitting.compute_comparison_summary(persons, persons_ist, 
                                                            YEAR, YEAR_IST, 
                                                            groupby='current_job_rank')
```

```{code-cell} ipython3
:tags: [remove-cell]

global_pop_growth = '{:.2f} %'.format((persons.shape[0] / persons_ist.shape[0]) - 1)
glue('global_pop_growth', global_pop_growth, display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

summary_table['growth'] = summary_table['growth'] - 1
summary_table.columns = [f'counts {YEAR_IST}', f'counts {YEAR}', 
                         'absolute growth (%)', f'proportion {YEAR_IST}', f'proportion {YEAR}']
summary_table[['absolute growth (%)', f'proportion {YEAR_IST}', f'proportion {YEAR}']] = (
    100 * summary_table[['absolute growth (%)', f'proportion {YEAR_IST}', f'proportion {YEAR}']])
summary_table.index.name = 'Job Rank'
print('As reference, the global population growth is: {}'.format(global_pop_growth))
```

```{code-cell} ipython3
:tags: [remove-input]

summary_table
```

```{note}
As reference, the global population growth is: {glue:text}`global_pop_growth`.
```

+++ 

## Job Rank Per Age

```{code-cell} ipython3
:tags: [remove-cell]

%%time 
marginals = marginal_fitting.compute_ist_vs_scenario_marginal_counts(persons, persons_ist,
                                                                     YEAR, YEAR_IST, 
                                                                     feature='current_job_rank', 
                                                                     control_level_list=['age'])
```

+++ {"tags": ["remove-cell"]}

**Year IST**

```{code-cell} ipython3
:tags: [remove-cell]

counts_per_cat_ist = (marginals
                      .reset_index()
                      .pivot(index='age', columns='current_job_rank', values=f'counts_{YEAR_IST}')
                      )
counts_per_cat_ist = counts_per_cat_ist[ORDERED_CATEGORIES]
```

```{code-cell} ipython3
:tags: [remove-input]

title = f'Current Job Rank - SynPop {YEAR_IST}'

ax = plot_multi_class_feature_per_age(counts_per_cat_ist, colour_dict=COLOUR_DICT, ymax=150_000, title=title)
```

+++ {"tags": ["remove-cell"]}

**Scenario Year**

```{code-cell} ipython3
:tags: [remove-cell]

counts_per_cat = (marginals
                  .reset_index()
                  .pivot(index='age', columns='current_job_rank', values=f'counts_{YEAR}')
                  )
counts_per_cat = counts_per_cat[ORDERED_CATEGORIES]
```

```{code-cell} ipython3
:tags: [remove-input]

title = f'Current Job Rank - SynPop {YEAR}'

ax = plot_multi_class_feature_per_age(counts_per_cat, colour_dict=COLOUR_DICT, ymax=150_000, title=title)
```

+++ 

## Cross-Relationship with Education

&NewLine;

```{code-cell} ipython3
:tags: [remove-input]

ax_ct2, _ = plot_ct2(persons_ist, title=f'Education vs Job Rank - SynPop {YEAR_IST}')
```

&NewLine;

```{code-cell} ipython3
:tags: [remove-input]

ax_ct2, _ = plot_ct2(persons, title=f'Education vs Job Rank - SynPop {YEAR}')
```

