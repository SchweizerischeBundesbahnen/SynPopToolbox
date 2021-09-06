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

# Households

:::{note}
The comparisons to BFS data presented here refer to the ["Szenarien zur Entwicklung der Haushalte 2020-2050"](https://www.bfs.admin.ch/bfs/de/home/statistiken/bevoelkerung/zukuenftige-entwicklung/haushaltsszenarien.assetdetail.16344855.html), at its reference scenario.
:::

+++ {"tags": ["remove-cell"]}

* **Input**: cleaned and optimized pickeled DataFrames (persons.csv and households.csv) for 20XX
* **Output**: Visualisations

+++ {"toc": true, "tags": ["remove-cell"]}

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Settings" data-toc-modified-id="Settings-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Settings</a></span></li><li><span><a href="#Loading-Data" data-toc-modified-id="Loading-Data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Loading Data</a></span><ul class="toc-item"><li><span><a href="#SynPop" data-toc-modified-id="SynPop-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>SynPop</a></span></li><li><span><a href="#BFS---Predictions" data-toc-modified-id="BFS---Predictions-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>BFS - Predictions</a></span></li></ul></li><li><span><a href="#Analysis" data-toc-modified-id="Analysis-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Analysis</a></span><ul class="toc-item"><li><span><a href="#Household-Size" data-toc-modified-id="Household-Size-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Household Size</a></span></li><li><span><a href="#Number-of-Children" data-toc-modified-id="Number-of-Children-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Number of Children</a></span></li></ul></li><li><span><a href="#Export-Notebook-to-HTML" data-toc-modified-id="Export-Notebook-to-HTML-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Export Notebook to HTML</a></span></li></ul></div>

```{code-cell} ipython3
:tags: [remove-cell]

import os
import sys
import re
import pandas as pd

from matplotlib import pyplot as plt
```

```{code-cell} ipython3
:tags: [remove-cell]

import yaml
from myst_nb import glue

# not displayed
# load config and synpop code
with open('_config.yml', 'r') as f:
    config = yaml.safe_load(f)['synpop_report_config']

sys.path.append(config['codebase'])

import synpop.utils as utils
from synpop.synpop_tables import Persons
from synpop import visualisations
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

# Download from: 
#  https://www.bfs.admin.ch/bfs/fr/home/statistiques/population/evolution-future/scenarios-menages.assetdetail.3623353.html
#  https://www.bfs.admin.ch/bfs/de/home/statistiken/bevoelkerung/zukuenftige-entwicklung/haushaltsszenarien.assetdetail.16344855.html
BFS_HOUSEHOLD_STATS_EXCEL_IST = r'\\Filer16L\P-V160L\SIMBA.A11244\40_Projekte\20190101_Synpop_2030\20_Arbeiten\20_Daten_BFS\BFS2015\su-f-01.03.03.03.xlsx'
BFS_HOUSEHOLD_STATS_EXCEL = r'\\Filer16L\P-V160L\SIMBA.A11244\40_Projekte\20190101_Synpop_2030\20_Arbeiten\20_Daten_BFS\su-f-01.03.03.03.xlsx'

SAVE_FIGURES = False
```

+++ {"tags": ["remove-cell"]}

## Loading Data

+++ {"tags": ["remove-cell"]}

### SynPop

```{code-cell} ipython3
:tags: [remove-cell]

%%time
synpop_persons = Persons(YEAR)
synpop_persons.load(SYNPOP_PERSONS_FILE)
```

```{code-cell} ipython3
:tags: [remove-cell]

%%time
synpop_persons_ist = Persons(YEAR_IST)
synpop_persons_ist.load(SYNPOP_PERSONS_FILE_IST)
```

```{code-cell} ipython3
:tags: [remove-cell]

# as defined for the ownership model
def get_hh_structure(pers_df):
    pers_df = pers_df.copy()
    # find out if the person has a child but is not the child itself
    pers_df['has_child_in_hh'] = False
    youngest_person_in_hh = pers_df.groupby('household_id')['age'].min().rename('youngest_person_in_hh')
    pers_df = pers_df.join(youngest_person_in_hh, on='household_id')
    pers_df.loc[((pers_df['youngest_person_in_hh'] < 18) & (pers_df['age'] >= 18)),
                'has_child_in_hh'] = True
    alone_in_hh = pers_df.groupby('household_id')['person_id'].count().rename('alone_in_hh')
    alone_in_hh = alone_in_hh == 1
    pers_df = pers_df.join(alone_in_hh, on='household_id')

    hh_df = pers_df.groupby('household_id')['person_id'].count().to_frame('members')

    # get number of adults and number of kids per household
    hh_df['nb_of_kids'] = pers_df.loc[pers_df['age'] <= 17, 'household_id'].value_counts().reindex(hh_df.index).fillna(0).astype(int)
    hh_df['nb_of_adults'] = pers_df.loc[pers_df['age'] >= 18, 'household_id'].value_counts().reindex(hh_df.index).fillna(0).astype(int)

    # get couple type age cats for each household
    hh_df["under24"] = pers_df.loc[pers_df['age'] <= 23, 'household_id'].value_counts().reindex(hh_df.index)
    hh_df["over24"] = pers_df.loc[pers_df['age'] >= 24, 'household_id'].value_counts().reindex(hh_df.index)
    hh_df = hh_df.fillna(0)

    # create couple type variable
    hh_df.loc[(hh_df["under24"] >= 1) & (hh_df["over24"] >= 2) & (hh_df["over24"] <= 3), 'couple_type'] = 'couple_with_children'
    hh_df.loc[(hh_df["under24"] == 0) & (hh_df["over24"] >= 2) & (hh_df["over24"] <= 3), 'couple_type'] = 'couple_without_children'
    hh_df = hh_df.fillna("not a couple")
    
    return pers_df, hh_df
```

```{code-cell} ipython3
:tags: [remove-cell]

persons_ist, households_ist = get_hh_structure(synpop_persons_ist.data)
persons, households = get_hh_structure(synpop_persons.data)
```

+++ {"tags": ["remove-cell"]}

### BFS - Predictions

```{code-cell} ipython3
:tags: [remove-cell]

def read_bfs_pred(year):
    path = BFS_HOUSEHOLD_STATS_EXCEL_IST if year < 2020 else BFS_HOUSEHOLD_STATS_EXCEL
    pc_households_bsf_pred = (pd.read_excel(path, header=1+(year < 2020), engine='openpyxl')
                          .iloc[8 + 3*(year < 2020):14 + 3*(year < 2020)]
                          .drop('Unnamed: 0', axis=1)
                          .loc[:, year]
                          .astype(float)
                          .round(2)
                          )
    pc_households_bsf_pred.index = [1, 2, 3, 4, 5, 6]
    pc_households_bsf_pred.index.name = 'members'
    return pc_households_bsf_pred
```

```{code-cell} ipython3
:tags: [remove-cell]

pc_households_bsf_pred = read_bfs_pred(YEAR)
pc_households_bsf_ist = read_bfs_pred(YEAR_IST)
```

```{code-cell} ipython3
:tags: [remove-cell]

if YEAR == YEAR_IST:
    YEAR_IST = f'{YEAR_IST} (Ref)'
```

```{code-cell} ipython3
:tags: [remove-cell]

bfs_pred = pd.concat([pc_households_bsf_ist, pc_households_bsf_pred], axis=1)
bfs_pred = bfs_pred.rename(columns={YEAR: f'BFS-{YEAR}', int(re.sub("\D", "", str(YEAR_IST))): f'BFS-{YEAR_IST}'})
bfs_pred
```

+++ {"tags": ["remove-cell"]}

# Analysis

+++

## Household Size

```{code-cell} ipython3
:tags: [remove-cell]

members_per_hh = households.reset_index().groupby('members').count().iloc[:, 0]
pc_members_per_hh = (members_per_hh / members_per_hh.sum() * 100).rename(f'SynPop{YEAR}').iloc[:6]

members_per_hh_ist = households_ist.reset_index().groupby('members').count().iloc[:, 0]
pc_members_per_hh_ist = (members_per_hh_ist / members_per_hh_ist.sum() * 100).rename(f'SynPop{YEAR_IST}').iloc[:6]
```

+++ {"tags": ["remove-cell"]}

**Plotting SynPop vs. BFS**

```{code-cell} ipython3
:tags: [remove-cell]

hh_sizes_stats = pd.concat([bfs_pred, pc_members_per_hh_ist, pc_members_per_hh], axis=1)
```

```{code-cell} ipython3
:tags: [remove-cell]

hh_sizes_stats
```

```{code-cell} ipython3
:tags: [remove-cell]

ax = hh_sizes_stats.plot.bar(figsize=(10, 6), alpha=0.8, color=['k', 'grey', 'C0', 'C1'])
ax.grid(axis='y')
plt.ylabel('% of HH')
title = 'Households Proportions by HH-Size: BFS vs. SynPop'
_ = ax.set_title(title, fontdict={'fontsize': 14, 'fontweight': 'bold'})
```

## Household Structure

+++ {"tags": ["remove-cell"]}

### Number of Children

```{code-cell} ipython3
:tags: [remove-cell]

children_per_hh = households.groupby('nb_of_kids').sum().rename(columns={'members': 'nbr_hh'})
children_per_hh['SynPop{}'.format(YEAR)] = children_per_hh['nbr_hh'] / children_per_hh['nbr_hh'].sum()

children_per_hh_ist = households_ist.groupby('nb_of_kids').sum().rename(columns={'members': 'nbr_hh'})
children_per_hh_ist['SynPop{}'.format(YEAR_IST)] = children_per_hh_ist['nbr_hh'] / children_per_hh_ist['nbr_hh'].sum()

prop_hh_nbr_children = pd.merge(children_per_hh_ist.loc[:,'SynPop{}'.format(YEAR_IST)], 
                                children_per_hh.loc[:,'SynPop{}'.format(YEAR)], left_index=True, right_index=True)
```

```{code-cell} ipython3
:tags: [remove-cell]

ax = prop_hh_nbr_children.iloc[:10].plot.bar(alpha=0.8, figsize=(10, 6))
ax.grid(axis='y')
plt.ylabel('Proportion of HH')
title = 'Children per HH'
_ = ax.set_title(title, fontdict={'fontsize': 14, 'fontweight': 'bold'})
glue("children_per_hh", ax.get_figure(), display=False)
```

+++ {"tags": ["remove-cell"]}

### Adults per HH

```{code-cell} ipython3
:tags: [remove-cell]

adults_per_hh = households.groupby('nb_of_adults').sum().rename(columns={'members': 'nbr_hh'})
adults_per_hh['SynPop{}'.format(YEAR)] = adults_per_hh['nbr_hh'] / adults_per_hh['nbr_hh'].sum()

adults_per_hh_ist = households_ist.groupby('nb_of_adults').sum().rename(columns={'members': 'nbr_hh'})
adults_per_hh_ist['SynPop{}'.format(YEAR_IST)] = adults_per_hh_ist['nbr_hh'] / adults_per_hh_ist['nbr_hh'].sum()

prop_hh_nbr_adults = pd.merge(adults_per_hh_ist.loc[:,'SynPop{}'.format(YEAR_IST)], 
                              adults_per_hh.loc[:,'SynPop{}'.format(YEAR)], left_index=True, right_index=True)
```

```{code-cell} ipython3
:tags: [remove-input]

ax = prop_hh_nbr_adults.iloc[:10].plot.bar(alpha=0.8, figsize=(10, 6))
ax.grid(axis='y')
plt.ylabel('Proportion of HH')
title = 'Adults per HH'
_ = ax.set_title(title, fontdict={'fontsize': 14, 'fontweight': 'bold'})
glue("adults_per_hh", ax.get_figure(), display=False)
```

+++ {"tags": ["remove-cell"]}

### Couple Type

```{code-cell} ipython3
:tags: [remove-cell]

ax = pd.concat([(households_ist['couple_type']
                 .value_counts()
                 .div(len(households_ist))
                 .rename('SynPop{}'.format(YEAR_IST))),
                (households['couple_type']
                 .value_counts()
                 .div(len(households))
                 .rename('SynPop{}'.format(YEAR)))], axis=1
              ).plot.bar()
ax.grid(axis='y')
plt.ylabel('Proportions')
plt.xticks(rotation=10)
title = 'Couple Types'
_ = ax.set_title(title, fontdict={'fontsize': 14, 'fontweight': 'bold'})
glue("couple_type", ax.get_figure(), display=False)
```

````{tabbed} Children per Household
```{glue:figure} children_per_hh
:figwidth: 800px
```
````

````{tabbed} Adults per Household
```{glue:figure} adults_per_hh
:figwidth: 800px
```
````

````{tabbed} Couple Type
```{glue:figure} couple_type
:figwidth: 800px
```
````

### Household Structure per age

```{code-cell} ipython3
:tags: [remove-cell]

print(f"Total number of persons with a child at home in {YEAR_IST}: {pers_ist['has_child_in_hh'].sum()} ({pers_ist['has_child_in_hh'].mean():.2%})")
```

```{code-cell} ipython3
:tags: [remove-cell]

print(f"Total number of persons with a child at home in {YEAR}: {pers['has_child_in_hh'].sum()} ({pers['has_child_in_hh'].mean():.2%})")
```

```{code-cell} ipython3
:tags: [remove-cell]

has_child_in_hh_ist = persons_ist.groupby('age').agg({'has_child_in_hh': 'sum', 'person_id': 'count'})
has_child_in_hh_ist['person_id'] = has_child_in_hh_ist['person_id'] - has_child_in_hh_ist['has_child_in_hh']
has_child_in_hh_ist.columns = ['true', 'false']
```

```{code-cell} ipython3
:tags: [remove-cell]

title = f'Child in HH by Age - SynPop{YEAR_IST}'
ax = visualisations.plot_multi_class_feature_per_age(has_child_in_hh_ist, colour_dict=visualisations.COLOUR_DICTS['is_swiss'], ymax=150_000, title=title)
glue("child_in_hh_per_age_ist", ax.get_figure(), display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

has_child_in_hh = persons.groupby('age').agg({'has_child_in_hh': 'sum', 'person_id': 'count'})
has_child_in_hh['person_id'] = has_child_in_hh['person_id'] - has_child_in_hh['has_child_in_hh']
has_child_in_hh.columns = ['true', 'false']
```

```{code-cell} ipython3
:tags: [remove-cell]

title = f'Child in HH by Age - SynPop{YEAR}'
ax = visualisations.plot_multi_class_feature_per_age(has_child_in_hh, colour_dict=visualisations.COLOUR_DICTS['is_swiss'], ymax=150_000, title=title)
glue("child_in_hh_per_age", ax.get_figure(), display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

alone_in_hh_ist = persons_ist.groupby('age').agg({'alone_in_hh': 'sum', 'person_id': 'count'})
alone_in_hh_ist['person_id'] = alone_in_hh_ist['person_id'] - alone_in_hh_ist['alone_in_hh']
alone_in_hh_ist.columns = ['true', 'false']
```

```{code-cell} ipython3
:tags: [remove-cell]

title = f'Is Alone in HH by Age - SynPop{YEAR_IST}'
ax = visualisations.plot_multi_class_feature_per_age(alone_in_hh_ist, colour_dict=visualisations.COLOUR_DICTS['is_swiss'], ymax=150_000, title=title)
glue("isalone_in_hh_per_age_ist", ax.get_figure(), display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

alone_in_hh = persons.groupby('age').agg({'alone_in_hh': 'sum', 'person_id': 'count'})
alone_in_hh['person_id'] = alone_in_hh['person_id'] - alone_in_hh['alone_in_hh']
alone_in_hh.columns = ['true', 'false']
```

```{code-cell} ipython3
:tags: [remove-cell]

title = f'Is Alone in HH by Age - SynPop{YEAR}'
ax = visualisations.plot_multi_class_feature_per_age(alone_in_hh, colour_dict=visualisations.COLOUR_DICTS['is_swiss'], ymax=150_000, title=title)
glue("isalone_in_hh_per_age", ax.get_figure(), display=False)
```

### SynPop Ref

+++

````{tabbed} Child in Household
```{glue:figure} child_in_hh_per_age_ist
:figwidth: 800px
```
````

````{tabbed} Single-dweller per age
```{glue:figure} isalone_in_hh_per_age_ist
:figwidth: 800px
```
````

+++

### SynPop Scenario

+++

````{tabbed} Child in Household
```{glue:figure} child_in_hh_per_age
:figwidth: 800px
```
````

````{tabbed} Single-dweller per age
```{glue:figure} isalone_in_hh_per_age
:figwidth: 800px
```
````
