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

# Level of Employment

```{note}
The comparisons to BFS data presented here refer to the ["Szenarien zur Entwicklung der Erwerbsbevölkerung 2020-2050"](https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken/daten.assetdetail.12947685.html), at its reference scenario.
```

```{important}
"Erwerbsbevölkerung" includes unemployed actively searching for a job, this is not 100% compatible with MOBi but serves as a good proxy.

In some cases a 2% unemployment assumption is used as a simple correction for the analysis. A note will be included whenever this is the case.
```

+++ {"tags": ["remove-cell"]}

* **Input**: cleaned and optimized pickeled DataFrames (persons.csv) for 20XX, BSF age predictions
* **Output**: Visualisations

+++ {"toc": true, "tags": ["remove-cell"]}

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Settings" data-toc-modified-id="Settings-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Settings</a></span></li><li><span><a href="#Loading-Data" data-toc-modified-id="Loading-Data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Loading Data</a></span><ul class="toc-item"><li><span><a href="#SynPop" data-toc-modified-id="SynPop-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>SynPop</a></span></li><li><span><a href="#BFS-Erwerbsbewölkerung" data-toc-modified-id="BFS-Erwerbsbewölkerung-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>BFS Erwerbsbewölkerung</a></span></li></ul></li><li><span><a href="#Analysis" data-toc-modified-id="Analysis-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Analysis</a></span><ul class="toc-item"><li><span><a href="#Globally" data-toc-modified-id="Globally-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Globally</a></span></li><li><span><a href="#Per-Age" data-toc-modified-id="Per-Age-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Per Age</a></span><ul class="toc-item"><li><span><a href="#Scenario-Year" data-toc-modified-id="Scenario-Year-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>Scenario-Year</a></span></li><li><span><a href="#IST" data-toc-modified-id="IST-3.2.2"><span class="toc-item-num">3.2.2&nbsp;&nbsp;</span>IST</a></span></li></ul></li></ul></li></ul></div>

```{code-cell} ipython3
:tags: [remove-cell]

import logging
import os
import sys
import pandas as pd

from matplotlib import pyplot as plt
```

```{code-cell} ipython3
:tags: [remove-cell]

import yaml
from myst_nb import glue

# load config and synpop code
with open('_config.yml', 'r') as f:
    config = yaml.safe_load(f)['synpop_report_config']

sys.path.append(config['codebase'])

from synpop.marginals import FSO_ActivePopPredictionsClient
from synpop.visualisations import plot_people_employed_by_age, plot_synpop_vs_bfs_avg_fte_per_age
import synpop.utils as utils
from synpop.synpop_tables import Persons
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

## Loading Data

+++ {"tags": ["remove-cell"]}

### SynPop

```{code-cell} ipython3
:tags: [remove-cell]

SYNPOP_PERSONS_FILE
```

```{code-cell} ipython3
:tags: [remove-cell]

%%time
synpop_persons = Persons(YEAR)
synpop_persons.load(SYNPOP_PERSONS_FILE)

persons = synpop_persons.data 

# level of employment
labels = ['0%', 'part_time', '100%']
persons.loc[persons['level_of_employment'] > 100, 'level_of_employment'] = 100
persons['loe'] = pd.cut(persons['level_of_employment'], bins=(-0.01, 0, 99.9, 100), labels=labels)
```

```{code-cell} ipython3
:tags: [remove-cell]

SYNPOP_PERSONS_FILE_IST
```

```{code-cell} ipython3
:tags: [remove-cell]

%%time
synpop_persons_ist = Persons(YEAR_IST)
synpop_persons_ist.load(SYNPOP_PERSONS_FILE_IST)

persons_ist = synpop_persons_ist.data 

# level of employment
labels = ['0%', 'part_time', '100%']
persons_ist['loe'] = pd.cut(persons_ist['level_of_employment'], bins=(-0.01, 0, 99.9, 100), labels=labels)
```

+++ {"tags": ["remove-cell"]}

### BFS Erwerbsbewölkerung

+++ {"tags": ["remove-cell"]}

**Personnes active = personnes actives occupées + chômeurs**

+++ {"tags": ["remove-cell"]}

**Global**

```{code-cell} ipython3
:tags: [remove-cell]

active_pop = FSO_ActivePopPredictionsClient(granularity='global').load(YEAR)
active_pop_stats_global = active_pop.stats
```

```{code-cell} ipython3
:tags: [remove-cell]

active_pop_stats_global
```

+++ {"tags": ["remove-cell"]}

**By Age**

```{code-cell} ipython3
:tags: [remove-cell]

active_pop_client = FSO_ActivePopPredictionsClient(granularity='age').load(YEAR)
active_pop_stats_by_age = active_pop_client.stats
```

```{code-cell} ipython3
:tags: [remove-cell]

ref_year = 2019 if YEAR_IST < 2019 else YEAR_IST

active_pop_client = FSO_ActivePopPredictionsClient(granularity='age').load(ref_year)
active_pop_stats_by_age_ist = active_pop_client.stats
```

```{code-cell} ipython3
:tags: [remove-cell]

if YEAR == YEAR_IST:
    YEAR_IST = f'{YEAR_IST} (Ref)'
```

+++ {"tags": ["remove-cell"]}

## Analysis

+++

## Global

+++ {"tags": ["remove-cell"]}

**Level of Activity**

```{code-cell} ipython3
:tags: [remove-cell]

# create references
bfs_global_active = active_pop_stats_global['active_people'].iloc[0]
bfs_global_active_avg = round(active_pop_stats_global['avg_active_people'].iloc[0], 1)
synpop_global_employed = (persons['level_of_employment'] > 0).sum()
synpop_global_employed_avg = round((persons['level_of_employment'] > 0).mean()*100, 1)
diff_global_employed = (bfs_global_active - synpop_global_employed)
diff_global_employed_avg = round(bfs_global_active_avg - synpop_global_employed_avg)

# glue variables
glue("bfs_global_active", bfs_global_active)
glue("bfs_global_active_avg", bfs_global_active_avg)
glue("synpop_global_employed", synpop_global_employed)
glue("synpop_global_employed_avg", synpop_global_employed_avg)
glue("diff_global_employed", diff_global_employed)
glue("diff_global_employed_avg", diff_global_employed_avg)
```

````{panels}
:column: col-4
**BFS**
^^^
**{glue:text}`bfs_global_active` ({glue:text}`bfs_global_active_avg` %)**
---
**SynPop**
^^^
**{glue:text}`synpop_global_employed` ({glue:text}`synpop_global_employed_avg` %)**
---
**Delta**
^^^
**{glue:text}`diff_global_employed` ({glue:text}`diff_global_employed_avg` %)**
````

:::{note}
The differences are at least partially (or mostly) due to the different definitions, i.e. the inclusion of unemployed persons in the BFS statistics.
:::

+++ {"tags": ["remove-cell"]}

The difference could be explained by job seekers.

## Level of Employment Distribution

```{code-cell} ipython3
:tags: [remove-input]

title = f'Level of Employment Distribution - SynPop {YEAR_IST} vs SynPop {YEAR}'

bins = list(range(5, 101, 5))
ax = (pd.concat(
         [
             (pd.cut(persons_ist.query('is_employed')['level_of_employment'], bins=bins, labels=[f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)])
              .value_counts()
              .div(sum(persons_ist['is_employed']))
              .mul(100)
              .sort_index()), 
             (pd.cut(persons.query('is_employed')['level_of_employment'], bins=bins, labels=[f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)])
              .value_counts()
              .div(sum(persons['is_employed']))
              .mul(100)
              .sort_index())
         ], 
         axis=1, 
         keys=[YEAR_IST, YEAR])
     .plot.bar(figsize=(10, 6)))
plt.locator_params(axis='x', nbins=7)
plt.xticks(rotation=0)
plt.xlabel('Level of Employment ranges (%)', fontsize=12)
plt.ylabel('Frequency (%)', fontsize=11)
#_ = ax.set_xlim([1, 101])
#_ = ax.set_xticks(range(0,100,4))
_ = ax.set_title(title, pad=25, fontdict={'fontsize': 16, 'fontweight': 'bold'})
```

## Employment Per Age

+++ {"tags": ["remove-cell"]}

#### IST

````{margin}
```{note}
:class: dropdown
The BFS data in the reference plot refers to 2019.
```
````

```{code-cell} ipython3
:tags: [remove-input]

title = 'Active People - SynPop{year} vs. BFS-Predictions for {year}*'.format(year=ref_year)
ax = plot_people_employed_by_age(persons_ist, active_pop_stats_by_age_ist['active_people'], title=title)
plt.figtext(0.99, 0.01, 'BFS-Predictions data start in 2019.', horizontalalignment='right', fontsize=10)
glue("active_age_reference", ax.get_figure(), display=False)
```

+++ {"tags": ["remove-cell"]}

#### Scenario-Year

```{code-cell} ipython3
:tags: [remove-input]

title = 'Active People - SynPop{year} vs. BFS-Predictions for {year}'.format(year=YEAR)
ax = plot_people_employed_by_age(persons, active_pop_stats_by_age['active_people'], title=title)
glue("active_age_projection", ax.get_figure(), display=False)
```

```{code-cell} ipython3

```

## Average Level of Employment per Age

```{code-cell} ipython3
:tags: [remove-input]

title = f'Average FTE SynPop {YEAR_IST} vs BFS'
ax = plot_synpop_vs_bfs_avg_fte_per_age(persons_ist, active_pop_stats_by_age_ist['avg_fte_per_person'], YEAR_IST, title=title)
```

```{code-cell} ipython3
:tags: [remove-input]

title = f'Average FTE SynPop {YEAR} vs BFS'

ax = plot_synpop_vs_bfs_avg_fte_per_age(persons, active_pop_stats_by_age['avg_fte_per_person'], YEAR, title=title)
```

```{code-cell} ipython3

```
