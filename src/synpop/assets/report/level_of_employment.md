---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [remove-cell]

%matplotlib inline
```

# Level of Employment

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

sys.path.insert(1, config['codebase'])

from synpop.marginals import ActivePopPredictionsClient
from synpop import visualisations
from synpop import marginal_fitting
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

SYNPOP_PATH_IST = config['reference_synpop']

SYNPOP_PATH = config['target_synpop']
```

+++ {"tags": ["remove-cell"]}

## Loading Data

+++ {"tags": ["remove-cell"]}

### SynPop

```{code-cell} ipython3
:tags: [remove-cell]

%%time
synpop_persons = Persons(YEAR)
synpop_persons.load(SYNPOP_PATH, config["validate"])

age_groups = [0, 18, 25, 45, 65, 75, 1000]
labels = ['0-17', '18-24', '25-44', '45-64', '65-74', '75+']
loe_groups = (-0.01, 0, 40., 80., 100)
loe_labels = ['Unemployed', '1-40pct', '40-80pct', '80-100pct']
persons = synpop_persons.data
persons['age_group'] = pd.cut(persons['age'], age_groups, right=False, labels=labels)
persons['loe'] = pd.cut(persons['level_of_employment'], bins=loe_groups, labels=loe_labels)
persons['is_employed'] = persons['level_of_employment'] > 0
```

```{code-cell} ipython3
:tags: [remove-cell]

%%time
synpop_persons_ist = Persons(YEAR_IST)
synpop_persons_ist.load(SYNPOP_PATH_IST, config["validate"])

persons_ist = synpop_persons_ist.data
persons_ist['age_group'] = pd.cut(persons_ist['age'], age_groups, right=False, labels=labels)
persons_ist['loe'] = pd.cut(persons_ist['level_of_employment'], bins=loe_groups, labels=loe_labels)
persons_ist['is_employed'] = persons_ist['level_of_employment'] > 0
```

+++ {"tags": ["remove-cell"]}

### BFS Erwerbsbewölkerung

+++ {"tags": ["remove-cell"]}

**Personnes active = personnes actives occupées + chômeurs**

+++ {"tags": ["remove-cell"]}

**Global**

```{code-cell} ipython3
:tags: [remove-cell]

active_pop = ActivePopPredictionsClient(granularity='global').load(YEAR)
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

active_pop_client = ActivePopPredictionsClient(granularity='age').load(YEAR)
active_pop_stats_by_age = active_pop_client.stats
```

```{code-cell} ipython3
:tags: [remove-cell]

ref_year = 2019 if YEAR_IST < 2019 else YEAR_IST

active_pop_client = ActivePopPredictionsClient(granularity='age').load(ref_year)
active_pop_stats_by_age_ist = active_pop_client.stats
```

```{code-cell} ipython3
:tags: [remove-cell]

if YEAR == YEAR_IST:
    YEAR_IST = f'{YEAR_IST} (Ref)'
```

+++ {"tags": ["remove-cell"]}

## Analysis

+++ {"tags": ["remove-cell"]}

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

+++ {"tags": ["remove-cell"]}

The difference could be explained by job seekers.

+++

## Level of Employment Distribution

```{code-cell} ipython3
:tags: [remove-input]

_ = visualisations.plot_level_of_employment_distribution(persons, persons_ist, YEAR, YEAR_IST)
```

```{code-cell} ipython3
:tags: [remove-cell]

loe_by_age_group = marginal_fitting.comparison_table_categories(
    persons, persons_ist, YEAR, YEAR_IST, 'loe', 'age_group')
```

```{code-cell} ipython3
:tags: [remove-cell]

column_toggles = {c: [f'{c} {YEAR}', f'{c} {YEAR_IST}', f'AbsGrowth {c}', f'RelGrowth {c}'] for c in loe_labels}
hidden_cols = [c for sl in column_toggles.values() for c in sl if '100pct' not in c]
```

```{code-cell} ipython3
:tags: [remove-input]

visualisations.generate_simple_grid_table(
    loe_by_age_group.reset_index(), header_height=60,
    col_widths={'AgeGroup': 150}, colnames={'age_group': 'AgeGroup'},
    column_toggles=column_toggles, hidden_cols=hidden_cols)
```

## Employment Per Age

```{note}
The comparisons to BFS data presented here refer to the ["Szenarien zur Entwicklung der Erwerbsbevölkerung 2020-2050"](https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken/daten.assetdetail.12947685.html), at its reference scenario.
```

+++ {"tags": ["remove-cell"]}

#### IST

```{code-cell} ipython3
:tags: [remove-input]

title = 'Active People - SynPop{year} vs. BFS-Predictions for {year}*'.format(year=ref_year)
ax = visualisations.plot_people_employed_by_age(
    persons_ist, active_pop_stats_by_age_ist['active_people'], title=title)
plt.figtext(0.9, 0.01, 'BFS-Predictions data start in 2019.', horizontalalignment='right', fontsize=10)
glue("active_age_reference", ax.get_figure(), display=False)
```

+++ {"tags": ["remove-cell"]}

#### Scenario-Year

```{code-cell} ipython3
:tags: [remove-input]

title = 'Active People - SynPop{year} vs. BFS-Predictions for {year}'.format(year=YEAR)
ax = visualisations.plot_people_employed_by_age(
    persons, active_pop_stats_by_age['active_people'], title=title)
glue("active_age_projection", ax.get_figure(), display=False)
```

## Average Level of Employment per Age

```{code-cell} ipython3
:tags: [remove-input]

title = f'Average FTE SynPop {YEAR_IST} vs BFS'
ax = visualisations.plot_synpop_vs_bfs_avg_fte_per_age(
    persons_ist, active_pop_stats_by_age_ist['avg_fte_per_person'], YEAR_IST, title=title)
```

```{code-cell} ipython3
:tags: [remove-input]

title = f'Average FTE SynPop {YEAR} vs BFS'

ax = visualisations.plot_synpop_vs_bfs_avg_fte_per_age(
    persons, active_pop_stats_by_age['avg_fte_per_person'], YEAR, title=title)
```
