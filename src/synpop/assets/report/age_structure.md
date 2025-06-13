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

# Age Structure

+++ {"tags": ["remove-cell"]}

* **Input**: cleaned and optimized pickeled DataFrames (persons.csv) for 20XX, BSF age predictions
* **Output**: Visualisations

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

from synpop.marginals import PopPredictionsClient
import synpop.utils as utils
from synpop.synpop_tables import Persons
from synpop.visualisations import generate_simple_grid_table
```

+++ {"tags": ["remove-cell"]}

## Settings

```{code-cell} ipython3
:tags: [remove-cell]

# not displayed
YEAR_IST = config['reference_year']
YEAR = config['target_year']

SYNPOP_PATH_IST = config['reference_synpop']

SYNPOP_PATH = config['target_synpop']


if YEAR == YEAR_IST:
    YEAR_IST = f'{YEAR_IST} (Ref)'
```

+++ {"tags": ["remove-cell"]}

## Loading Data

```{code-cell} ipython3
:tags: [remove-cell]

synpop_persons = Persons(YEAR)
synpop_persons.load(SYNPOP_PATH, config["validate"])

persons = synpop_persons.data
```

```{code-cell} ipython3
:tags: [remove-cell]

synpop_persons_ist = Persons(YEAR_IST)
synpop_persons_ist.load(SYNPOP_PATH_IST, config["validate"])

persons_ist = synpop_persons_ist.data
```

```{code-cell} ipython3
:tags: [remove-cell]

persons['age_group'] = utils.bin_variable(persons['age'], interval_size=5, max_regular_intervals=100, last_bin_name='100-120')
persons_ist['age_group'] = utils.bin_variable(persons_ist['age'], interval_size=5, max_regular_intervals=100, last_bin_name='100-120')
```

```{code-cell} ipython3
:tags: [remove-cell]

synpop_stats = (persons.groupby('age_group').size()
                       .rename(f'SynPop {YEAR}')
                      )
synpop_stats_ist = (persons_ist.groupby('age_group').size()
                       .rename(f'SynPop {YEAR_IST}')
                      )
```

+++ {"tags": ["remove-cell"]}

### FSO Predictions (only permanent residents)

```{code-cell} ipython3
:tags: [remove-cell]

fso_client = PopPredictionsClient().load(year=YEAR)
```

```{code-cell} ipython3
:tags: [remove-cell]

bfs_stats = fso_client.pop_by_age_group.set_index('age_group')['pop'].rename(f'BFS {YEAR}')
```

+++ {"tags": ["remove-cell"]}

## Analysis

```{code-cell} ipython3
:tags: [remove-cell]

pop_per_age_group = (pd.concat([synpop_stats_ist, synpop_stats], axis=1)
                     .fillna(0)
                     .astype(int)
                     .loc[synpop_stats.index]  # Order age groups correctly
                    )
pop_per_age_group['delta_abs'] = (pop_per_age_group[f'SynPop {YEAR}'] - pop_per_age_group[f'SynPop {YEAR_IST}'])
pop_per_age_group['delta_pc'] = (pop_per_age_group['delta_abs'] / pop_per_age_group[f'SynPop {YEAR_IST}'] * 100).round(1)
```

```{code-cell} ipython3
:tags: [remove-cell]

ax = pop_per_age_group[[f'SynPop {YEAR_IST}', f'SynPop {YEAR}']].plot.bar(stacked=False, rot=45, color=['gray', 'navy'], figsize=(12, 5))
_ = ax.legend(loc='best', bbox_to_anchor=(1.1, 1))
_ = ax.set_ylabel('Population')
ax.grid(axis='y')

title = 'SynPop {} vs SynPop {}'.format(YEAR, YEAR_IST)
_ = ax.set_title(title, pad=25, fontdict={'fontsize':18, 'fontweight':'bold'})
glue("CH", ax.get_figure(), display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

def get_kt_pyramid(name, bfs=False):
    synpop_stats = (persons.query(f'KT_full == "{name}"').groupby('age_group').size().rename(f'SynPop {YEAR}'))
    bfs_stats = fso_client.pop_by_canton_and_age_group.query(f'KT_full == "{name}"').groupby('age_group')['pop'].sum().rename(f'BFS {YEAR}')

    col = f'SynPop {YEAR_IST}'
    color = ['gray', 'navy']
    title = 'SynPop {} vs SynPop {}'.format(YEAR, YEAR_IST)
    synpop_stats_ist = (persons_ist.query(f'KT_full == "{name}"').groupby('age_group').size().rename(f'SynPop {YEAR_IST}'))
    pop_per_age_group = (pd.concat([synpop_stats, synpop_stats_ist], axis=1).fillna(0).astype(int).loc[synpop_stats.index])  # Order age groups correctly

    if bfs:
        col = f'BFS {YEAR}'
        color = ['green', 'navy']
        title = '{} BFS (perm. only) vs. SynPop{}'.format(name, YEAR)
        pop_per_age_group = (pd.concat([bfs_stats, synpop_stats], axis=1).fillna(0).astype(int).loc[synpop_stats.index])  # Order age groups correctly

    ax = pop_per_age_group[[col, f'SynPop {YEAR}']].plot.bar(stacked=False, rot=45, color=color, figsize=(12, 5))
    _ = ax.legend(loc='best', bbox_to_anchor=(1.1, 1))
    _ = ax.set_ylabel('Population')
    ax.grid(axis='y')


    _ = ax.set_title(title, pad=25, fontdict={'fontsize':18, 'fontweight':'bold'})

    return pop_per_age_group, ax
```

```{code-cell} ipython3
:tags: [remove-cell]

df, ax = get_kt_pyramid('Zürich')
glue("Zürich", ax.get_figure(), display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

df, ax = get_kt_pyramid('Vaud')
glue("Vaud", ax.get_figure(), display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

df, ax = get_kt_pyramid('Ticino')
glue("Ticino", ax.get_figure(), display=False)
```

````{tabbed} Switzerland
```{glue:figure} CH
:figwidth: 800px
```
````

````{tabbed} Zürich
```{glue:figure} Zürich
:figwidth: 800px
```
````

````{tabbed} Vaud
```{glue:figure} Vaud
:figwidth: 800px
```
````

````{tabbed} Ticino
```{glue:figure} Ticino
:figwidth: 800px
```
````

<div style="text-align: right; font-weight: bold"> Detailed results per age group </div>

```{code-cell} ipython3
:tags: [remove-input, hide-output]

colnames = ['AgeGroup', f'Synpop {YEAR_IST}', f'Synpop {YEAR}', 'AbsGrowth', 'RelGrowth']
generate_simple_grid_table(pop_per_age_group.reset_index(), colnames)
```

## Comparison to BFS

```{note}
The comparisons to BFS data presented here refer to the ["Kantonale Bevölkerungsszenarien 2020-2050"](https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken.gnpdetail.2020-0194.html), at its reference scenario.
```

```{code-cell} ipython3
:tags: [remove-cell]

pop_per_age_group = (pd.concat([bfs_stats, synpop_stats], axis=1)
                     .fillna(0)
                     .astype(int)
                     .loc[bfs_stats.index]  # Order age groups correctly
                    )
pop_per_age_group['delta_abs'] = (pop_per_age_group[f'SynPop {YEAR}'] - pop_per_age_group[f'BFS {YEAR}'])
pop_per_age_group['delta_pc'] = (pop_per_age_group['delta_abs'] / pop_per_age_group[f'BFS {YEAR}'] * 100).round(1)
```

```{code-cell} ipython3
:tags: [remove-cell]

ax = pop_per_age_group[[f'BFS {YEAR}', f'SynPop {YEAR}']].plot.bar(stacked=False, rot=45, color=['green', 'navy'], figsize=(12, 5))
_ = ax.legend(loc='best', bbox_to_anchor=(1.1, 1))
_ = ax.set_ylabel('Population')
ax.grid(axis='y')

title = 'BFS (perm. only) vs. SynPop{}'.format(YEAR)
_ = ax.set_title(title, pad=25, fontdict={'fontsize':18, 'fontweight':'bold'})
glue("CH_bfs", ax.get_figure(), display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

df, ax = get_kt_pyramid('Zürich', True)
glue("Zürich_bfs", ax.get_figure(), display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

df, ax = get_kt_pyramid('Vaud', True)
glue("Vaud_bfs", ax.get_figure(), display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

df, ax = get_kt_pyramid('Ticino', True)
glue("Ticino_bfs", ax.get_figure(), display=False)
```

````{tabbed} Switzerland
```{glue:figure} CH_bfs
:figwidth: 800px
```
````

````{tabbed} Zürich
```{glue:figure} Zürich_bfs
:figwidth: 800px
```
````

````{tabbed} Vaud
```{glue:figure} Vaud_bfs
:figwidth: 800px
```
````

````{tabbed} Ticino
```{glue:figure} Ticino_bfs
:figwidth: 800px
```
````

<div style="text-align: right; font-weight: bold"> Detailed results per age group </div>

```{code-cell} ipython3
:tags: [remove-input, hide-output]

colnames = ['AgeGroup', f'BFS {YEAR}', f'Synpop {YEAR}', 'AbsGrowth', 'RelGrowth']
generate_simple_grid_table(pop_per_age_group.reset_index(), colnames)
```
