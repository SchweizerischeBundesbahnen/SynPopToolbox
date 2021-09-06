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

# Age Structure

```{note}
The comparisons to BFS data presented here refer to the ["Kantonale Bevölkerungsszenarien 2020-2050"](https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken.gnpdetail.2020-0194.html), at its reference scenario.
```

```{important}
The statistics refer to the population at 31st of December (not 1st of January as mistakenly used in previous SynPop versions until the beginning of 2021). 
```

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

sys.path.append(config['codebase'])

from synpop.marginals import FSO_PopPredictionsClient
import synpop.utils as utils
from synpop.synpop_tables import Persons
```

+++ {"tags": ["remove-cell"]}

## Settings

```{code-cell} ipython3
:tags: [remove-cell]

# not displayed
YEAR = config['target_year']

DATA_DIR = config['target_synpop']
SYNPOP_PERSONS_FILE = os.path.join(DATA_DIR, 'persons_{}'.format(YEAR))

SAVE_FIGURES = False
```

+++ {"tags": ["remove-cell"]}

**Plot Export Settings**

+++ {"tags": ["remove-cell"]}

## Loading Data

+++ {"tags": ["remove-cell"]}

### SynPop

```{code-cell} ipython3
:tags: [remove-cell]

%%time
synpop_persons = Persons(YEAR)
synpop_persons.load(SYNPOP_PERSONS_FILE)

persons = synpop_persons.data 
persons.shape
```

```{code-cell} ipython3
:tags: [remove-cell]

persons['age_group'] = utils.bin_variable(persons['age'], interval_size=5, max_regular_intervals=150, last_bin_name='100-120')
```

```{code-cell} ipython3
:tags: [remove-cell]

synpop_stats = (persons.groupby('age_group')['person_id'].count()
                       .rename('SynPop')
                      )
```

+++ {"tags": ["remove-cell"]}

### FSO Predictions (only permanent residents)

```{code-cell} ipython3
:tags: [remove-cell]

fso_client = FSO_PopPredictionsClient().load(year=YEAR)
```

```{code-cell} ipython3
:tags: [remove-cell]

bfs_stats = fso_client.pop_by_age_group.set_index('age_group')['pop'].rename('BFS')
```

+++ {"tags": ["remove-cell"]}

## Analysis

+++ {"tags": ["remove-cell"]}

### BLS - SynPop Comparison

```{code-cell} ipython3
:tags: [remove-cell]

pop_per_age_group = (pd.concat([bfs_stats, synpop_stats], axis=1)
                     .fillna(0)
                     .astype(int)
                     .loc[bfs_stats.index]  # Order age groups correctly 
                    )
pop_per_age_group
```

```{code-cell} ipython3
:tags: [remove-cell]

ax = pop_per_age_group.plot.bar(stacked=False, rot=45, color=['green', 'navy'], figsize=(12, 5))
_ = ax.legend(loc='best', bbox_to_anchor=(1.1, 1))
_ = ax.set_ylabel('Population')
ax.grid(axis='y')

title = 'BFS (perm. only) vs. SynPop{}'.format(YEAR)
_ = ax.set_title(title, pad=25, fontdict={'fontsize':18, 'fontweight':'bold'})
glue("CH", ax.get_figure(), display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

def get_kt_pyramid(name):
    synpop_stats = (persons.query(f'KT_full == "{name}"').groupby('age_group')['person_id'].count().rename('SynPop'))
    bfs_stats = fso_client.pop_by_canton_and_age_group.query(f'KT_full == "{name}"').groupby('age_group')['pop'].sum().rename('BFS')
    pop_per_age_group = (pd.concat([bfs_stats, synpop_stats], axis=1).fillna(0).astype(int).loc[bfs_stats.index])  # Order age groups correctly 

    ax = pop_per_age_group.plot.bar(stacked=False, rot=45, color=['green', 'navy'], figsize=(12, 5))
    _ = ax.legend(loc='best', bbox_to_anchor=(1.1, 1))
    _ = ax.set_ylabel('Population')
    ax.grid(axis='y')

    title = '{} BFS (perm. only) vs. SynPop{}'.format(name, YEAR)
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
