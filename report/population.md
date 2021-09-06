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

# Population Counts

:::{note}
The comparisons to BFS data presented here refer to the ["Kantonale Bevölkerungsszenarien 2020-2050"](https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken.gnpdetail.2020-0194.html), at its reference scenario.
:::

```{code-cell} ipython3
:tags: [remove-cell]

# not displayed
import logging
import math
import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import altair as alt

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

from synpop.marginals import FSO_PopPredictionsClient
from synpop.zone_maps import SwissZoneMap
from synpop.visualisations import generate_simple_grid_table
import synpop.utils as utils
from synpop.synpop_tables import Persons
```

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

if YEAR == YEAR_IST:
    YEAR_IST = f'{YEAR_IST} (Ref)'
```

+++ {"tags": ["remove-cell"]}

## Loading Data

```{code-cell} ipython3
:tags: [remove-cell]

synpop_persons = Persons(YEAR)
synpop_persons.load(SYNPOP_PERSONS_FILE)

persons = synpop_persons.data 
```

```{code-cell} ipython3
:tags: [remove-cell]

synpop_persons_ist = Persons(YEAR_IST)
synpop_persons_ist.load(SYNPOP_PERSONS_FILE_IST)

persons_ist = synpop_persons_ist.data
```

+++ {"tags": ["remove-cell"]}

### BFS-Data

```{code-cell} ipython3
:tags: [remove-cell]

bfs_pop_pred = FSO_PopPredictionsClient().load(year=YEAR)
```

+++ {"tags": ["remove-cell"]}

# Analysis

+++

## Global Population

+++ {"tags": ["remove-cell"]}

### Whole of Switerland

```{code-cell} ipython3
:tags: [remove-cell]

print('Global population BFS (only permanent residents): {}'.format(bfs_pop_pred.pop_total))
print('Global population SynPop: {}'.format(len(persons)))

delta = len(persons) - bfs_pop_pred.pop_total
print('Delta: {:.0f} ({:.1f}%)'.format(delta, delta / bfs_pop_pred.pop_total * 100))
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("bfs_global", bfs_pop_pred.pop_total, display=False)
glue("synpop_global", len(persons), display=False)
glue("delta_global", delta, display=False)
glue("delta_global_pct", '{:.1f}%'.format(delta / bfs_pop_pred.pop_total * 100), display=False)
```

````{panels}
:column: col-4
BFS
^^^
{glue:text}`bfs_global`
---
SynPop
^^^
{glue:text}`synpop_global`
---
Delta
^^^
{glue:text}`delta_global` ({glue:text}`delta_global_pct`)
````

## By Canton

```{code-cell} ipython3
:tags: [remove-cell]

synpop_by_canton = persons.groupby('KT_full')['person_id'].count().rename('SynPop_pop')
```

```{code-cell} ipython3
:tags: [remove-cell]

bfs_pop = bfs_pop_pred.pop_by_canton.set_index('KT_full')['pop'].rename('BFS_pop')
```

```{code-cell} ipython3
:tags: [remove-cell]

pop_by_canton = pd.concat([bfs_pop, synpop_by_canton], axis=1, sort=False)
pop_by_canton['delta_abs'] = (pop_by_canton['SynPop_pop'] - pop_by_canton['BFS_pop'])
pop_by_canton['delta_pc'] = (pop_by_canton['delta_abs'] / pop_by_canton['BFS_pop'] * 100).round(1)
glue("pop_by_canton", pop_by_canton, display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

default_map_client = SwissZoneMap(outline_cantons=True)
```

```{code-cell} ipython3
:tags: [remove-cell]

def round_up(x):
    return int(math.ceil(x / 1000.0)) * 1000
```

```{code-cell} ipython3
:tags: [remove-cell]

title = 'SynPop{} vs. BFS Reference-Scenario: Absolute Population Diff'.format(YEAR)
scale_bound = round_up(pop_by_canton['delta_abs'].abs().max())
ax1, _ = default_map_client.draw_cantons(pop_by_canton, 'delta_abs', vmin=-scale_bound, vmax=scale_bound, title=title)
```

```{code-cell} ipython3
:tags: [remove-cell]

title = 'SynPop{} vs. BFS Reference-Scenario: % Population Diff'.format(YEAR)
scale_bound = round(pop_by_canton['delta_pc'].abs().max())
ax2, _ = default_map_client.draw_cantons(pop_by_canton, 'delta_pc', vmin=-scale_bound, vmax=scale_bound, title=title)
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("abs_diff", ax1.get_figure(), display=False)
glue("rel_diff", ax2.get_figure(), display=False)
plt.close()
```

````{tabbed} Absolute differences
```{glue:figure} abs_diff
:figwidth: 800px
```
````

````{tabbed} Relative differences
```{glue:figure} rel_diff
:figwidth: 800px
```
````

````{tabbed} Table of results
```{glue:figure} pop_by_canton
:figwidth: 800px
```
````

<div style="text-align: right; font-weight: bold"> Detailed results per Canton </div>

```{code-cell} ipython3
:tags: [remove-input, hide-output]

generate_simple_grid_table(pop_by_canton)
```

<div style="text-align: right; font-weight: bold"> Detailed results per Municipality </div>

```{code-cell} ipython3
:tags: [remove-cell]

pop_by_mun = persons.groupby(['KT_full', 'mun_name'])['person_id'].count().rename(f'SynPop {YEAR}')
pop_by_mun_ist = persons_ist.groupby(['KT_full', 'mun_name'])['person_id'].count().rename(f'SynPop {YEAR_IST}')
pop_by_mun = pd.concat([pop_by_mun, pop_by_mun_ist], axis=1)
pop_by_mun['delta_abs'] = (pop_by_mun[f'SynPop {YEAR}'] - pop_by_mun[f'SynPop {YEAR_IST}'])
pop_by_mun['delta_pc'] = (pop_by_mun[f'SynPop {YEAR}'] / pop_by_mun[f'SynPop {YEAR_IST}'] * 100).round(1)
pop_by_mun = pop_by_mun.reset_index()
```

```{code-cell} ipython3
:tags: [remove-input, hide-output]

generate_simple_grid_table(pop_by_mun)
```

## Growth Analysis

```{code-cell} ipython3
:tags: [remove-cell]

%%time
mun_stats = (pd.concat([persons_ist.groupby('mun_name').count().iloc[:, 0].rename(f'pop {YEAR_IST}'),
                        persons.groupby('mun_name').count().iloc[:, 0].rename(f'pop {YEAR}')
                        ], axis=1, join='outer')
             .fillna(0)
             .astype(int)
            )
```

```{code-cell} ipython3
:tags: [remove-cell]

mun_stats['growth abs'] = mun_stats[f'pop {YEAR}'] - mun_stats[f'pop {YEAR_IST}']
mun_stats['growth factor'] = mun_stats[f'pop {YEAR}'] / mun_stats[f'pop {YEAR_IST}']
mun_stats['growth factor'] = mun_stats['growth factor'].replace(np.inf, np.nan)
mun_stats['growth factor'] = mun_stats['growth factor'].round(3)
mun_stats = mun_stats.sort_values(f'pop {YEAR_IST}', ascending=False)

mun_stats = mun_stats[[f'pop {YEAR_IST}', f'pop {YEAR}', 'growth abs', 'growth factor']].reset_index()
```

```{code-cell} ipython3
:tags: [remove-cell]

mun_stats
```

### Absolute Growth

```{code-cell} ipython3
:tags: [remove-input]

df = mun_stats
x_col = f'pop {YEAR_IST}'
y_col = 'growth abs' 
other_cols = [c for c in df.columns if c not in [x_col, y_col]]

# Brush for selection
brush = alt.selection(type='interval')

# Zoom interaction with ctrl-Key
interaction = alt.selection(
    type="interval",
    bind="scales",
    on="[mousedown[event.ctrlKey], mouseup] > mousemove",
    translate="[mousedown[event.ctrlKey], mouseup] > mousemove!",
    zoom="wheel![event.ctrlKey]",
)

# Scatter Plot
points = alt.Chart(df).mark_point().encode(
    x=f'{x_col}:Q',
    y=f'{y_col}:Q'
).add_selection(
    brush,
    interaction
)

# Base chart for data tables
ranked_text = alt.Chart(df).mark_text().encode(
    y=alt.Y('row_number:O',axis=None)
).transform_window(
    row_number='row_number()'
).transform_filter(
    brush
).transform_window(
    rank='rank(row_number)'
).transform_filter(
    alt.datum.rank<20
)

# Data Tables
cols = [ranked_text.encode(text=f'{c}:N').properties(title=c) for c in [x_col, y_col] + other_cols]
text = alt.hconcat(*cols) # Combine data tables

# Build chart
alt.hconcat(
    points,
    text
).resolve_legend(
    color="independent"
)
```

### Relative Growth

```{code-cell} ipython3
:tags: [remove-input]

df = mun_stats
x_col = f'pop {YEAR_IST}'
y_col = 'growth factor' 
other_cols = [f'pop {YEAR}', 'mun_name', 'growth abs']

# Brush for selection
brush = alt.selection(type='interval')

# Zoom interaction with ctrl-Key
interaction = alt.selection(
    type="interval",
    bind="scales",
    on="[mousedown[event.ctrlKey], mouseup] > mousemove",
    translate="[mousedown[event.ctrlKey], mouseup] > mousemove!",
    zoom="wheel![event.ctrlKey]",
)

# Scatter Plot
points = alt.Chart(df).mark_point().encode(
    x=f'{x_col}:Q',
    y=f'{y_col}:Q'
).add_selection(
    brush,
    interaction
)

# Base chart for data tables
ranked_text = alt.Chart(df).mark_text().encode(
    y=alt.Y('row_number:O',axis=None)
).transform_window(
    row_number='row_number()'
).transform_filter(
    brush
).transform_window(
    rank='rank(row_number)'
).transform_filter(
    alt.datum.rank<20
)

# Data Tables
cols = [ranked_text.encode(text=f'{c}:N').properties(title=c) for c in [x_col, y_col] + other_cols]
text = alt.hconcat(*cols) # Combine data tables

# Build chart
alt.hconcat(
    points,
    text
).resolve_legend(
    color="independent"
)
```

```{code-cell} ipython3

```
