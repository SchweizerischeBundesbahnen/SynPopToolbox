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

# Population Counts

```{code-cell} ipython3
:tags: [remove-cell]

# not displayed
import logging
import math
import sys
import geopandas as gpd
import pandas as pd
import numpy as np

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

sys.path.insert(1, config['codebase'])

from synpop.marginals import PopPredictionsClient
from synpop.zone_maps import SwissZoneMap
from synpop.visualisations import generate_simple_grid_table, altair_scatter
import synpop.utils as utils
from synpop.synpop_tables import Persons
```

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

+++ {"tags": ["remove-cell"]}

### BFS-Data

```{code-cell} ipython3
:tags: [remove-cell]

bfs_pop_pred = PopPredictionsClient().load(year=YEAR)
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
print('Global population SynPop Ref: {}'.format(len(persons_ist)))
print('Global population SynPop: {}'.format(len(persons)))

delta = len(persons) - bfs_pop_pred.pop_total
growth = len(persons) - len(persons_ist)
print('Delta: {:.0f} ({:.1f}%)'.format(delta, delta / bfs_pop_pred.pop_total * 100))
print('Growth: {:.0f} ({:.1f}%)'.format(growth, growth / len(persons) * 100))
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("bfs_global", bfs_pop_pred.pop_total, display=False)
glue("synpop_global", len(persons), display=False)
glue("synpop_ref_global", len(persons_ist), display=False)
glue("pop_growth", growth, display=False)
glue("pop_growth_pct", '{:.1f}%'.format((growth / len(persons_ist)) * 100), display=False)
glue("delta_global", delta, display=False)
glue("delta_global_pct", '{:.1f}%'.format((delta / bfs_pop_pred.pop_total) * 100), display=False)
```

````{panels}
:column: col-4
SynPop {glue:text}`year`
^^^
{glue:text}`synpop_global`
---
SynPop {glue:text}`year_ist`
^^^
{glue:text}`synpop_ref_global`
---
Growth
^^^
{glue:text}`pop_growth` ({glue:text}`pop_growth_pct`)
````

## By Canton

```{code-cell} ipython3
:tags: [remove-cell]

synpop_by_canton = persons.groupby('KT_full').size().rename(f'SynPop {YEAR}')
```

```{code-cell} ipython3
:tags: [remove-cell]

synpop_by_canton_ist = persons_ist.groupby('KT_full').size().rename(f'SynPop {YEAR_IST}')
```

```{code-cell} ipython3
:tags: [remove-cell]

pop_by_canton = pd.concat([synpop_by_canton, synpop_by_canton_ist], axis=1, sort=False)
pop_by_canton['delta_abs'] = (pop_by_canton[f'SynPop {YEAR}'] - pop_by_canton[f'SynPop {YEAR_IST}'])
pop_by_canton['delta_pc'] = (pop_by_canton['delta_abs'] / pop_by_canton[f'SynPop {YEAR_IST}'] * 100).round(1)
glue("pop_by_canton", pop_by_canton, display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

default_map_client = SwissZoneMap(outline_cantons=True)
```

```{code-cell} ipython3
:tags: [remove-cell]

def round_up(x, precision=1000.0):
    return int(math.ceil(x / precision) * precision)
```

```{code-cell} ipython3
:tags: [remove-cell]

title = 'SynPop {} vs. SynPop {}: Absolute Population Growth'.format(YEAR, YEAR_IST)
scale_bound = round_up(pop_by_canton['delta_abs'].abs().max())
ax1, _ = default_map_client.draw_cantons(pop_by_canton, 'delta_abs', vmin=-scale_bound, vmax=scale_bound, title=title)
```

```{code-cell} ipython3
:tags: [remove-cell]

title = 'SynPop {} vs. SynPop {}: % Population Growth'.format(YEAR, YEAR_IST)
scale_bound = round(pop_by_canton['delta_pc'].abs().max())
ax2, _ = default_map_client.draw_cantons(pop_by_canton, 'delta_pc', vmin=-scale_bound, vmax=scale_bound, title=title)
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("abs_diff", ax1.get_figure(), display=False)
glue("rel_diff", ax2.get_figure(), display=False)
plt.close()
```

````{tabbed} Absolute growth
```{glue:figure} abs_diff
:figwidth: 800px
```
````

````{tabbed} Relative growth
```{glue:figure} rel_diff
:figwidth: 800px
```
````

<div style="text-align: right; font-weight: bold"> Detailed results per Canton </div>

```{code-cell} ipython3
:tags: [remove-input, hide-output]

colnames = ['Canton', f'Synpop {YEAR}', f'Synpop {YEAR_IST}', 'AbsGrowth', 'RelGrowth']
generate_simple_grid_table(pop_by_canton.reset_index(), colnames)
```

<div style="text-align: right; font-weight: bold"> Detailed results per Municipality </div>

```{code-cell} ipython3
:tags: [remove-cell]

pop_by_mun = persons.groupby(['KT_full', 'mun_name']).size().rename(f'SynPop {YEAR}')
pop_by_mun_ist = persons_ist.groupby(['KT_full', 'mun_name']).size().rename(f'SynPop {YEAR_IST}')
pop_by_mun = pd.concat([pop_by_mun, pop_by_mun_ist], axis=1)
pop_by_mun['delta_abs'] = (pop_by_mun[f'SynPop {YEAR}'] - pop_by_mun[f'SynPop {YEAR_IST}'])
pop_by_mun['delta_pc'] = (pop_by_mun[f'SynPop {YEAR}'] / pop_by_mun[f'SynPop {YEAR_IST}'] * 100).round(1)
pop_by_mun = pop_by_mun.reset_index()
```

```{code-cell} ipython3
:tags: [remove-input, hide-output]

colnames = ['Canton', 'Commune', f'Synpop {YEAR}', f'Synpop {YEAR_IST}', 'AbsGrowth', 'RelGrowth']
generate_simple_grid_table(pop_by_mun, colnames)
```

## Growth Analysis

```{code-cell} ipython3
:tags: [remove-cell]

%%time
mun_stats = (pd.concat([persons_ist.groupby(['KT_full', 'mun_name']).size().rename(f'pop {YEAR_IST}'),
                        persons.groupby(['KT_full', 'mun_name']).size().rename(f'pop {YEAR}')
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

### Absolute Growth

```{code-cell} ipython3
:tags: [remove-input]

altair_scatter(mun_stats, f'pop {YEAR_IST}','growth abs', dropdown_col='KT_full', columns=mun_stats.columns.tolist()[1:])
```

### Relative Growth

```{code-cell} ipython3
:tags: [remove-input]

altair_scatter(mun_stats, f'pop {YEAR_IST}','growth factor', dropdown_col='KT_full', columns=mun_stats.columns.tolist()[1:])
```

## Comparison to BFS

:::{note}
The comparisons to BFS data presented here refer to the ["Kantonale Bevölkerungsszenarien 2020-2050"](https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken.gnpdetail.2020-0194.html), at its reference scenario.
:::

+++

````{panels}
:column: col-4
BFS {glue:text}`year`
^^^
{glue:text}`bfs_global`
---
SynPop {glue:text}`year`
^^^
{glue:text}`synpop_global`
---
Delta
^^^
{glue:text}`delta_global` ({glue:text}`delta_global_pct`)
````

```{code-cell} ipython3
:tags: [remove-cell]

bfs_pop = bfs_pop_pred.pop_by_canton.set_index('KT_full')['pop'].rename(f'BFS {YEAR}')
```

```{code-cell} ipython3
:tags: [remove-cell]

delta_by_canton = pd.concat([bfs_pop, synpop_by_canton], axis=1, sort=False)
delta_by_canton['delta_abs'] = (delta_by_canton[f'SynPop {YEAR}'] - delta_by_canton[f'BFS {YEAR}'])
delta_by_canton['delta_pc'] = (delta_by_canton['delta_abs'] / delta_by_canton[f'BFS {YEAR}'] * 100).round(1)
glue("delta_by_canton", delta_by_canton, display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

title = 'SynPop{} vs. BFS Reference-Scenario: Absolute Population Diff'.format(YEAR)
scale_bound = round_up(delta_by_canton['delta_abs'].abs().max())
ax1, _ = default_map_client.draw_cantons(delta_by_canton, 'delta_abs', vmin=-scale_bound, vmax=scale_bound, title=title)
```

```{code-cell} ipython3
:tags: [remove-cell]

title = 'SynPop{} vs. BFS Reference-Scenario: % Population Diff'.format(YEAR)
scale_bound = round(delta_by_canton['delta_pc'].abs().max())
ax2, _ = default_map_client.draw_cantons(delta_by_canton, 'delta_pc', vmin=-scale_bound, vmax=scale_bound, title=title)
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("abs_diff_bfs", ax1.get_figure(), display=False)
glue("rel_diff_bfs", ax2.get_figure(), display=False)
plt.close()
```

````{tabbed} Absolute differences
```{glue:figure} abs_diff_bfs
:figwidth: 800px
```
````

````{tabbed} Relative differences
```{glue:figure} rel_diff_bfs
:figwidth: 800px
```
````

<div style="text-align: right; font-weight: bold"> Detailed results per Canton </div>

```{code-cell} ipython3
:tags: [remove-input, hide-output]

colnames = ['Canton', f'BFS {YEAR}', f'Synpop {YEAR}', 'AbsDelta', 'RelDelta']
generate_simple_grid_table(delta_by_canton.reset_index(), colnames)
```
