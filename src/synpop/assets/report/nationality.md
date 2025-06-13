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

# Nationality

:::{note}
The comparisons to BFS data presented here refer to the ["Kantonale Bevölkerungsszenarien 2020-2050"](https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken.gnpdetail.2020-0194.html), at its reference scenario.
:::

+++ {"tags": ["remove-cell"]}

* *nation* is fixed to match the BFS predictions per age canton and age
* The choice of persons to fix is done randomly

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

sys.path.insert(1, config['codebase'])

from synpop.visualisations import save_figure, plot_binary_feature_per_age_with_marginals, plot_multi_class_feature_per_age
from synpop import marginal_fitting, utils
from synpop.synpop_tables import Persons
from synpop.marginals import PopPredictionsClient
```

+++ {"tags": ["remove-cell"], "toc": true}

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Settings" data-toc-modified-id="Settings-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Settings</a></span></li><li><span><a href="#Loading-Data" data-toc-modified-id="Loading-Data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Loading Data</a></span><ul class="toc-item"><li><span><a href="#SynPop" data-toc-modified-id="SynPop-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>SynPop</a></span><ul class="toc-item"><li><span><a href="#Scenario-Year" data-toc-modified-id="Scenario-Year-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>Scenario-Year</a></span></li></ul></li><li><span><a href="#FSO-Data" data-toc-modified-id="FSO-Data-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>FSO Data</a></span></li></ul></li><li><span><a href="#Examining-SynPop-raw" data-toc-modified-id="Examining-SynPop-raw-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Examining SynPop-raw</a></span><ul class="toc-item"><li><span><a href="#Global-Counts" data-toc-modified-id="Global-Counts-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Global Counts</a></span></li><li><span><a href="#Marginals-Per-Age" data-toc-modified-id="Marginals-Per-Age-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Marginals Per Age</a></span><ul class="toc-item"><li><span><a href="#Global" data-toc-modified-id="Global-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>Global</a></span></li><li><span><a href="#A-few-examples" data-toc-modified-id="A-few-examples-3.2.2"><span class="toc-item-num">3.2.2&nbsp;&nbsp;</span>A few examples</a></span></li></ul></li></ul></li><li><span><a href="#Fixing-methodology" data-toc-modified-id="Fixing-methodology-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Fixing methodology</a></span><ul class="toc-item"><li><span><a href="#Expected-Counts" data-toc-modified-id="Expected-Counts-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Expected Counts</a></span></li><li><span><a href="#Fixing..." data-toc-modified-id="Fixing...-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Fixing...</a></span></li></ul></li><li><span><a href="#Examining-SynPop+" data-toc-modified-id="Examining-SynPop+-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Examining SynPop+</a></span><ul class="toc-item"><li><span><a href="#Global-Counts" data-toc-modified-id="Global-Counts-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Global Counts</a></span></li><li><span><a href="#Marginals-Per-Age" data-toc-modified-id="Marginals-Per-Age-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Marginals Per Age</a></span><ul class="toc-item"><li><span><a href="#Global" data-toc-modified-id="Global-5.2.1"><span class="toc-item-num">5.2.1&nbsp;&nbsp;</span>Global</a></span></li><li><span><a href="#A-few-examples" data-toc-modified-id="A-few-examples-5.2.2"><span class="toc-item-num">5.2.2&nbsp;&nbsp;</span>A few examples</a></span></li></ul></li></ul></li><li><span><a href="#Saving-Results" data-toc-modified-id="Saving-Results-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Saving Results</a></span></li><li><span><a href="#Export-Notebook-to-HTML" data-toc-modified-id="Export-Notebook-to-HTML-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Export Notebook to HTML</a></span></li></ul></div>

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

COLUMNS_OF_INTEREST = ['person_id', 'language', 'nation', 'age', 'zone_id', 'kt_id', 'kt_name', 'mun_name', 'KT_full' ]
```

+++ {"tags": ["remove-cell"]}

#### Scenario-Year

```{code-cell} ipython3
:tags: [remove-cell]

%%time
synpop_persons = Persons(YEAR)
synpop_persons.load(SYNPOP_PATH, config["validate"])

persons = synpop_persons.data
```

```{code-cell} ipython3
:tags: [remove-cell]

%%time
synpop_persons_ist = Persons(YEAR_IST)
synpop_persons_ist.load(SYNPOP_PATH_IST, config["validate"])

persons_ist = synpop_persons_ist.data
```

```{code-cell} ipython3
:tags: [remove-cell]

if YEAR == YEAR_IST:
    YEAR_IST = f'{YEAR_IST} (Ref)'
```

```{code-cell} ipython3
:tags: [remove-cell]

print('Imported data:')
print('persons DataFrame for {}: {}'.format(YEAR, persons.shape))
```

+++ {"tags": ["remove-cell"]}

### FSO Data

```{code-cell} ipython3
:tags: [remove-cell]

%%time
bfs_pred = PopPredictionsClient().load(year=YEAR).pop_by_canton_age_and_nationality
```

```{code-cell} ipython3
:tags: [remove-cell]

bfs_counts = bfs_pred.set_index(['KT_full', 'age', 'is_swiss'])['pop']
bfs_totals_per_cat = bfs_pred.groupby(['KT_full', 'age']).sum()['pop']
bfs_predicted_ratio = (bfs_counts / bfs_totals_per_cat).dropna()
```

+++ {"tags": ["remove-cell"]}

## Examining SynPop-raw

```{code-cell} ipython3
:tags: [remove-cell]

COLOUR_DICT = {'true': 'firebrick', 'false': 'grey'}
```

## Global Counts

```{code-cell} ipython3
:tags: [remove-input]

synpop_counts = (persons.groupby('is_swiss').size()).loc[True]
synpop_rel = (synpop_counts / persons.shape[0])

synpop_counts_ist = (persons_ist.groupby('is_swiss').size()).loc[True]
synpop_rel_ist = (synpop_counts_ist / persons_ist.shape[0])

global_stats = pd.DataFrame({'Population': {f'SynPop {YEAR}': persons.shape[0],
                                            f'SynPop {YEAR_IST}': persons_ist.shape[0], },
                             'Swiss':   {f'SynPop {YEAR}': synpop_counts,
                                         f'SynPop {YEAR_IST}': synpop_counts_ist},
                             '% Swiss': {f'SynPop {YEAR}': round(synpop_rel * 100, 1),
                                         f'SynPop {YEAR_IST}': round(synpop_rel_ist * 100, 1)}
                            }
                           )
# print(YEAR)
global_stats
```

## Nationality Per Age

+++ {"tags": ["remove-cell"]}

#### Global

```{code-cell} ipython3
:tags: [remove-cell]

is_swiss_counts = persons.groupby('age').agg({'is_swiss': ['sum', lambda x: sum(~x)]})
is_swiss_counts = is_swiss_counts.droplevel(0, axis=1)
is_swiss_counts.columns = ['true', 'false']
```

```{code-cell} ipython3
:tags: [remove-input]

title = 'Nationality by Age - Global Population - SynPop {}'.format(YEAR)

ax = plot_multi_class_feature_per_age(is_swiss_counts, colour_dict=COLOUR_DICT, ymax=150_000, title=title)

glue('global', ax.get_figure(), display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

is_swiss_counts_ist = persons_ist.groupby('age').agg({'is_swiss': ['sum', lambda x: sum(~x)]})
is_swiss_counts_ist = is_swiss_counts_ist.droplevel(0, axis=1)
is_swiss_counts_ist.columns = ['true', 'false']
```

```{code-cell} ipython3
:tags: [remove-input]

title = 'Nationality by Age - Global Population - SynPop {}'.format(YEAR_IST)

ax = plot_multi_class_feature_per_age(is_swiss_counts_ist,
                                      colour_dict=COLOUR_DICT, ymax=150_000, title=title)

glue('global', ax.get_figure(), display=False)
```

## Comparison to BFS

```{code-cell} ipython3
:tags: [remove-input]

synpop_counts = (persons.groupby('is_swiss').size()).loc[True]
synpop_rel = (synpop_counts / persons.shape[0])

bfs_counts = (bfs_pred.groupby('is_swiss')['pop'].sum()).loc[True]
bfs_rel = bfs_counts / bfs_pred['pop'].sum()

global_stats = pd.DataFrame({'Population': {'SynPop': persons.shape[0],
                                         'BFS': bfs_pred['pop'].sum()},
                             'Swiss':   {'SynPop': synpop_counts,
                                         'BFS': bfs_counts},
                             '% Swiss': {'SynPop': round(synpop_rel * 100, 1),
                                         'BFS': round(bfs_rel * 100, 1)}
                            }
                           )
# print(YEAR)
global_stats
```

## Nationality Per Age

```{code-cell} ipython3
:tags: [remove-cell]

%%time
marginals = marginal_fitting.compute_bfs_vs_scenario_marginal_summary_table(persons,
                                                                              bfs_predicted_ratio,
                                                                              YEAR,
                                                                              feature='is_swiss',
                                                                              control_level_list=['KT_full', 'age']
                                                                             )
```

```{code-cell} ipython3
:tags: [remove-cell]

counts_per_cat = (marginals
                  .reset_index()
                  .astype({'is_swiss': str})  # to avoid boolean values as column names
                  .pivot_table(index=['KT_full', 'age'], columns='is_swiss', values='counts_{}'.format(YEAR))
                  .fillna(0)
                  .astype(int)
                  .rename(columns={'True': 'true', 'False': 'false'})  # to avoid confusion between bool and str
                 )
counts_per_cat = counts_per_cat[['true', 'false']]
```

```{code-cell} ipython3
:tags: [remove-cell]

expected_true_counts = (marginals['expected_counts_{}'.format(YEAR)]
                        .reset_index()
                        .query('is_swiss')
                        .groupby(['KT_full', 'age']).sum()['expected_counts_{}'.format(YEAR)]
                        .rename('bfs: {} prediction'.format('is_swiss'))
                       )
```

+++ {"tags": ["remove-cell"]}

#### Global

```{code-cell} ipython3
:tags: [remove-cell]

title = 'Nationality by Age - Global Population - SynPop {}'.format(YEAR)

ax = plot_binary_feature_per_age_with_marginals(counts_per_cat.groupby(level=1).sum(),
                                                expected_true_counts.groupby(level=1).sum(),
                                                colour_dict=COLOUR_DICT, ymax=150_000, title=title)

glue('global', ax.get_figure(), display=False)
```

+++ {"tags": ["remove-cell"]}

#### A few examples

+++ {"tags": ["remove-cell"]}

*Zürich*

```{code-cell} ipython3
:tags: [remove-cell]

CANTON = 'Zürich'
```

```{code-cell} ipython3
:tags: [remove-cell]

marginals.loc[CANTON]
```

```{code-cell} ipython3
:tags: [remove-cell]

title = 'Nationality by Age - Canton {} - SynPop {}'.format(CANTON, YEAR)

ax = plot_binary_feature_per_age_with_marginals(counts_per_cat.loc[CANTON],
                                                expected_true_counts.loc[CANTON],
                                                colour_dict=COLOUR_DICT, ymax=27_000, title=title)

glue('zh_nat', ax.get_figure(), display=False)
plt.close()
```

+++ {"tags": ["remove-cell"]}

*Vaud*

```{code-cell} ipython3
:tags: [remove-cell]

CANTON = 'Vaud'
```

```{code-cell} ipython3
:tags: [remove-cell]

marginals.loc[CANTON]
```

```{code-cell} ipython3
:tags: [remove-cell]

title = 'Nationality by Age - Canton {} - SynPop {}'.format(CANTON, YEAR)

ax = plot_binary_feature_per_age_with_marginals(counts_per_cat.loc[CANTON],
                                                expected_true_counts.loc[CANTON],
                                                colour_dict=COLOUR_DICT, ymax=27_000, title=title)

glue('vd_nat', ax.get_figure(), display=False)
plt.close()
```

+++ {"tags": ["remove-cell"]}

*Ticino*

```{code-cell} ipython3
:tags: [remove-cell]

CANTON = 'Ticino'
```

```{code-cell} ipython3
:tags: [remove-cell]

marginals.loc[CANTON]
```

```{code-cell} ipython3
:tags: [remove-cell]

title = 'Nationality by Age - Canton {} - SynPop {}'.format(CANTON, YEAR)

ax = plot_binary_feature_per_age_with_marginals(counts_per_cat.loc[CANTON],
                                                expected_true_counts.loc[CANTON],
                                                colour_dict=COLOUR_DICT, ymax=6_500, title=title)

glue('ti_nat', ax.get_figure(), display=False)
plt.close()
```

````{tabbed} Switzerland
```{glue:figure} global
:figwidth: 800px
```
````

````{tabbed} Zürich
```{glue:figure} zh_nat
:figwidth: 800px
```
````

````{tabbed} Vaud
```{glue:figure} vd_nat
:figwidth: 800px
```
````

````{tabbed} Ticino
```{glue:figure} ti_nat
:figwidth: 800px
```
````
