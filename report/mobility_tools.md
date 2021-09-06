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

# Mobility Tools

+++ {"tags": ["remove-cell"]}

* **Input**: cleaned and optimized pickeled DataFrames (persons.csv) for 20XX and 2017
* **Output**: Visualisations

+++ {"toc": true, "tags": ["remove-cell"]}

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Settings" data-toc-modified-id="Settings-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Settings</a></span></li><li><span><a href="#Loading-Data" data-toc-modified-id="Loading-Data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Loading Data</a></span><ul class="toc-item"><li><span><a href="#SynPop" data-toc-modified-id="SynPop-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>SynPop</a></span></li></ul></li><li><span><a href="#Analysis" data-toc-modified-id="Analysis-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Analysis</a></span><ul class="toc-item"><li><span><a href="#Global-Counts" data-toc-modified-id="Global-Counts-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Global Counts</a></span><ul class="toc-item"><li><span><a href="#Mobility-Category" data-toc-modified-id="Mobility-Category-3.1.1"><span class="toc-item-num">3.1.1&nbsp;&nbsp;</span>Mobility Category</a></span></li><li><span><a href="#Avaiable-Mobility-Tools" data-toc-modified-id="Avaiable-Mobility-Tools-3.1.2"><span class="toc-item-num">3.1.2&nbsp;&nbsp;</span>Avaiable Mobility Tools</a></span></li></ul></li><li><span><a href="#Evolution-By-Canton" data-toc-modified-id="Evolution-By-Canton-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Evolution By Canton</a></span><ul class="toc-item"><li><span><a href="#Population-Evolution" data-toc-modified-id="Population-Evolution-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>Population Evolution</a></span></li><li><span><a href="#GA-Evolution" data-toc-modified-id="GA-Evolution-3.2.2"><span class="toc-item-num">3.2.2&nbsp;&nbsp;</span>GA Evolution</a></span></li><li><span><a href="#HT-Evolution" data-toc-modified-id="HT-Evolution-3.2.3"><span class="toc-item-num">3.2.3&nbsp;&nbsp;</span>HT Evolution</a></span></li><li><span><a href="#VA-Evolution" data-toc-modified-id="VA-Evolution-3.2.4"><span class="toc-item-num">3.2.4&nbsp;&nbsp;</span>VA Evolution</a></span></li><li><span><a href="#Cars-Evolution" data-toc-modified-id="Cars-Evolution-3.2.5"><span class="toc-item-num">3.2.5&nbsp;&nbsp;</span>Cars Evolution</a></span></li></ul></li></ul></li></ul></div>

```{code-cell} ipython3
:tags: [remove-cell]

import logging
import os
import sys
import math
import geopandas as gpd
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

import synpop.utils as utils
from synpop.visualisations import plot_multi_class_feature_per_age
from synpop import marginal_fitting
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

if YEAR == YEAR_IST:
    YEAR_IST = f'{YEAR_IST} (Ref)'
```

+++ {"tags": ["remove-cell"]}

## Loading Data

+++ {"tags": ["remove-cell"]}

### SynPop

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

persons = synpop_persons.data
```

```{code-cell} ipython3
:tags: [remove-cell]

persons.shape
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
```

```{code-cell} ipython3
:tags: [remove-cell]

persons_ist.shape
```

+++ {"tags": ["remove-cell"]}

## Analysis

+++ {"tags": ["remove-cell"]}

### Global Counts

+++ {"tags": ["remove-cell"]}

Since there is no `null` in 2040, change to `nothing`

```{code-cell} ipython3
:tags: [remove-cell]

persons_ist['mobility'] = persons_ist['mobility'].mask(persons_ist['mobility'] == 'null', 'nothing')
persons['mobility'] = persons['mobility'].mask(persons['mobility'] == 'null', 'nothing')
```

+++ 

## Mobility Tool Ownership

```{code-cell} ipython3
:tags: [remove-cell]

mob_counts_ist = (persons_ist.groupby('mobility')['person_id'].count()).rename('SynPop{}'.format(YEAR_IST))

mob_counts = (persons.groupby('mobility')['person_id'].count()).rename('SynPop{}'.format(YEAR))

mobility = (pd.merge(mob_counts, mob_counts_ist, how='outer', left_index=True, right_index=True)
            .fillna(0).astype(int)
           )

mobility['SynPop{} (%)'.format(YEAR)] = ((mobility['SynPop{}'.format(YEAR)] / len(persons)) * 100).round(1)
mobility['SynPop{} (%)'.format(YEAR_IST)] = ((mobility['SynPop{}'.format(YEAR_IST)] / len(persons_ist)) * 100).round(1)

glue("mobility_stats", mobility.drop('null', axis=0, errors='ignore'), display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

plt.figure(figsize=(10, 20))

explode = [0.02] * mobility.shape[0]

ax1 = plt.subplot(1,2,1)
ax1 = mobility['SynPop{} (%)'.format(YEAR_IST)].sort_index().plot.pie(explode=explode, ax=ax1)
_ = ax1.set_title('Mobility Tools - SynPop{}'.format(YEAR_IST), fontdict={'fontsize': 14, 'fontweight': 'bold'})
_ = ax1.set_ylabel('')
```

```{code-cell} ipython3
:tags: [remove-cell]

plt.figure(figsize=(10, 20))

ax2 = plt.subplot(1,2,2)
ax2 = mobility['SynPop{} (%)'.format(YEAR)].sort_index().plot.pie(explode=explode, ax=ax2)
_ = ax2.set_title('Mobility Tools - SynPop{}'.format(YEAR), fontdict={'fontsize': 14, 'fontweight': 'bold'})
_ = ax2.set_ylabel('')
```

```{code-cell} ipython3
:tags: [remove-cell]

plt.close()
glue("mobownership_prog", ax2.get_figure(), display=False)
glue("mobownership_ist", ax1.get_figure(), display=False)
plt.close()
```

````{panels}
**SynPop {glue:text}`year_ist`**
^^^
```{glue:figure} mobownership_ist
```
---
**SynPop {glue:text}`year`**
^^^
```{glue:figure} mobownership_prog
```
````

````{admonition} Table with detailed results
:class: dropdown
```{glue:figure} mobility_stats
```
````

```{code-cell} ipython3
:tags: [remove-cell]

mobility.drop('null', axis=0, errors='ignore')
```

+++ 

## Avaiable Mobility Tools

```{code-cell} ipython3
:tags: [remove-cell]

df1 = (persons_ist[['has_ga', 'has_ht', 'has_va', 'car_available']].sum()).rename('SynPop{}'.format(YEAR_IST))
df2 = (persons[['has_ga', 'has_ht', 'has_va', 'car_available']].sum()).rename('SynPop{}'.format(YEAR))
    
mobility_tools = pd.merge(df1, df2, left_index=True, right_index=True)

mobility_tools['SynPop{} (%)'.format(YEAR_IST)] = (100*(mobility_tools['SynPop{}'.format(YEAR_IST)] / len(persons_ist))).round(1)
mobility_tools['SynPop{} (%)'.format(YEAR)] = (100*(mobility_tools['SynPop{}'.format(YEAR)] / len(persons))).round(1)

mobility_tools.index = ['ga', 'ht', 'va', 'cars']
```

```{code-cell} ipython3
:tags: [remove-cell]

ax = mobility_tools[['SynPop{}'.format(YEAR_IST), 'SynPop{}'.format(YEAR)]].plot.bar()
ax.grid(axis='y')
_ = plt.ylabel('global count')
title = 'Mobility Tools: Absolute Number'
_ = ax.set_title(title, fontdict={'fontsize': 14, 'fontweight': 'bold'})

glue("mobtools_abs", ax.get_figure(), display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

ax = mobility_tools[['SynPop{} (%)'.format(YEAR_IST), 'SynPop{} (%)'.format(YEAR)]].plot.bar()
ax.grid(axis='y')
_ = plt.ylabel('per person averages')
title = 'Mobility Tools: Per Person Averages'
_ = ax.set_title(title, fontdict={'fontsize': 14, 'fontweight': 'bold'})

glue("mobtools_rel", ax.get_figure(), display=False)
```

````{panels}
**Absolute comparison**
^^^
```{glue:figure} mobtools_abs
```
---
**Relative comparison**
^^^
```{glue:figure} mobtools_rel
```
````

```{code-cell} ipython3
:tags: [remove-input, hide_output]

mobility_tools
```

+++ 

## Mobility Tool Ownership per Age

```{code-cell} ipython3
:tags: [remove-cell]

# defaults for plots
ct_mobility_order = ['car', 'car & ga', 'car & ht', 'car & va', 'car & va & ht', 'ga', 'ht', 'va', 'va & ht', 'nothing']
COLOUR_DICT = {'car': 'k', 'car & ga': 'C0', 'car & ht': 'C1', 'car & va': 'C2',
               'car & va & ht': 'C3', 'ga': 'C4', 'ht': 'C5', 'va': 'C6', 'va & ht': 'C7', 'null': 'grey', 'nothing': 'grey'}
```

```{code-cell} ipython3
:tags: [remove-cell]

marginals = marginal_fitting.compute_ist_vs_scenario_marginal_counts(persons, persons_ist,
                                                                     YEAR, YEAR_IST, 
                                                                     feature='mobility', 
                                                                     control_level_list=['age'])
```

```{code-cell} ipython3
:tags: [remove-cell]

counts_per_cat_ist = (marginals
                  .reset_index()
                  .pivot(index='age', columns='mobility', values=f'counts_{YEAR_IST}')
                  )
counts_per_cat_ist = counts_per_cat_ist[ct_mobility_order]
```

```{code-cell} ipython3
:tags: [remove-input]

title = 'Current Mobility by Age - SynPop{}'.format(YEAR_IST)
_ = plot_multi_class_feature_per_age(counts_per_cat_ist, colour_dict=COLOUR_DICT, ymax=150_000, title=title)
```

```{code-cell} ipython3
:tags: [remove-cell]

counts_per_cat = (marginals
                  .reset_index()
                  .pivot(index='age', columns='mobility', values=f'counts_{YEAR}')
                  )
counts_per_cat = counts_per_cat[ct_mobility_order]
```

```{code-cell} ipython3
:tags: [remove-input]

title = 'Current Mobility by Age - SynPop{}'.format(YEAR)
_ = plot_multi_class_feature_per_age(counts_per_cat, colour_dict=COLOUR_DICT, ymax=150_000, title=title)
```

```{code-cell} ipython3

```
