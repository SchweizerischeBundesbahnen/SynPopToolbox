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

# Overview of Attributes

```{code-cell} ipython3
:tags: [remove-cell]

import logging
import os
import sys

from matplotlib import pyplot as plt
import pandas as pd
from ipyaggrid import Grid
```

```{code-cell} ipython3
:tags: [remove-cell]

import yaml
from myst_nb import glue

# load config and synpop code
with open('_config.yml', 'r') as f:
    config = yaml.safe_load(f)['synpop_report_config']

sys.path.insert(1, config['codebase'])

from synpop.visualisations import save_figure, plot_multi_class_feature_per_age, generate_simple_grid_table
from synpop import marginal_fitting, utils, fitting_analysis
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

+++ {"tags": ["remove-cell"]}

#### Scenario-Year

```{code-cell} ipython3
:tags: [remove-cell]

%%time
synpop_persons = Persons(YEAR)
synpop_persons.load(SYNPOP_PATH, config["validate"])

persons = synpop_persons.data
```

+++ {"tags": ["remove-cell"]}

#### IST

```{code-cell} ipython3
:tags: [remove-cell]

%%time
synpop_persons_ist = Persons(YEAR_IST)
synpop_persons_ist.load(SYNPOP_PATH_IST, config["validate"])

persons_ist = synpop_persons_ist.data


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

## Examining SynPop-raw

## Global Counts

```{code-cell} ipython3
:tags: [remove-cell]

variables = ['language', 'current_edu', 'current_job_rank']
```

```{code-cell} ipython3
:tags: [remove-cell]

%%time
summary_tables = {var: fitting_analysis.compute_comparison_summary(persons, persons_ist,
                                                                   YEAR, YEAR_IST,
                                                                   groupby=var)
                  for var in variables}
```

```{code-cell} ipython3
:tags: [remove-cell]

_ = [glue(f'{var}_summary', table, display=False) for var, table in summary_tables.items()]
```

````{tabbed} Education
```{glue:figure} current_edu_summary
:figwidth: 800px
```
````

````{tabbed} Job Rank
```{glue:figure} current_job_rank_summary
:figwidth: 800px
```
````

````{tabbed} Language
```{glue:figure} language_summary
:figwidth: 800px
```
````

+++ {"tags": ["remove-cell"]}

## Detailed analysis

+++

### Education

<div style="text-align: right; font-weight: bold"> Education per Canton</div>

```{code-cell} ipython3
:tags: [remove-cell]

edu_by_canton = marginal_fitting.comparison_table_categories(
    persons, persons_ist, YEAR, YEAR_IST, 'current_edu', 'KT_full')
edu_by_commune = marginal_fitting.comparison_table_categories(
    persons, persons_ist, YEAR, YEAR_IST, 'current_edu', ['KT_full', 'mun_name'])
```

```{code-cell} ipython3
:tags: [remove-cell]

def get_toggles_and_hidden(categories, default_cat=None):
    column_toggles = {c: [f'{c} {YEAR}', f'{c} {YEAR_IST}', f'AbsGrowth {c}', f'RelGrowth {c}'] for c in categories}
    if default_cat is None:
        default_cat = list(column_toggles.keys())[-1]
    hidden_cols = [c for sl in column_toggles.values() for c in sl if default_cat not in c]
    return {'column_toggles': column_toggles, 'hidden_cols': hidden_cols}
```

```{code-cell} ipython3
:tags: [remove-cell]

colnames_kt = {'KT_full': 'Canton'}
col_widths_kt= {'Canton': 150}
colnames_commune = {'KT_full': 'Canton', 'mun_name': 'Commune'}
col_widths_commune = {'Canton': 150, 'Commune': 150}
```

```{code-cell} ipython3
:tags: [remove-input, hide-output, full-width]

generate_simple_grid_table(edu_by_canton.reset_index(), colnames_kt, header_height=60,
                           col_widths=col_widths_kt,
                           **get_toggles_and_hidden(persons['current_edu'].cat.categories, 'student'))
```

<div style="text-align: right; font-weight: bold"> Education per Commune</div>

```{code-cell} ipython3
:tags: [remove-input, hide-output, full-width]

generate_simple_grid_table(edu_by_commune.reset_index(), colnames_commune, header_height=60,
                           col_widths=col_widths_commune,
                           **get_toggles_and_hidden(persons['current_edu'].cat.categories, 'student'))
```

### Job-Rank

<div style="text-align: right; font-weight: bold"> Job-Rank per Canton</div>

```{code-cell} ipython3
:tags: [remove-cell]

job_rank_by_canton = marginal_fitting.comparison_table_categories(
    persons, persons_ist, YEAR, YEAR_IST, 'current_job_rank', 'KT_full')
job_rank_by_commune = marginal_fitting.comparison_table_categories(
    persons, persons_ist, YEAR, YEAR_IST, 'current_job_rank', ['KT_full', 'mun_name'])
```

```{code-cell} ipython3
:tags: [remove-input, hide-output, full-width]

generate_simple_grid_table(job_rank_by_canton.reset_index(), colnames_kt, header_height=60,
                           **get_toggles_and_hidden(persons['current_job_rank'].cat.categories, 'employee'))
```

<div style="text-align: right; font-weight: bold"> Job-Rank per Commune</div>

```{code-cell} ipython3
:tags: [remove-input, hide-output, full-width]

generate_simple_grid_table(job_rank_by_commune.reset_index(), colnames_commune, header_height=60,
                           **get_toggles_and_hidden(persons['current_job_rank'].cat.categories, 'employee'))
```

### Language

<div style="text-align: right; font-weight: bold"> Language per Canton</div>

```{code-cell} ipython3
:tags: [remove-cell]

language_by_canton = marginal_fitting.comparison_table_categories(
    persons, persons_ist, YEAR, YEAR_IST, 'language', 'KT_full')
language_by_commune = marginal_fitting.comparison_table_categories(
    persons, persons_ist, YEAR, YEAR_IST, 'language', ['KT_full', 'mun_name'])
```

```{code-cell} ipython3
:tags: [remove-input, hide-output, full-width]

generate_simple_grid_table(language_by_canton.reset_index(), colnames_kt, header_height=60,
                           **get_toggles_and_hidden(persons['language'].cat.categories, 'german'))
```

<div style="text-align: right; font-weight: bold"> Language per Commune</div>

```{code-cell} ipython3
:tags: [remove-input, hide-output, full-width]

generate_simple_grid_table(language_by_commune.reset_index(), colnames_commune, header_height=60,
                           **get_toggles_and_hidden(persons['language'].cat.categories, 'german'))
```
