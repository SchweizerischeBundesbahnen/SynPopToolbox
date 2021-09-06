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

# Education

+++ {"tags": ["remove-cell"]}

**Comments**: 
* "current_edu" is fixed to match the proportions per age in the 2017 SynPop
* the choice of agents to change is random

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

sys.path.append(config['codebase'])

from synpop.visualisations import save_figure, plot_multi_class_feature_per_age, plot_ct1, plot_ct2, plot_ct3
from synpop import marginal_fitting, utils
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

glue("year_ist", YEAR_IST)
glue("year", YEAR)
```

+++ {"tags": ["remove-cell"]}

# Loading Data

+++ {"tags": ["remove-cell"]}

## SynPop

```{code-cell} ipython3
:tags: [remove-cell]

COLUMNS_OF_INTEREST = ['person_id', 'level_of_employment', 'position_in_bus', 'position_in_edu', 'age', 'zone_id', 
                       'is_employed', 'loe_group', 'current_edu', 'current_job_rank', 'is_apprentice', 
                       'kt_id', 'kt_name', 'mun_name', 'KT_full' ]
```

```{code-cell} ipython3
:tags: [remove-cell]

COLOUR_DICT = {'kindergarten': 'k', 'pupil_primary': 'C0', 'pupil_secondary': 'C1', 'student': 'C2',
               'apprentice': 'C3',  'null': 'grey'}
```

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

persons = synpop_persons.data[COLUMNS_OF_INTEREST].copy(deep=True)

del synpop_persons
```

+++ {"tags": ["remove-cell"]}

**Year IST**

```{code-cell} ipython3
:tags: [remove-cell]

SYNPOP_PERSONS_FILE_IST
```

```{code-cell} ipython3
:tags: [remove-cell]

%%time
synpop_persons_ist = Persons(YEAR_IST)
synpop_persons_ist.load(SYNPOP_PERSONS_FILE_IST)

persons_ist = synpop_persons_ist.data[COLUMNS_OF_INTEREST].copy(deep=True)

del synpop_persons_ist
```

```{code-cell} ipython3
:tags: [remove-cell]

print('Imported data:')
print('persons DataFrame for {}: {}'.format(YEAR_IST, persons_ist.shape))
print('persons DataFrame for {}: {}'.format(YEAR, persons.shape))
```

+++ {"tags": ["remove-cell"]}

# Examining SynPop-raw

```{code-cell} ipython3
:tags: [remove-cell]

ORDERED_CATEGORIES = list(persons['current_edu'].cat.categories)
```

+++

## Global Counts

```{code-cell} ipython3
:tags: [remove-cell]

%%time
summary_table = marginal_fitting.compute_comparison_summary(persons, persons_ist, 
                                                            YEAR, YEAR_IST, 
                                                            groupby='current_edu')
```

```{code-cell} ipython3
:tags: [remove-cell]

global_pop_growth = '{:.2f} %'.format((persons.shape[0] / persons_ist.shape[0]) - 1)
glue('global_pop_growth', global_pop_growth, display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

summary_table['growth'] = summary_table['growth'] - 1
summary_table.columns = [f'counts {YEAR_IST}', f'counts {YEAR}', 
                         'absolute growth (%)', f'proportion {YEAR_IST}', f'proportion {YEAR}']
summary_table[['absolute growth (%)', f'proportion {YEAR_IST}', f'proportion {YEAR}']] = (
    100 * summary_table[['absolute growth (%)', f'proportion {YEAR_IST}', f'proportion {YEAR}']])
summary_table.index.name = 'Education'
print('As reference, the global population growth is: {}'.format(global_pop_growth))
```

```{code-cell} ipython3
:tags: [remove-input]

summary_table
```

```{note}
As reference, the global population growth is: {glue:text}`global_pop_growth`.
```

+++ 

## Education Per Age

```{code-cell} ipython3
:tags: [remove-cell]

%%time 
marginals = marginal_fitting.compute_ist_vs_scenario_marginal_counts(persons, persons_ist,
                                                                     YEAR, YEAR_IST, 
                                                                     feature='current_edu', 
                                                                     control_level_list=['age'])
```

+++ {"tags": ["remove-cell"]}

**Year IST**

```{code-cell} ipython3
:tags: [remove-cell]

counts_per_cat_ist = (marginals
                      .reset_index()
                      .pivot(index='age', columns='current_edu', values='counts_{}'.format(YEAR_IST))
                      )
counts_per_cat_ist = counts_per_cat_ist[ORDERED_CATEGORIES]
```

```{code-cell} ipython3
:tags: [remove-input]

title = f'Current Education by Age - SynPop{YEAR_IST}'

ax = plot_multi_class_feature_per_age(counts_per_cat_ist, colour_dict=COLOUR_DICT, ymax=150_000, title=title)
```

+++ {"tags": ["remove-cell"]}

**Scenario Year**

```{code-cell} ipython3
:tags: [remove-cell]

counts_per_cat = (marginals
                  .reset_index()
                  .pivot(index='age', columns='current_edu', values=f'counts_{YEAR}')
                  )
counts_per_cat = counts_per_cat[ORDERED_CATEGORIES]
```

```{code-cell} ipython3
:tags: [remove-input]

title = 'Current Education by Age - SynPop{}'.format(YEAR)

ax = plot_multi_class_feature_per_age(counts_per_cat, colour_dict=COLOUR_DICT, ymax=150_000, title=title)
```

+++ 

## Cross-Relationship with Employment and Job Rank

+++ 

### Reference Year - SynPop {glue:text}`year_ist`

+++ {"tags": ["remove-cell"]}

**CT1**

```{code-cell} ipython3
:tags: [remove-cell]

ax_ct1, _ = plot_ct1(persons_ist, title=f'Education vs Employment Status - SynPop {YEAR_IST}')
```

+++ {"tags": ["remove-cell"]}

**CT2**

```{code-cell} ipython3
:tags: [remove-cell]

ax_ct2, _ = plot_ct2(persons_ist, title=f'Education vs Job Rank - SynPop {YEAR_IST}')
```

+++ {"tags": ["remove-cell"]}

**CT3**

```{code-cell} ipython3
:tags: [remove-cell]

ax_ct3, _ = plot_ct2(persons_ist, title=f'Job Rank vs Employment Status - SynPop {YEAR_IST}')
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("ct1_ref", ax_ct1.get_figure(), display=False)
glue("ct2_ref", ax_ct2.get_figure(), display=False)
glue("ct3_ref", ax_ct3.get_figure(), display=False)
plt.close()
```

````{tabbed} Employment
```{glue:figure} ct1_ref
:figwidth: 800px
```
````

````{tabbed} Job Rank
```{glue:figure} ct2_ref
:figwidth: 800px
```
````

+++ 

### Prognosis Year - SynPop {glue:text}`year`

+++ {"tags": ["remove-cell"]}

**CT1**

```{code-cell} ipython3
:tags: [remove-cell]

ax_ct1, _ = plot_ct1(persons, title=f'Education vs Employment Status - SynPop {YEAR}')
```

+++ {"tags": ["remove-cell"]}

**CT2**

```{code-cell} ipython3
:tags: [remove-cell]

ax_ct2, _ = plot_ct2(persons, title=f'Education vs Job Rank - SynPop {YEAR}')
```

+++ {"tags": ["remove-cell"]}

**CT3**

```{code-cell} ipython3
:tags: [remove-cell]

ax_ct3, _ = plot_ct3(persons, title=f'Job Rank vs Employment Status - SynPop {YEAR}')
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("ct1_prog", ax_ct1.get_figure(), display=False)
glue("ct2_prog", ax_ct2.get_figure(), display=False)
glue("ct3_prog", ax_ct3.get_figure(), display=False)
plt.close()
```

````{tabbed} Employment
```{glue:figure} ct1_prog
:figwidth: 800px
```
````

````{tabbed} Job Rank
```{glue:figure} ct2_prog
:figwidth: 800px
```
````
