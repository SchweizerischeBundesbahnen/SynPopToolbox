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

# Language

+++ {"tags": ["remove-cell"]}

* *language* is fixed to match the proportions per commune in the 2017 SynPop
* The choice of persons to fix is done randomly

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

sys.path.append(config['codebase'])

from synpop.visualisations import save_figure, plot_multi_class_feature_per_age
from synpop import marginal_fitting, utils
from synpop.synpop_tables import Persons
```

+++ {"toc": true, "tags": ["remove-cell"]}

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Settings" data-toc-modified-id="Settings-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Settings</a></span></li><li><span><a href="#Loading-Data" data-toc-modified-id="Loading-Data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Loading Data</a></span><ul class="toc-item"><li><span><a href="#SynPop" data-toc-modified-id="SynPop-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>SynPop</a></span><ul class="toc-item"><li><span><a href="#Scenario-Year" data-toc-modified-id="Scenario-Year-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>Scenario-Year</a></span></li><li><span><a href="#IST" data-toc-modified-id="IST-2.1.2"><span class="toc-item-num">2.1.2&nbsp;&nbsp;</span>IST</a></span></li></ul></li></ul></li><li><span><a href="#Examining-SynPop-raw" data-toc-modified-id="Examining-SynPop-raw-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Examining SynPop-raw</a></span><ul class="toc-item"><li><span><a href="#Global-Counts" data-toc-modified-id="Global-Counts-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Global Counts</a></span></li><li><span><a href="#Marginals-per-Commune" data-toc-modified-id="Marginals-per-Commune-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Marginals per Commune</a></span><ul class="toc-item"><li><span><a href="#A-few-examples" data-toc-modified-id="A-few-examples-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>A few examples</a></span></li></ul></li></ul></li><li><span><a href="#Fixing-methodology" data-toc-modified-id="Fixing-methodology-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Fixing methodology</a></span><ul class="toc-item"><li><span><a href="#Expected-Counts" data-toc-modified-id="Expected-Counts-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Expected Counts</a></span></li><li><span><a href="#Fixing-..." data-toc-modified-id="Fixing-...-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Fixing ...</a></span></li></ul></li><li><span><a href="#Examining-SynPop+" data-toc-modified-id="Examining-SynPop+-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Examining SynPop+</a></span><ul class="toc-item"><li><span><a href="#Global-Counts" data-toc-modified-id="Global-Counts-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Global Counts</a></span></li><li><span><a href="#Marginals-per-Commune" data-toc-modified-id="Marginals-per-Commune-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Marginals per Commune</a></span><ul class="toc-item"><li><span><a href="#A-few-examples" data-toc-modified-id="A-few-examples-5.2.1"><span class="toc-item-num">5.2.1&nbsp;&nbsp;</span>A few examples</a></span></li></ul></li><li><span><a href="#A-few-examples" data-toc-modified-id="A-few-examples-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>A few examples</a></span></li></ul></li><li><span><a href="#Saving-Results" data-toc-modified-id="Saving-Results-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Saving Results</a></span></li><li><span><a href="#Export-Notebook-to-HTML" data-toc-modified-id="Export-Notebook-to-HTML-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Export Notebook to HTML</a></span></li></ul></div>

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

COLUMNS_OF_INTEREST = ['person_id', 'language', 'age', 'zone_id', 'kt_id', 'kt_name', 'mun_name', 'KT_full' ]
```

+++ {"tags": ["remove-cell"]}

#### Scenario-Year

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

#### IST

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

+++

## Global Counts

```{code-cell} ipython3
:tags: [remove-cell]

%%time
summary_table = marginal_fitting.compute_comparison_summary(persons, persons_ist, 
                                                            YEAR, YEAR_IST, 
                                                            groupby='language')
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
summary_table.index.name = 'Language'
summary_table = summary_table.astype(int)
print('As reference, the global population growth is: {}'.format(global_pop_growth))
```

```{code-cell} ipython3
:tags: [remove-input]

grid_options = {
    'columnDefs': [{'field': summary_table.index.name}] + [{'field': c} for c in summary_table.columns],
    'defaultColDef': {'sortable': 'true', 'filter': 'true', 'resizable': 'true'},
    'enableRangeSelection': 'false',  # premium feature
    'rowSelection': 'multiple'
}

ag = Grid(grid_data=summary_table,
          grid_options=grid_options,
          height=len(summary_table)*31,
          quick_filter=True,
          export_csv=True,
          export_excel=False,  # premium feature
          show_toggle_edit=False,
          export_mode='auto',
          index=True,
          theme='ag-theme-fresh'
         )
ag
```

```{note}
As reference, the global population growth is: {glue:text}`global_pop_growth`.
```

+++

## Language per Commune

```{code-cell} ipython3
:tags: [remove-cell]

%%time 
marginals = marginal_fitting.compute_ist_vs_scenario_marginal_summary_table(persons, persons_ist,
                                                                            YEAR, YEAR_IST, 
                                                                            feature='language', 
                                                                            control_level_list=['mun_name'])
```

```{code-cell} ipython3
:tags: [remove-cell]

marginals_to_export = (marginals[['expected_counts_{}'.format(YEAR), 
                                  'counts_{}'.format(YEAR), 
                                  'counts_{}'.format(YEAR_IST)]]
                       .reset_index()
                       .pivot(index='mun_name', 
                              columns='language', 
                              values=['counts_{}'.format(YEAR_IST),
                                      'counts_{}'.format(YEAR),
                                      'expected_counts_{}'.format(YEAR)]))
```

```{code-cell} ipython3
:tags: [remove-cell]

# marginals_to_export.to_csv(
#     r"\\wsbbrz0283\mobi\40_Projekte\20210407_Prognose_2050\02_SynPop\Marginals_2050\01_language_per_commune.csv", 
#     sep=";")
```

```{code-cell} ipython3
:tags: [remove-cell]

marginals_to_export.head(2)
```

```{code-cell} ipython3
:tags: [remove-cell]

def create_figure(df, counts, attr, title, small_scale=True):
    fig=plt.figure(figsize=(5,5), dpi=120, facecolor='w', edgecolor='k')
    max_x = df[counts][attr].max()
    max_y = df[f'expected_{counts}'][attr].max()
    max_tot = max(max_x, max_y)
    
    plt.grid(b=True, color='gray', alpha=0.3, linestyle='--')
    plt.plot([0, max_tot], [0, max_tot], 'black', linestyle='-', linewidth=0.5, alpha=0.4)
    plt.scatter(df[counts][attr].values, df[f'expected_{counts}'][attr].values, s=15, alpha=0.3, 
                 linewidths=0, c='#c21c1c')

    if small_scale:
        plt.xlim([0, 50000])
        plt.ylim([0, 50000])
    else:
        plt.xlim([0, max_tot])
        plt.ylim([0, max_tot])
    plt.ylabel(f"expected counts based on {YEAR_IST} ratios", size=10)
    plt.xlabel(f"actual {YEAR} counts", size=10)
    plt.title(title, size=14)
    return fig
```

```{code-cell} ipython3
:tags: [remove-cell]

language = 'french'
title = f'Marginals per commune - {language}'
fig = create_figure(marginals_to_export, 'counts_{}'.format(YEAR), language, title, small_scale=False)

glue(language, fig, display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

language = 'german'
title = f'Marginals per commune - {language}'
fig = create_figure(marginals_to_export, 'counts_{}'.format(YEAR), language, title, small_scale=False)

glue(language, fig, display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

language = 'italian'
title = f'Marginals per commune - {language}'
fig = create_figure(marginals_to_export, 'counts_{}'.format(YEAR), language, title, small_scale=False)

glue(language, fig, display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

language = 'other'
title = f'Marginals per commune - {language}'
fig = create_figure(marginals_to_export, 'counts_{}'.format(YEAR), language, title, small_scale=False)

glue(language, fig, display=False)
```

```{code-cell} ipython3

```
