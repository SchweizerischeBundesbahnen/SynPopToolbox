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

# Businesses

+++ {"tags": ["remove-cell"]}

* **Input**: cleaned and optimized pickeled DataFrames (businesses.csv) for 20XX
* **Output**: Visualisations

+++ {"toc": true, "tags": ["remove-cell"]}

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Settings" data-toc-modified-id="Settings-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Settings</a></span></li><li><span><a href="#Loading-Data" data-toc-modified-id="Loading-Data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Loading Data</a></span><ul class="toc-item"><li><span><a href="#ZoneId-to-MS-Region-Mapping" data-toc-modified-id="ZoneId-to-MS-Region-Mapping-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>ZoneId to MS-Region Mapping</a></span></li><li><span><a href="#SynPop" data-toc-modified-id="SynPop-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>SynPop</a></span></li></ul></li><li><span><a href="#Analysis" data-toc-modified-id="Analysis-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Analysis</a></span><ul class="toc-item"><li><span><a href="#Globally" data-toc-modified-id="Globally-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Globally</a></span></li><li><span><a href="#By-Sector" data-toc-modified-id="By-Sector-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>By Sector</a></span></li><li><span><a href="#By-Canton" data-toc-modified-id="By-Canton-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>By Canton</a></span></li><li><span><a href="#By-MS-Region" data-toc-modified-id="By-MS-Region-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>By MS-Region</a></span></li><li><span><a href="#Create-Jobs-Report-(exel-document)" data-toc-modified-id="Create-Jobs-Report-(exel-document)-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>Create Jobs Report (exel document)</a></span></li></ul></li></ul></div>

```{code-cell} ipython3
:tags: [remove-cell]

import logging
import os
import sys
import math

import pandas as pd
import geopandas as gpd
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

from synpop import visualisations
from synpop.visualisations import generate_simple_grid_table, altair_scatter
from synpop.zone_maps import SwissZoneMap
from synpop import mobi_zones_path
from synpop.synpop_tables import SynPop
import synpop.utils as utils
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

MOBI_ZONES_SHP = mobi_zones_path()


if YEAR == YEAR_IST:
    YEAR_IST = f'{YEAR_IST} (Ref)'
```

+++ {"tags": ["remove-cell"]}

## Loading Data

+++ {"tags": ["remove-cell"]}

### SynPop

```{code-cell} ipython3
:tags: [remove-cell]

synpop = SynPop(YEAR)
synpop.load(SYNPOP_PATH, config["validate"])

persons = synpop.persons.data
if "is_employed" not in persons.columns:
    persons["is_employed"] = persons["level_of_employment"] > 0
businesses = synpop.businesses.data
```

```{code-cell} ipython3
:tags: [remove-cell]

synpop_ist = SynPop(YEAR_IST)
synpop_ist.load(SYNPOP_PATH_IST, config["validate"])

persons_ist = synpop_ist.persons.data
if "is_employed" not in persons_ist.columns:
    persons_ist["is_employed"] = persons_ist["level_of_employment"] > 0
businesses_ist = synpop_ist.businesses.data
```

+++ {"tags": ["remove-cell"]}

## Analysis

+++

## Global Counts

```{code-cell} ipython3
:tags: [remove-cell]

def compute_stats_summary(df_ist, df_scenario, persons_ist, persons, year_ist, year_scenario):
    stats_list = []
    for df, persons, year in zip((df_ist, df_scenario), (persons_ist, persons), (year_ist,  year_scenario)):
        stats = (df.agg({'sector': 'count',
                         'jobs_endo':sum,
                         'jobs_exo':sum,
                         'fte_endo':sum,
                         'fte_exo':sum}
                        )
                   .rename(index={'sector': 'businesses'})
                   .astype(int)
                )
        stats.loc['total_jobs'] = stats.loc['jobs_endo'] + stats.loc['jobs_exo']
        stats.loc['total_fte'] = stats.loc['fte_endo'] + stats.loc['fte_exo']
        stats.loc['total_employed'] = persons['level_of_employment'].gt(0).sum()
        stats = stats.rename(year)
        stats_list.append(stats)

    stats_summary = pd.concat(stats_list, axis=1)
    stats_summary['% change'] = ((stats_summary[year_scenario] - stats_summary[year_ist]) / stats_summary[year_scenario] * 100).round(1)

    return stats_summary
```

+++ {"tags": ["remove-cell"]}

**Businesses & Schools**

```{code-cell} ipython3
:tags: [remove-cell]

stats_summary_with_schools = compute_stats_summary(
    businesses_ist, businesses, persons_ist, persons, YEAR_IST, YEAR)
stats_summary_with_schools
glue("business_with_schools", stats_summary_with_schools)
```

+++ {"tags": ["remove-cell"]}

**Businesses without Schools**

```{code-cell} ipython3
:tags: [remove-cell]

stats_summary = compute_stats_summary(businesses_ist.query('school_type == "no_school"'),
                                      businesses.query('school_type == "no_school"'),
                                      persons_ist, persons,
                                      YEAR_IST, YEAR
                                     )
stats_summary
glue("business_without_schools", stats_summary)
```

+++ {"tags": ["remove-cell"]}

**Schools**

```{code-cell} ipython3
:tags: [remove-cell]

def compute_school_summary(df_ist, df_scenario, year_ist, year_scenario):
    remap = {
        'kindergarten': 'primary',
        'secondary_1': 'secondary',
        'secondary_2': 'secondary',
        'univ_applied_sciences': 'other_higher_education',
        'higher_technical_school': 'other_higher_education',
        'professional_school': 'secondary'
        }
    stats_list = []
    for df, year in zip((df_ist, df_scenario), (year_ist,  year_scenario)):
        stats = (df.query('school_type != "no_school"')
                   .replace({'school_type': remap})
                   .groupby('school_type', observed=True).size()
                   .rename(year)
                 )
        stats.index = stats.index.tolist()
        stats.loc['TOTAL'] = stats.sum()

        stats_list.append(stats)

    stats_summary = pd.concat(stats_list, axis=1)
    stats_summary['% change'] = ((stats_summary[year_scenario] - stats_summary[year_ist]) / stats_summary[year_scenario] * 100).astype(int)

    return stats_summary
```

```{code-cell} ipython3
:tags: [remove-cell]

school_stats = compute_school_summary(businesses_ist, businesses, YEAR_IST, YEAR)
school_stats
glue("schools_only", school_stats)
```

```{code-cell} ipython3
:tags: [remove-cell]

title = 'Businesses - Number of Schools - SynPop {year_ist} vs. SynPop {year}'.format(year_ist=YEAR_IST, year=YEAR)

ax = school_stats.iloc[:-1, :2].plot.bar(figsize=(12, 6), rot=45)
_ = plt.grid(axis='y')
_ = ax.set_ylabel('nbr schools')
_ = ax.set_xlabel('')
_ = ax.set_title(title, pad=25, fontdict={'fontsize': 16, 'fontweight': 'bold'})
```

````{tabbed} Businesses & Schools
```{glue:figure} business_with_schools
:figwidth: 400px
```
````

````{tabbed} Business except Schools
```{glue:figure} business_without_schools
:figwidth: 400px
```
````

````{tabbed} Schools only
```{glue:figure} schools_only
:figwidth: 400px
```
````

+++

### By Sector

```{code-cell} ipython3
:tags: [remove-cell]

stats_per_sector = visualisations.get_businesses_comparison_summary_by_category(businesses_ist, businesses,
                                                                                YEAR_IST, YEAR,
                                                                                query='school_type == "no_school"',
                                                                                agg_column='sector'
                                                                                )
```

```{code-cell} ipython3
:tags: [remove-cell]

category = 'sector'
query='school_type == "no_school"'
statistic = 'total_businesses'
title = ('Businesses - {statistic} per {category} - SynPop {year_ist} vs. SynPop {year}'
         .format(year_ist=YEAR_IST, year=YEAR, statistic=statistic, category=category)
         .replace('_', '-')
        )

ax = visualisations.plot_businesses_comparison_by_category(businesses_ist, businesses,
                                                           YEAR_IST, YEAR,
                                                           statistic=statistic,
                                                           query=query,
                                                           agg_column=category,
                                                           title=title
                                                          )
glue("sector_businesses", ax.get_figure(), display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

category = 'sector'
query='school_type == "no_school"'
statistic = 'total_jobs'
title = ('Businesses - {statistic} per {category} - SynPop {year_ist} vs. SynPop {year}'
         .format(year_ist=YEAR_IST, year=YEAR, statistic=statistic, category=category)
         .replace('_', '-')
        )

ax = visualisations.plot_businesses_comparison_by_category(businesses_ist, businesses,
                                                           YEAR_IST, YEAR,
                                                           statistic=statistic,
                                                           query=query,
                                                           agg_column=category,
                                                           title=title
                                                          )
glue("sector_jobs", ax.get_figure(), display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

category = 'sector'
query='school_type == "no_school"'
statistic = 'total_fte'
title = ('Businesses - {statistic} per {category} - SynPop {year_ist} vs. SynPop {year}'
         .format(year_ist=YEAR_IST, year=YEAR, statistic=statistic, category=category)
         .replace('_', '-')
        )

ax = visualisations.plot_businesses_comparison_by_category(businesses_ist, businesses,
                                                           YEAR_IST, YEAR,
                                                           statistic=statistic,
                                                           query=query,
                                                           agg_column=category,
                                                           title=title
                                                          )
glue("sector_fte", ax.get_figure(), display=False)
```

````{tabbed} Businesses by sector
```{glue:figure} sector_businesses
:figwidth: 800px
```
````

````{tabbed} Jobs by sector
```{glue:figure} sector_jobs
:figwidth: 800px
```
````

````{tabbed} FTE by sector
```{glue:figure} sector_fte
:figwidth: 800px
```
````

## Per Canton

```{code-cell} ipython3
:tags: [remove-cell]

businesses_by_canton = businesses.groupby('KT_full')['jobs_endo'].sum().rename(f'Jobs {YEAR}')
employed_by_canton = persons.groupby('KT_full')['is_employed'].sum().rename(f'Employed {YEAR}')
ratio_by_canton = employed_by_canton.div(businesses_by_canton).rename(f'Employed-Job Ratio {YEAR}').round(2)
```

```{code-cell} ipython3
:tags: [remove-cell]

businesses_by_canton_ist = businesses_ist.groupby('KT_full')['jobs_endo'].sum().rename(f'Jobs {YEAR_IST}')
employed_by_canton_ist = persons_ist.groupby('KT_full')['is_employed'].sum().rename(f'Employed {YEAR_IST}')
ratio_by_canton_ist = employed_by_canton_ist.div(businesses_by_canton_ist).rename(f'Employed-Job Ratio {YEAR_IST}').round(2)
```

```{code-cell} ipython3
:tags: [remove-cell]

jobs_by_canton = pd.concat([businesses_by_canton, businesses_by_canton_ist], axis=1, sort=False)
jobs_by_canton['AbsJobGrowth'] = (jobs_by_canton[f'Jobs {YEAR}'] - jobs_by_canton[f'Jobs {YEAR_IST}'])
jobs_by_canton['RelJobGrowth'] = (jobs_by_canton['AbsJobGrowth'] / jobs_by_canton[f'Jobs {YEAR_IST}'] * 100).round(1)
jobs_by_canton = pd.concat([jobs_by_canton, employed_by_canton, employed_by_canton_ist], axis=1, sort=False)
jobs_by_canton['AbsEmployedGrowth'] = (jobs_by_canton[f'Employed {YEAR}'] - jobs_by_canton[f'Employed {YEAR_IST}'])
jobs_by_canton['RelEmployedGrowth'] = (jobs_by_canton['AbsEmployedGrowth'] / jobs_by_canton[f'Employed {YEAR_IST}'] * 100).round(1)
jobs_by_canton = pd.concat([jobs_by_canton, ratio_by_canton, ratio_by_canton_ist], axis=1, sort=False)
jobs_by_canton['RatioChange'] = (ratio_by_canton - ratio_by_canton_ist).round(2)
glue("jobs_by_canton", jobs_by_canton, display=False)
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

title = 'SynPop {} vs. SynPop {}: Absolute Job Growth (endo only)'.format(YEAR, YEAR_IST)
scale_bound = round_up(jobs_by_canton['AbsJobGrowth'].abs().max())
ax1, _ = default_map_client.draw_cantons(jobs_by_canton, 'AbsJobGrowth', vmin=-scale_bound, vmax=scale_bound, title=title)
```

```{code-cell} ipython3
:tags: [remove-cell]

title = 'SynPop {} vs. SynPop {}: % Job Growth (endo only)'.format(YEAR, YEAR_IST)
scale_bound = round(jobs_by_canton['RelJobGrowth'].abs().max())
ax2, _ = default_map_client.draw_cantons(jobs_by_canton, 'RelJobGrowth', vmin=-scale_bound, vmax=scale_bound, title=title)
```

```{code-cell} ipython3
:tags: [remove-cell]

title = 'SynPop {} vs. SynPop {}: Absolute Employed Growth'.format(YEAR, YEAR_IST)
scale_bound = round_up(jobs_by_canton['AbsEmployedGrowth'].abs().max())
ax3, _ = default_map_client.draw_cantons(jobs_by_canton, 'AbsEmployedGrowth', vmin=-scale_bound, vmax=scale_bound, title=title)
```

```{code-cell} ipython3
:tags: [remove-cell]

title = 'SynPop {} vs. SynPop {}: % Employed Growth'.format(YEAR, YEAR_IST)
scale_bound = round(jobs_by_canton['RelEmployedGrowth'].abs().max())
ax4, _ = default_map_client.draw_cantons(jobs_by_canton, 'RelEmployedGrowth', vmin=-scale_bound, vmax=scale_bound, title=title)
```

```{code-cell} ipython3
:tags: [remove-cell]

title = 'SynPop {} vs. SynPop {}: % Employed-to-Job Ratio Change'.format(YEAR, YEAR_IST)
scale_bound = round(jobs_by_canton['RatioChange'].abs().max())
ax5, _ = default_map_client.draw_cantons(jobs_by_canton, 'RatioChange', vmin=-scale_bound, vmax=scale_bound, title=title)
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("abs_diff_jobs", ax1.get_figure(), display=False)
glue("rel_diff_jobs", ax2.get_figure(), display=False)
glue("abs_diff_empl", ax3.get_figure(), display=False)
glue("rel_diff_empl", ax4.get_figure(), display=False)
glue("diff_ratio", ax5.get_figure(), display=False)
plt.close()
```

````{tabbed} Absolute Job growth
```{glue:figure} abs_diff_jobs
:figwidth: 800px
```
````

````{tabbed} Relative Job growth
```{glue:figure} rel_diff_jobs
:figwidth: 800px
```
````

````{tabbed} Absolute Employed growth
```{glue:figure} abs_diff_empl
:figwidth: 800px
```
````

````{tabbed} Relative Employed growth
```{glue:figure} rel_diff_empl
:figwidth: 800px
```
````

````{tabbed} Employed-to-Job ratio change
```{glue:figure} diff_ratio
:figwidth: 800px
```
````

<div style="text-align: right; font-weight: bold"> Detailed results per Canton</div>

```{code-cell} ipython3
:tags: [remove-input, hide-output, full-width]

col_widths = {'Canton': 150}
colnames = {'KT_full': 'Canton'}
column_toggles = {
    'Job': [f'Jobs {YEAR}', f'Jobs {YEAR_IST}', 'AbsJobGrowth', 'RelJobGrowth'],
    'Employed': [f'Employed {YEAR}', f'Employed {YEAR_IST}', 'AbsEmployedGrowth', 'RelEmployedGrowth'],
    'Ratio': [f'Employed-Job Ratio {YEAR}', f'Employed-Job Ratio {YEAR_IST}', 'RatioChange']}
generate_simple_grid_table(jobs_by_canton.reset_index(), colnames, header_height=60, column_toggles=column_toggles, col_widths=col_widths)
```

<div style="text-align: right; font-weight: bold"> Detailed results per Municipality </div>

```{code-cell} ipython3
:tags: [remove-cell]

job_by_mun = businesses.groupby(['KT_full', 'mun_name'])['jobs_endo'].sum().rename(f'Jobs {YEAR}')
job_by_mun_ist = businesses_ist.groupby(['KT_full', 'mun_name'])['jobs_endo'].sum().rename(f'Jobs {YEAR_IST}')
employed_by_mun = persons.groupby(['KT_full', 'mun_name'])['is_employed'].sum().rename(f'Employed {YEAR}')
ratio_by_mun = employed_by_mun.div(job_by_mun).rename(f'Employed-Job Ratio {YEAR}').round(2)
employed_by_mun_ist = persons_ist.groupby(['KT_full', 'mun_name'])['is_employed'].sum().rename(f'Employed {YEAR_IST}')
ratio_by_mun_ist = employed_by_mun_ist.div(job_by_mun_ist).rename(f'Employed-Job Ratio {YEAR_IST}').round(2)
```

```{code-cell} ipython3
:tags: [remove-cell]

job_by_mun = pd.concat([job_by_mun, job_by_mun_ist], axis=1)
job_by_mun['AbsJobGrowth'] = (job_by_mun[f'Jobs {YEAR}'] - job_by_mun[f'Jobs {YEAR_IST}'])
job_by_mun['RelJobGrowth'] = (job_by_mun['AbsJobGrowth'] / job_by_mun[f'Jobs {YEAR_IST}'] * 100).round(1)
job_by_mun = pd.concat([job_by_mun, employed_by_mun, employed_by_mun_ist], axis=1)
job_by_mun['AbsEmployedGrowth'] = (job_by_mun[f'Employed {YEAR}'] - job_by_mun[f'Employed {YEAR_IST}'])
job_by_mun['RelEmployedGrowth'] = (job_by_mun['AbsEmployedGrowth'] / job_by_mun[f'Employed {YEAR_IST}'] * 100).round(1)
job_by_mun = pd.concat([job_by_mun, ratio_by_mun, ratio_by_mun_ist], axis=1)
job_by_mun['RatioChange'] = (ratio_by_mun - ratio_by_mun_ist ).round(2)
```

```{code-cell} ipython3
:tags: [remove-input, hide-output, full-width]

col_widths = {'Canton': 150, 'Commune': 150}
colnames = {'KT_full': 'Canton', 'mun_name': 'Commune'}
generate_simple_grid_table(job_by_mun.reset_index(), colnames, header_height=60, column_toggles=column_toggles, col_widths=col_widths)
```

## Growth Analysis

```{code-cell} ipython3
:tags: [remove-cell]

%%time
mun_stats = (pd.concat([businesses_ist.groupby(['KT_full', 'mun_name'])['jobs_endo'].sum().rename(f'jobs {YEAR_IST}'),
                        businesses.groupby(['KT_full', 'mun_name'])['jobs_endo'].sum().rename(f'jobs {YEAR}')
                        ], axis=1, join='outer')
             .fillna(0)
             .astype(int)
            )
```

```{code-cell} ipython3
:tags: [remove-cell]

mun_stats['growth abs'] = mun_stats[f'jobs {YEAR}'] - mun_stats[f'jobs {YEAR_IST}']
mun_stats['growth factor'] = mun_stats[f'jobs {YEAR}'] / mun_stats[f'jobs {YEAR_IST}']
mun_stats['growth factor'] = mun_stats['growth factor'].replace(np.inf, np.nan)
mun_stats['growth factor'] = mun_stats['growth factor'].round(3)
mun_stats = mun_stats.sort_values(f'jobs {YEAR_IST}', ascending=False)

mun_stats = mun_stats[[f'jobs {YEAR_IST}', f'jobs {YEAR}', 'growth abs', 'growth factor']].reset_index()
```

### Absolute Growth - Jobs

```{code-cell} ipython3
:tags: [remove-input]

altair_scatter(mun_stats, f'jobs {YEAR_IST}', 'growth abs', dropdown_col='KT_full', columns=mun_stats.columns.tolist()[1:])
```

### Relative Growth - Jobs

```{code-cell} ipython3
:tags: [remove-input]

altair_scatter(mun_stats, f'jobs {YEAR_IST}', 'growth factor', dropdown_col='KT_full', columns=mun_stats.columns.tolist()[1:])
```

```{code-cell} ipython3
:tags: [remove-cell]

%%time
mun_stats_empl = (pd.concat([persons_ist.groupby(['KT_full', 'mun_name'])['is_employed'].sum().rename(f'employed {YEAR_IST}'),
                        persons.groupby(['KT_full', 'mun_name'])['is_employed'].sum().rename(f'employed {YEAR}')
                        ], axis=1, join='outer')
             .fillna(0)
             .astype(int)
            )
```

```{code-cell} ipython3
:tags: [remove-cell]

mun_stats_empl['growth abs'] = mun_stats_empl[f'employed {YEAR}'] - mun_stats_empl[f'employed {YEAR_IST}']
mun_stats_empl['growth factor'] = mun_stats_empl[f'employed {YEAR}'] / mun_stats_empl[f'employed {YEAR_IST}']
mun_stats_empl['growth factor'] = mun_stats_empl['growth factor'].replace(np.inf, np.nan)
mun_stats_empl['growth factor'] = mun_stats_empl['growth factor'].round(3)
mun_stats_empl = mun_stats_empl.sort_values(f'employed {YEAR_IST}', ascending=False)

mun_stats_empl = mun_stats_empl[[f'employed {YEAR_IST}', f'employed {YEAR}', 'growth abs', 'growth factor']].reset_index()
```

### Absolute Growth - Employed

```{code-cell} ipython3
:tags: [remove-input]

altair_scatter(mun_stats_empl, f'employed {YEAR_IST}', 'growth abs', dropdown_col='KT_full', columns=mun_stats_empl.columns.tolist()[1:])
```

### Relative Growth - Employed

```{code-cell} ipython3
:tags: [remove-input]

altair_scatter(mun_stats_empl, f'employed {YEAR_IST}', 'growth factor', dropdown_col='KT_full', columns=mun_stats_empl.columns.tolist()[1:])
```
