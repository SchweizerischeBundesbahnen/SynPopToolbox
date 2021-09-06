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

import pandas as pd
import geopandas as gpd

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

from synpop import visualisations
from synpop.synpop_tables import Businesses
import synpop.utils as utils
```

+++ {"tags": ["remove-cell"]}

##  Settings

```{code-cell} ipython3
:tags: [remove-cell]

# not displayed
YEAR_IST = config['reference_year']
YEAR = config['target_year']

DATA_DIR_IST = config['reference_synpop']
SYNPOP_BUSINESSES_FILE_IST = os.path.join(DATA_DIR_IST, 'businesses_{}'.format(YEAR_IST))

DATA_DIR = config['target_synpop']
SYNPOP_BUSINESSES_FILE = os.path.join(DATA_DIR, 'businesses_{}'.format(YEAR))

MOBI_ZONES_SHP = r'\\Filer16L\P-V160L\SIMBA.A11244\10_Daten\72_NPVM\2016\20_NPVM2016_Daten\02-LV95_NPVM_7978_Verkehrszonen_20171120\NPVM_Verkehrszonen.shp'

SAVE_FIGURES = False

if YEAR == YEAR_IST:
    YEAR_IST = f'{YEAR_IST} (Ref)'
```

+++ {"tags": ["remove-cell"]}

## Loading Data

+++ {"tags": ["remove-cell"]}

###  ZoneId to MS-Region Mapping

```{code-cell} ipython3
:tags: [remove-cell]


zones = gpd.read_file(MOBI_ZONES_SHP)
```

```{code-cell} ipython3
:tags: [remove-cell]


ms_region_mapping = (zones[['ID', 'N_MSR']]
                     .astype({'ID': int})
                     .fillna('None')
                     .rename(columns={'ID':'zone_id', 'N_MSR':'ms_region'})
                     .set_index('zone_id')
                     .iloc[:,0]
                     .to_dict()                
                    )
```

+++ {"tags": ["remove-cell"]}

### SynPop

```{code-cell} ipython3
:tags: [remove-cell]


SYNPOP_BUSINESSES_FILE
```

```{code-cell} ipython3
:tags: [remove-cell]


%%time
synpop_businesses = Businesses(YEAR)
synpop_businesses.load(SYNPOP_BUSINESSES_FILE)
businesses = synpop_businesses.data 

businesses['ms_region'] = businesses['zone_id'].map(ms_region_mapping)
businesses['ms_region'] = businesses['ms_region'].fillna('_Null_')
```

```{code-cell} ipython3
:tags: [remove-cell]


SYNPOP_BUSINESSES_FILE_IST
```

```{code-cell} ipython3
:tags: [remove-cell]


%%time
synpop_businesses_ist = Businesses(YEAR_IST)
synpop_businesses_ist.load(SYNPOP_BUSINESSES_FILE_IST)
businesses_ist = synpop_businesses_ist.data 

businesses_ist['ms_region'] = businesses_ist['zone_id'].map(ms_region_mapping)
businesses_ist['ms_region'] = businesses_ist['ms_region'].fillna('_Null_')
```

+++ {"tags": ["remove-cell"]}

## Analysis

```{code-cell} ipython3
:tags: [remove-cell]


businesses.head()
```

```{code-cell} ipython3
:tags: [remove-cell]


businesses_ist.head()
```

+++ 

## Global Counts

```{code-cell} ipython3
:tags: [remove-cell]


def compute_stats_summary(df_ist, df_scenario, year_ist, year_scenario):
    stats_list = []
    for df, year in zip((df_ist, df_scenario), (year_ist,  year_scenario)):
        stats = (df.agg({'business_id': 'count',
                         'jobs_endo':sum, 
                         'jobs_exo':sum,
                         'fte_endo':sum,
                         'fte_exo':sum}
                        )
                   .rename(index={'business_id': 'businesses'})
                   .astype(int)
                ) 
        stats.loc['total_jobs'] = stats.loc['jobs_endo'] + stats.loc['jobs_exo']
        stats.loc['total_fte'] = stats.loc['fte_endo'] + stats.loc['fte_exo'] 
        stats = stats.rename(year)
        stats_list.append(stats)
    
    stats_summary = pd.concat(stats_list, axis=1)
    stats_summary['% change'] = ((stats_summary[year_scenario] - stats_summary[year_ist]) / stats_summary[year_scenario] * 100).astype(int)

    return stats_summary
```

+++ {"tags": ["remove-cell"]}

**Businesses & Schools**

```{code-cell} ipython3
:tags: [remove-cell]


businesses.head(0)
```

```{code-cell} ipython3
:tags: [remove-cell]


stats_summary_with_schools = compute_stats_summary(businesses_ist, businesses, YEAR_IST, YEAR)
stats_summary_with_schools
glue("business_with_schools", stats_summary_with_schools)
```

+++ {"tags": ["remove-cell"]}

**Businesses without Schools**

```{code-cell} ipython3
:tags: [remove-cell]


stats_summary = compute_stats_summary(businesses_ist.query('school_type == "no_school"'),
                                      businesses.query('school_type == "no_school"'),
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
    stats_list = []
    for df, year in zip((df_ist, df_scenario), (year_ist,  year_scenario)):
        stats = (df.query('school_type != "no_school"')
                   .groupby('school_type', observed=True)['business_id'].count()
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


title = 'Businesses - Number of Schools - SynPop{year_ist} vs. SynPop{year}'.format(year_ist=YEAR_IST, year=YEAR)

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

## By Sector

```{code-cell} ipython3
:tags: [remove-cell]

businesses_ist.columns
```

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
title = ('Businesses - {statistic} per {category} - SynPop{year_ist} vs. SynPop{year}'
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
title = ('Businesses - {statistic} per {category} - SynPop{year_ist} vs. SynPop{year}'
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
title = ('Businesses - {statistic} per {category} - SynPop{year_ist} vs. SynPop{year}'
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

+++ {"tags": ["remove-cell"]}

### By Canton

```{code-cell} ipython3
:tags: [remove-cell]


stats_per_canton = visualisations.get_businesses_comparison_summary_by_category(businesses_ist, businesses, 
                                                                                YEAR_IST, YEAR, 
                                                                                query='school_type == "no_school"', 
                                                                                agg_column='kt_name'
                                                                                )
```

```{code-cell} ipython3
:tags: [remove-cell]


stats_per_canton
```

+++ {"tags": ["remove-cell"]}

### By MS-Region

```{code-cell} ipython3
:tags: [remove-cell]


stats_per_msr = visualisations.get_businesses_comparison_summary_by_category(businesses_ist, businesses, 
                                                                             YEAR_IST, YEAR, 
                                                                             query='school_type == "no_school"', 
                                                                             agg_column='ms_region'
                                                                             )
```

+++ {"tags": ["remove-cell"]}

**Total Businesses**

```{code-cell} ipython3
:tags: [remove-cell]


stats_per_msr.query('statistic == "total_businesses"').reset_index(level=1, drop=True).sort_values(YEAR, ascending=False)
```

+++ {"tags": ["remove-cell"]}

**Total Jobs**

```{code-cell} ipython3
:tags: [remove-cell]


stats_per_msr.query('statistic == "total_jobs"').reset_index(level=1, drop=True).sort_values(YEAR, ascending=False)
```

+++ {"tags": ["remove-cell"]}

**Total Full Time Equivalent**

```{code-cell} ipython3
:tags: [remove-cell]


stats_per_msr.query('statistic == "total_fte"').reset_index(level=1, drop=True).sort_values(YEAR, ascending=False)
```

+++ {"tags": ["remove-cell"]}

### Create Jobs Report (exel document) 

```{code-cell} ipython3
:tags: [remove-cell]


aggregate_by = ['ms_region', 'kt_name', 'sector']
```

```{code-cell} ipython3
:tags: [remove-cell]

# for report_year, df in zip([YEAR, YEAR_IST], [businesses, businesses_ist]):
#     excel_file_name ='11-SynPop{}_jobs_report.xlsx'.format(report_year, statistic)
#     excel_file_path = os.path.join(FIG_OUTPUT_DIR, excel_file_name)
#     with pd.ExcelWriter(excel_file_path) as writer:
#         for agg_column in aggregate_by:
#             stats = (df.groupby(agg_column).agg({'business_id': 'count',
#                                                  'jobs_endo': sum,
#                                                  'jobs_exo': sum,
#                                                  'fte_endo': sum,
#                                                  'fte_exo': sum}
#                                                 )
#                  .rename(columns={'business_id': 'total_businesses'})
#                  .astype(int)
#                  )
#             stats['total_jobs'] = stats['jobs_endo'] + stats['jobs_exo']
#             stats['total_fte'] = stats['fte_endo'] + stats['fte_exo']
#             stats = stats[['total_businesses', 'jobs_endo', 'jobs_exo', 'total_jobs',
#                           'fte_endo', 'fte_exo', 'total_fte']]
#             stats.to_excel(writer, sheet_name=agg_column)
```

```{code-cell} ipython3
:tags: [remove-cell]



```
