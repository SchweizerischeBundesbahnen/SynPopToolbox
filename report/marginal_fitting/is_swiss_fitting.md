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

# Nationality Fitting

+++ {"tags": ["remove-cell"]}

**Fitting notebooks are only for reporting, not for interactive use as the Notebooks from the other two chapters.**

```{code-cell} ipython3
:tags: [remove-cell]

import logging
from pathlib import Path
import sys

import pandas as pd
from IPython.display import Image
```

```{code-cell} ipython3
:tags: [remove-cell]

import yaml
from myst_nb import glue

# load config and synpop code
with open('../_config.yml', 'r') as f:
    config = yaml.safe_load(f)['synpop_report_config']

sys.path.append(config['codebase'])

from synpop import visualisations
```

```{code-cell} ipython3
:tags: [remove-cell]

YEAR_IST = config['reference_year']
YEAR = config['target_year']

if YEAR == YEAR_IST:
    YEAR_IST = f'{YEAR_IST} (Ref)'

fitting_path = Path(config['fitting_outputs'])
target_variable = 'is_swiss'
```

```{code-cell} ipython3
:tags: [remove-cell]

# load the summary tables
summary_table_fixed = pd.read_csv(fitting_path / target_variable / 'summary_table_fixed.csv', index_col=target_variable)
summary_table_raw = pd.read_csv(fitting_path / target_variable / 'summary_table_raw.csv', index_col=target_variable)
```

## Global Counts

### Original SynPop

```{code-cell} ipython3
:tags: [remove-input]

visualisations.generate_simple_grid_table(summary_table_raw)
```

### Fitted SynPop

```{code-cell} ipython3
:tags: [remove-input]

visualisations.generate_simple_grid_table(summary_table_fixed)
```

## By Age

### Original SynPop ({glue:text}`year`)

```{code-cell} ipython3
:tags: [remove-input]
Image(filename=fitting_path / target_variable / f"SynPop{YEAR}-Raw_{target_variable}_by_age.png") 
```

### Fitted SynPop ({glue:text}`year`)

```{code-cell} ipython3
:tags: [remove-input]
Image(filename=fitting_path / target_variable / f"SynPop{YEAR}_{target_variable}_by_age.png") 
```

### Reference SynPop ({glue:text}`year_ist`)

```{code-cell} ipython3
:tags: [remove-input]
Image(filename=fitting_path / target_variable / f"SynPop{YEAR_IST}_{target_variable}_by_age.png") 
```

## By Canton

```{code-cell} ipython3
:tags: [remove-cell]

# get figures into IPython

glue("is_swiss_abs_diff_kt_prog_raw", Image(filename=fitting_path / target_variable / f"SynPop{YEAR}-Raw_{target_variable}_by_Canton_absolute.png"), display=False)
glue("is_swiss_rel_diff_kt_prog_raw", Image(filename=fitting_path / target_variable / f"SynPop{YEAR}-Raw_{target_variable}_by_Canton_relative.png"), display=False)

glue("is_swiss_abs_diff_kt_prog", Image(filename=fitting_path / target_variable / f"SynPop{YEAR}_{target_variable}_by_Canton_absolute.png"), display=False)
glue("is_swiss_rel_diff_kt_prog", Image(filename=fitting_path / target_variable / f"SynPop{YEAR}_{target_variable}_by_Canton_relative.png"), display=False)

glue("is_swiss_abs_diff_kt_ref", Image(filename=fitting_path / target_variable / f"SynPop{YEAR_IST}_{target_variable}_by_Canton_absolute.png"), display=False)
glue("is_swiss_rel_diff_kt_ref", Image(filename=fitting_path / target_variable / f"SynPop{YEAR_IST}_{target_variable}_by_Canton_relative.png"), display=False)
```

### Original SynPop ({glue:text}`year`)

````{tabbed} Absolute differences
```{glue:figure} is_swiss_abs_diff_kt_prog_raw
:figwidth: 800px
:align: center
```
````

````{tabbed} Relative differences
```{glue:figure} is_swiss_rel_diff_kt_prog_raw
:figwidth: 800px
:align: center
```
````

### Fitted SynPop ({glue:text}`year`)

````{tabbed} Absolute differences
```{glue:figure} is_swiss_abs_diff_kt_prog
:figwidth: 800px
:align: center
```
````

````{tabbed} Relative differences
```{glue:figure} is_swiss_rel_diff_kt_prog
:figwidth: 800px
:align: center
```
````
