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

# Language Fitting

+++ {"tags": ["remove-cell"]}

**Fitting notebooks are only for reporting, not for interactive use as the Notebooks from the other two chapters.**

```{code-cell} ipython3
:tags: [remove-cell]

import logging
from pathlib import Path
import sys

import pandas as pd
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
target_variable = 'language'
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
