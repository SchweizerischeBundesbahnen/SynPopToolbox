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

# Population Fitting

+++ {"tags": ["remove-cell"]}

**Fitting notebooks are only for reporting, not for interactive use as the Notebooks from the other two chapters.**

```{code-cell} ipython3
:tags: [remove-cell]

import logging
from pathlib import Path
import sys
# from IPython.display import display, Markdown  ## use in case of support for multiple layers of fitting

import pandas as pd
```

```{code-cell} ipython3
:tags: [remove-cell]

import yaml
from myst_nb import glue

# load config and synpop code
with open('../_config.yml', 'r') as f:
    config = yaml.safe_load(f)['synpop_report_config']

sys.path.insert(1, config['codebase'])

from synpop import visualisations
```

```{code-cell} ipython3
:tags: [remove-cell]

YEAR_IST = config['reference_year']
YEAR = config['target_year']

if YEAR == YEAR_IST:
    YEAR_IST = f'{YEAR_IST} (Ref)'

fitting_path = Path(config['fitting_output'])
target_variable = 'pop_total'
```

```{code-cell} ipython3

```

```{code-cell} ipython3
:tags: [remove-cell]

# load the summary tables
# load the summary tables
person_path = list(fitting_path.joinpath(target_variable).glob('Persons*'))[0]
household_path = list(fitting_path.joinpath(target_variable).glob('Households*'))[0]
summary_table_persons = pd.read_csv(person_path, index_col=0)
summary_table_households = pd.read_csv(household_path, index_col=0)
```

## Global Counts

+++

### Persons

```{code-cell} ipython3
:tags: [remove-input]

summary_table_persons
```

### Households

```{code-cell} ipython3
:tags: [remove-input]

summary_table_households
```
