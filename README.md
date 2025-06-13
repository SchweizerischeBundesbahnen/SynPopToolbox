# SynPopToolbox

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Poetry](https://img.shields.io/badge/package%20manager-Poetry-blue)](https://python-poetry.org/)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![Version](https://img.shields.io/badge/version-24.1.113-green.svg)](https://github.com/SchweizerischeBundesbahnen/SynPopToolbox)

A Python framework for analyzing, visualizing, and modifying the Swiss synthetic population.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Availability](#data-availability)
- [License](#license)
- [Contributors](#contributors)

## Overview

SynPopToolbox is a Python framework developed by the transport & mobility modelling team at the Swiss Federal Railways (SBB).

The synthetic population for 2022 was established through a joint effort from SBB and the Federal Office for Spatial Development (ARE) for transport and land-use modelling purposes. The project and applied methods are documented in a [technical report](https://www.are.admin.ch/flnm).

## Data Availability

**Important:** For privacy reasons, the synthetic population that SynPopToolbox wraps around is NOT publicly available. This toolbox is intended only for teams with authorized access to the raw synthetic population data.

An anonymized version of the SynPop is publicly available through [SWISS Ubase](https://www.swissubase.ch/en/catalogue/studies/20931/20576/datasets/2753/3470/overview).

The synthetic population aggregated to person groups and transport zones level is available through [SWISS Ubase](https://www.swissubase.ch/en/catalogue/studies/20931/20576/datasets/2753/3470/overview).

## Features

The core functionality includes:

1. **Pre-Processing** - Convert raw population files into optimized pandas DataFrames
   - Load raw data with configurable feature selection
   - Rename uninformative column names
   - Optimize memory usage with boolean and categorical types
   - Join person, household, and zone tables for comprehensive analysis

2. **Visualizations** - Analyze key population attributes:
   - Population counts by canton
   - Mobility tool ownership
   - Age structure distributions
   - Employment levels
   - Household structure
   - Business characteristics

3. **Marginal Fitting** - Adjust agent attributes to match control totals:
   - Random or weighted agent selection
   - Statistical adjustments to align with target distributions
   - Support for future year projections based on 2017 baseline

4. **Interactive Tools** - Streamlit & Jupyter-based interfaces for exploration

## Installation

SynPopToolbox requires Python 3.11 and uses Poetry for dependency management.

### Using Poetry (recommended)

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/SchweizerischeBundesbahnen/SynPopToolbox.git
cd SynPopToolbox

# Install dependencies
poetry install
```

### Using pip

```bash
pip install git+https://github.com/SchweizerischeBundesbahnen/SynPopToolbox.git
```

## Usage

### Command Line Interface

Most functionality is available through the CLI:

```bash
synpop --help
```

### Key Dependencies

- **Data Processing**: pandas 2.x, numpy, pyarrow, fastparquet
- **Visualization**: matplotlib 3.8.0, seaborn, altair 4.x
- **Geographic**: geopandas, shapely 2.0.2
- **Interactive**: streamlit, ipywidgets, voila, jupyter-book

## License

GNU GENERAL PUBLIC LICENSE Version 2

## Contributors

`SynPopToolbox` was originally written by `Raphaël Lüthi <raphael.luethi2@sbb.ch>` 
with part of the code adapted from the work of `Denis Métrailler <denis.metrailler@sbb.ch>`.
Further development, as well as Support and maintenance, is currently undertaken by `Davi Guggisberg <davi.guggisberg@sbb.ch>`.
