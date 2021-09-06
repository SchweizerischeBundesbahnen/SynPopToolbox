# SynPopToolbox

A set of tools to analyse, modify or extend the Swiss synthetic population.
 
## Executive summary 

SynPopToolbox is a Python framework developed by the transport & mobility modelling team at the Swiss Federal 
Railways (SBB).
It offers tools to pre-process, visualise and modify a synthetic population of Switzerland.

The synthetic population for 2017 has been established through a joint effort from SBB and the Federal Office for 
Spatial Development ARE for transport and land-use modelling purposes. The project and applied methods are documented 
in a technical report (www.are.admin.ch/flnm).

For privacy reasons, the synthetic population that SynPopToolbox wraps itself around, is NOT publicly available. 
At this stage, SynPopToolbox is only intended for the teams that have access to the raw synthetic population data.

The Synthetic population aggregated to person groups and transport zones level is, however, available 
(more information (https://forsbase.unil.ch/project/study-public-overview/16340/0/).

A public release of the SynPop in a reduced form, as Open-Government-Data, is planned. 

This project is a Python framework to interact with the current synthetic population. 
The core functionality are:

1. Pre-Processing the raw population files into optimised pandas DataFrames.
2. Visualisations to check the quality of the synthetic population product.  
3. A framework to change some of the agent's attribute in order fit some given control totals.   

## Getting Started

### Installing the environment
This project uses Python 3.7

The environment can be set up using a conda environment. 
Miniconda, can be downloaded from https://docs.conda.io/en/latest/miniconda.html.

If the proper conda environment is not already available, run in a terminal:  

```
conda env create -f environment.yml 
```

### Direct installation with pip

The project may also be installed directly using pip:

`pip install mobi-synpop`

Or if mobi-synpop isn't available, install directly from git with:

`pip install git+https://github.com/SchweizerischeBundesbahnen/SynPopToolbox.git`

If dependencies are missing in your current environment, 
include all dependencies by appending `[full]` to the repository name, such as `mobi-synpop[full]`.

## Functionality

### Running from the Command-Line Interface

Most of the functionality is available directly from the CLI. 
After installing with pip, the following should be possible in the terminal:

`synpop run config.yml`

### Pre-Processing

The ``preprocessing.py`` module contains all the functionality to prepare the
synthetic population data.

All pre-processing configuration are set in on the following config file:

* ``synpop_loading_config.json`` which defines the pre-processing steps for the
  synthetic population files.

The pre-processing steps are the following:

1. Load the raw data file and choose which features to keep (``features_to_load``),
2. Rename some features that uninformative names (``features_to_rename``),
3. Cast some features as boolean to reduce memory requirements (``boolean_features``),
4. Rename categories to informative names and cast these features as categorical to
   reduce memory requirements (``categorical_keys``)

The ``persons table`` is joined to ``households`` and then ``mobi_zones`` to get geographical features
(e.g canton, commune).

The output file is a pickled pandas DataFrame.

### Quality Checks & Visualisations

The following synthetic population attributes:

* Population counts per canton
* Mobility tools
* Age structure
* Level of employment 
* Household structure 
* Businesses

### Marginal Fitting

The fitting methodology consists forcing control total in subgroups of the
 population by randomly picking and modifying agents until the conditions are met.

The random pick can be uniform or weighted based on a priory probabilistic 
model which favors likely feature combinations. 

When no published control totals are available, the structure of the 
synthetic population of 2017 has been used as a proxy objective. 
This allows for fairer comparisons to be drawn between the 2017 and future model.  

## Updating GitHub Project

* Remove any hard-coded proxy paths
* Remove all filer links (CTRL + SHIFT + R)
* Remove email addresses 
* Delete Jenkins config file
* Checkout new orphan branch: `git checkout --orphan github_master`
* Commit using public email: `git commit --author="Name <name.surname@sbb.ch>" -m "Update GitHub"`
* Push to GitHub: 

```
git remote add github https://github.com/SchweizerischeBundesbahnen/SynPopToolbox.git
git push -u github github_master --force
```

## Licence
GNU GENERAL PUBLIC LICENSE Version 2

## Authors
`SynPopToolbox` was originally written by `Raphaël Lüthi <raphael.luethi2@sbb.ch>` 
with part of the code adapted from the work of `Denis Métrailler <denis.metrailler@sbb.ch>`.
Further development, as well as Support and maintenance, 
is currently undertaken by `Davi Guggisberg <davi.guggisberg@sbb.ch>`.
