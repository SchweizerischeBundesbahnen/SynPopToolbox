# mobi-synpop

A set of tools to analyse, modify or extend the Swiss synthetic population.
 
## Executive summary 
mobi-synpop is a Python framework developed by the transport & mobility modelling team at the Swiss Federal 
Railways (SBB).
It offers tools to pre-process, visualise and modify a synthetic population of Switzerland.

This synthetic population for 2017 has been established through a joint effort from SBB and the Federal Office for 
Spatial Development ARE for transport and land-use modelling purposes. The project and applied methods are documented 
in a technical report (www.are.admin.ch/flnm).

For privacy reasons, the synthetic population that mobi-synpop wraps itself around, is NOT publicly available. 
At this stage, mobi-synpop is only intended for the teams that have access to the raw synthetic population data. 

The Synthetic population aggregated to person groups and transport zones level is, however, available 
(more information (https://forsbase.unil.ch/project/study-public-overview/16340/0/).


This project is a Python framework to interact with this synthetic population. 
The core functionality are:

1. Pre-Processing the raw population files into optimised pandas DataFrames.
2. Visualisations to check the quality of the synthetic population product.  
3. A framework to change some of the agent's attribute in order fit some given control totals.   

Where to start:
* The Jupyter notebooks under `mobi-synpop/tools/*` serve as a UI for this project. 
New users should start there. 
 
## Getting Started
### Installing the environment
This project uses Python 3.7

The environment can be set up using a conda environment. 
Miniconda, can be downloaded from https://docs.conda.io/en/latest/miniconda.html.

If the environment mobi37 is not already available, run in a terminal:  
```
conda env create -f mobi37.yml 
```

### Cloning the project

In a normal terminal run: 
```
git clone https://code.sbb.ch/scm/simba/mobi-synpop.git
git checkout master
git pull
```

If this does not work out work out, you might need to use the new secured connections.
A token can be generated here: https://code.sbb.ch/plugins/servlet/ssh/projects/SIMBA/repos/mobi-synpop/keys
```
git clone https://USER:PASSWORD@/code.sbb.ch/scm/simba/mobi-synpop.git
OR
git clone https://USER:TOKEN@/code.sbb.ch/scm/simba/mobi-synpop.git
```

### Starting a the UI (Jupyter notebook tool) 
Once the project has been cloned and updated, the user can copy the Jupyter notebooks of
 interest (`tools/*`) into a project directory. 

Then start a normal terminal in the project repository and start Jupyter: 
```
conda activate synpop
jupyter notebook
```

## Functionality
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

Note: the current quality check notebooks have been developed for future years. 
It is possible that for 2017, these tools have to be adapted slightly. 
There should be all the available building blocks in `synpop` package.  

### Marginal Fitting
The fitting methodology consists forcing control total in subgroups of the
 population by randomly picking and modifying agents until the conditions are met.

The random pick can be uniform or weighted based on a priory probabilistic 
model which favors likely feature combinations. 

When no published control totals are available, the structure of the 
synthetic population of 2017 has been used as a proxy objective. This allows for fairer comparisons to be drawn between the 2017 and future model.  

The following synthetic population attributes are fitted:
* Languages
* Nationality
* Current Education
* Current Job Rank 
* Level of Employment 
* Businesses (Jobs & FTEs)

## Updating GitHub Project
* Remove any hard-coded proxy paths
* Remove all filer links (CTRL + SHIFT + R)
* Remove email addresses 
* Delete Jenkins config file
* Checkout new orphan branch: `git checkout --orphan github_master`
* Push to GitHub: 
```
git remote add github https://github.com/SchweizerischeBundesbahnen/SynPopToolbox.git
git push -u github github_master
```

**Note**: 

For GitHub, the SBB proxy address must be SET:
```
git config --global http.proxy http://PROXY_ADDRESS:PORT
git config --global https.proxy https://PROXY_ADDRESS:PORT
```  

For BitBucket, the SBB proxy address must be UNSET:
```
git config --unset --global http.proxy
git config --unset --global https.proxy
```  

It is possible to set the proxy address only for GitHub like this:
```
git config --global http.https://github.com.proxy http://PROXY_ADDRESS:PORT
```
## Licence
GNU GENERAL PUBLIC LICENSE Version 2

## Authors
`mobi-synpop` was written by `Raphaël Lüthi <raphael.luethi2@sbb.ch>` 
with part of the code adapted from the work of `Denis Métrailler <denis.metrailler@sbb.ch>`.
Support and maintenance is currently provided by `Davi Guggisberg <davi.guggisberg@sbb.ch>`.
