#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""
import setuptools
import xml.etree.ElementTree as ETree


with open("README.md") as readme_file:
    readme = readme_file.read()

version = "0.0.1"
tree = ETree.parse(r'pom.xml')
root = tree.getroot()
for child in root:
    if "version" in child.tag:
        version = child.text

requirements = []
with open('requirements.txt') as f:
    requirements = [l for l in f.read().splitlines()
                    if not l.startswith("git")]

setuptools.setup(
    name="mobi-synpop",
    version=version,
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    # https://setuptools.readthedocs.io/en/latest/setuptools.html
    dependency_links=[
        # workaround: we cannot extract the --index-url entries from requirements.txt since
        # - pip does not work with the library name (tf-object-detection) at the end of the URL
        # - setup.py (which uses easy_install) needs the library name (tf-object-detection) at the end of the URL
    ],
    include_package_data=True,
    zip_safe=False,
    #entry_points={  # TODO: entry point to create some kind of synpop report would be quite useful
    #    'console_scripts': [
    #        'matsimba=analyse.reporting_scripts.cli:matsimba'  # entry point for CLI interface
    #    ],
    #},
    extras_require={'full': requirements},
    classifiers=(
            "Programming Language :: Python :: 3.7",
            "Operating System :: Windows",
    )
)