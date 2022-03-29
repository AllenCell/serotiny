#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_namespace_packages, find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as reqs_file:
    requirements = reqs_file.read().strip().splitlines()

docs_requirements = [
    "sphinx",
    "furo",
    "m2r2",
]

setup(
    author="Ryan Spangler, Ritvik Vasan, Guilherme Pires, Caleb Chan, Theo Knijnenburg",
    author_email=(
        "ryan.spangler@alleninstitute.org, ritvik.vasan@alleninstitute.org, "
        "guilherme.pires@alleninstitute.org, caleb.chan@alleninstitute.org, "
        "theo.knijnenburg@alleninstitute.org"
    ),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="library and commands for deep learning workflows",
    entry_points={
        "console_scripts": [
            "serotiny=serotiny.cli.cli:main",
            "serotiny.train=serotiny.cli.cli:main",
            "serotiny.test=serotiny.cli.cli:main",
            "serotiny.predict=serotiny.cli.cli:main",
        ],
    },
    install_requires=requirements + docs_requirements,
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="serotiny",
    name="serotiny",
    packages=find_namespace_packages(include=["hydra_plugins.*"])
    + find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.7",
    url="https://github.com/AllenCellModeling/serotiny",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.0.a0",
    zip_safe=False,
)
