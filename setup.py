#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

setup_requirements = [
    "pytest-runner>=5.2",
]

test_requirements = [
    "black>=19.10b0",
    "codecov>=2.1.4",
    "flake8>=3.8.3",
    "flake8-debugger>=3.2.1",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bump2version>=1.0.1",
    "coverage>=5.1",
    "ipython>=7.15.0",
    "m2r2>=0.2.7",
    "pytest-runner>=5.2",
    "Sphinx>=3.4.3",
    "sphinx_rtd_theme>=0.5.1",
    "tox>=3.15.2",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

requirements = [
    # project requires
    "aicsimageio>=4.0.2",
    "numpy",
    "pandas",
    "Pillow",
    "pytorch-lightning",
    "pytorch-lightning-bolts",
    "torchvision",
    "tqdm",
    "seaborn",
    "urllib3",
    "brokenaxes",
    "torchio",
    "sphinx",
    "sphinx-rtd-theme",
    "hydra-core@git+https://github.com/facebookresearch/hydra.git",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "all": [*requirements, *dev_requirements],
}

setup(
    author="Ryan Spangler, Ritvik Vasan, Guilherme Pires, Caleb Chan, Theo Knijnenburg",
    author_email="ryan.spangler@alleninstitute.org, ritvik.vasan@alleninstitute.org, guilherme.pires@alleninstitute.org, caleb.chan@alleninstitute.org, theo.knijnenburg@alleninstitute.org",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="library and commands for deep learning workflows",
    entry_points={
        "console_scripts": [
            "apply_projection=serotiny_steps.apply_projection:main",
            "change_resolution=serotiny_steps.change_resolution:main",
            "diagnostic_sheets=serotiny_steps.diagnostic_sheets:main",
            "filter_data=serotiny_steps.filter_data:main",
            "merge_data=serotiny_steps.merge_data:main",
            "one_hot=serotiny_steps.one_hot:main",
            "select_fields=serotiny_steps.select_fields:main",
            "split_data=serotiny_steps.split_data:main",
            "train_classifier=serotiny_steps.train_classifier:main",
            "train_vae=serotiny_steps.train_vae:main",
            "train_model=serotiny_steps.train_model:main",
            "tune_model=serotiny_steps.tune_model:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="serotiny",
    name="serotiny",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.7",
    setup_requires=setup_requirements,
    test_suite="serotiny/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/AllenCellModeling/serotiny",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.0.0",
    zip_safe=False,
)
