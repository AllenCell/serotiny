# serotiny

[![Build Status](https://github.com/AllenCellModeling/serotiny/workflows/Build%20Main/badge.svg)](https://github.com/AllenCellModeling/serotiny/actions)
[![Documentation](https://github.com/AllenCellModeling/serotiny/workflows/Documentation/badge.svg)](https://AllenCellModeling.github.io/serotiny/)
[![Code Coverage](https://codecov.io/gh/AllenCellModeling/serotiny/branch/main/graph/badge.svg)](https://codecov.io/gh/AllenCellModeling/serotiny)

library and commands for deep learning workflows

![SEROTINY](https://github.com/AllenCellModeling/serotiny/blob/master/resources/serotiny.png)

Serotiny (n) - when fire triggers the release of a seed

---

## Features

-   An array of useful functionality for deep learning with pytorch and pytorch-lightning and associated data processing tasks. 
-   A set of modular "steps" that act as commands which can be assembled into a larger machine learning pipeline. 

## Quick Start

```python
from serotiny.library.image import project_2d

project_2d(
    "path/to/3d/image.ome.tiff",
    "Z",
    "mean",
    "path/to/2d/projection.png",
    channels=['membrane', 'dna', 'brightfield'],
    masks={'membrane': 'membrane_segmentation', 'dna': 'nucleus_segmentation'})
```

## Installation

**Stable Release:** `pip install serotiny`<br>
**Development Head:** `pip install git+https://github.com/AllenCellModeling/serotiny.git`

## Documentation

For full package documentation please visit [AllenCellModeling.github.io/serotiny](https://AllenCellModeling.github.io/serotiny).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

## The Four Commands You Need To Know

1. `pip install -e .[dev]`

    This will install your package in editable mode with all the required development
    dependencies (i.e. `tox`).

2. `make build`

    This will run `tox` which will run all your tests in both Python 3.7
    and Python 3.8 as well as linting your code.

3. `make clean`

    This will clean up various Python and build generated files so that you can ensure
    that you are working in a clean environment.

4. `make docs`

    This will generate and launch a web browser to view the most up-to-date
    documentation for your Python package.

#### Additional Optional Setup Steps:

-   Turn your project into a GitHub repository:
    -   Make an account on [github.com](https://github.com)
    -   Go to [make a new repository](https://github.com/new)
    -   _Recommendations:_
        -   _It is strongly recommended to make the repository name the same as the Python
            package name_
        -   _A lot of the following optional steps are *free* if the repository is Public,
            plus open source is cool_
    -   After a GitHub repo has been created, run the commands listed under:
        "...or push an existing repository from the command line"
-   Register your project with Codecov:
    -   Make an account on [codecov.io](https://codecov.io)(Recommended to sign in with GitHub)
        everything else will be handled for you.
-   Ensure that you have set GitHub pages to build the `gh-pages` branch by selecting the
    `gh-pages` branch in the dropdown in the "GitHub Pages" section of the repository settings.
    ([Repo Settings](https://github.com/AllenCellModeling/serotiny/settings))
-   Register your project with PyPI:
    -   Make an account on [pypi.org](https://pypi.org)
    -   Go to your GitHub repository's settings and under the
        [Secrets tab](https://github.com/AllenCellModeling/serotiny/settings/secrets/actions),
        add a secret called `PYPI_TOKEN` with your password for your PyPI account.
        Don't worry, no one will see this password because it will be encrypted.
    -   Next time you push to the branch `main` after using `bump2version`, GitHub
        actions will build and deploy your Python package to PyPI.


**MIT license**

