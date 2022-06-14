# serotiny

While going about the work of building deep learning projects, several simultaneous problems seemed to emerge:

* How do we reuse as much work from previous projects as possible, and focus on building the part of the project that makes it distinct?
* How can we automate the generation of new models that are based on existing models, but vary in a crucial yet non-trivial way?
* When generating a multiplicity of related models, how can we keep all of the results, predictions, and analyses straight?
* How can the results from any number of trainings and predictions be compared and integrated in an insightful yet generally applicable way?

Serotiny arose from the need to address these issues and convert the complexity of deep learning projects into something simple, reproducible, configurable, and automatable at scale.

Serotiny is still a work-in-progress, but as we go along the solutions to these problems become more clear. Maybe you've run into similar situations? We'd love to hear from you.

## Overview

`serotiny` is a framework and set of tools to structure, configure and drive deep
learning projects, developed with the intention of streamlining the lifecycle of
deep learning projects at [Allen Institute for Cell Science](https://www.allencell.org/).

It achieves this goal by:

- Standardizing the structure of DL projects
- Relying on the modularity afforded by this standard structure to make DL projects highly
  configurable, using [hydra](https://hydra.cc) as the framework for configuration
- Making it easy to adopt best-practices and latest-developments in DL infrastructure
  by tightly integrating with
    - [Pytorch Lightning](https://pytorchlightning.ai) for neural net training/testing/prediction
    - [MLFlow](https://mlflow.org) for experiment tracking and artifact management

In doing so, DL projects become reproducible, easy to collaborate on and can
benefit from general and powerful tooling.

## Getting started

For more information, check our [documentation](https://allencell.github.io/serotiny),
or jump straight into our [getting started](https://allencell.github.io/serotiny/getting_started.html)
page, and learn how training a DL model can be as simple as:

``` sh

$ serotiny train data=my_dataset model=my_model

```

## Authors

- Guilherme Pires @colobas
- Ryan Spangler @prismofeverything
- Ritvik Vasan @ritvikvasan
- Caleb Chan @calebium
- Theo Knijnenburg @tknijnen
- Nick Gomez @gomeznick86

## Citing

If you find serotiny useful, please cite this repository as:

```
Serotiny Authors (2022). Serotiny: a framework of tools to structure, configure and drive deep learning projects [Computer software]. GitHub. https://github.com/AllenCellModeling/serotiny
Free software: BSD-3-Clause
```
