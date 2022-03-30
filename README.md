# serotiny

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

Check our [getting started](https://allencellmodeling.github.io/serotiny/getting_started.html)
page to get started, and learn how training a DL model can be as simple as:

``` sh

$ serotiny train data=my_dataset model=my_model

```
