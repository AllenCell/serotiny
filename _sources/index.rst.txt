serotiny
========

``serotiny`` is a framework and set of tools to structure, configure and drive deep
learning projects. It integrates `hydra <https://hydra.cc>`_ for configuration and
execution, `Pytorch Lightning <https://pytorchlightning.ai>`_
for neural net training/testing/prediction, and `MLFlow <https://mlflow.org>**_
for experiment tracking and artifact management.

Intro
*****

``serotiny`` was developed with the intention of streamlining the lifecycle of
Deep Learning projects at `AICS <https://www.allencell.org/>`_ . It achieves this
goal by:

- Standardizing the structure of DL projects
- Relying on the modularity afforded by this standard structure to make DL projects highly
  configurable, using `hydra <https://hydra.cc>`_ as the framework for configuration
- Making it easy to adopt best-practices by tightly integrating with MLFlow for
  experiment tracking and artifact management

In doing so, DL projects become reproducible, easy to collaborate on and can
benefit from general tooling.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Contents:

   Home <self>
   installation
   getting_started
   cli
   dataframe_transforms
   Package modules <modules>
