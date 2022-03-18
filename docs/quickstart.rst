Quickstart
==========

Train a model
*************

(Assuming one YAML file containing all the configs)

::

   $ serotiny model train \
   --model=config.yaml:model \
   --datamodule=config.yaml:datamodule \
   --trainer=config.yaml:trainer \
   --model_zoo=config.yaml:model_zoo \



Load a trained model
********************

::

   from serotiny.models.zoo import get_model, get_trainer_at_checkpoint, _get_checkpoint

   model = get_model(model_path, model_id, model_zoo_path)
