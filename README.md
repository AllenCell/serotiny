# serotiny

[![Build Status](https://github.com/AllenCellModeling/serotiny/workflows/Build%20Main/badge.svg)](https://github.com/AllenCellModeling/serotiny/actions)
[![Documentation](https://github.com/AllenCellModeling/serotiny/workflows/Documentation/badge.svg)](https://AllenCellModeling.github.io/serotiny/)
[![Code Coverage](https://codecov.io/gh/AllenCellModeling/serotiny/branch/main/graph/badge.svg)](https://codecov.io/gh/AllenCellModeling/serotiny)

Library and commands for pytorch/lightning deep-learning workflows

---

Serotiny is essentially two things:

* A place to gather all of the useful image/data processing, network construction and model training code in one place, to generalize it and provide a clean interface to its functionality.
* A set of commands built out of this functionality that are intended to be invoked in deep-learning workflows. 

The deep-learning functionality is built on [pytorch](https://github.com/pytorch/pytorch) and [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning). 

## Features

- Provides an array of commonly-used functionality for pytorch/pytorch-lightning based deep learning tasks and associated data processing tasks. 
- Structured as a set of modular "steps" that act as commands which can be easily assembled into a larger machine learning pipeline, including steps that:
  - Split data into training, validation, and test sets
  - Apply one-hot encoding to class labels
  - Change resolution of input images
  - Apply 2D projections for 3D input images
  - Filter data down to specified columns
  - Train models

## Installation

**Stable Release:** `pip install serotiny`<br>
**Development Head:** `pip install git+https://github.com/AllenCellModeling/serotiny.git`

## Documentation

For full package documentation please visit [AllenCellModeling.github.io/serotiny](https://AllenCellModeling.github.io/serotiny).

## Quick Start

### To load a trained model:

,
```python
from serotiny.models.zoo import get_model, get_trainer_at_checkpoint, _get_checkpoint

model = get_model(model_path, model_zoo_path)
trainer = get_trainer_at_checkpoint(model_path, model_zoo_path)
ckpt_path, _, config  = _get_checkpoint(model_path, model_zoo_path)
``` 

### To setup a datamodule given the config from a trained model:

,
```python
datamodule_name = config["datamodule"]['^init']
datamodule_config = config["datamodule"]   
create_datamodule = module_get(datamodules, datamodule_name.split('.')[-1])
datamodule_config["num_workers"] = 2
datamodule_config["pin_memory"] = False
datamodule_config.pop('^init')
datamodule = create_datamodule(**datamodule_config)
datamodule.setup()
``` 


<!-- ### To change the resolution of input images:

```python
python -m  serotiny.steps.change_resolution \
    --manifest_in "data/manifest_merged.csv" \
    --path_3d_column "CellImage3DPath" \
    --manifest_out "/allen/aics/modeling/spanglry/data/mitotic-classifier/sampled_output/manifest.csv" \
    --path_3d_resized_column "CellSampledImage3DPath" \
    --path_out "/allen/aics/modeling/spanglry/data/mitotic-classifier/sampled_output/" \
    --resolution [10,20,50]
``` -->

<!-- ### To apply 2D projections to 3D images:

```python
python -m serotiny.steps.apply_projection \
    --dataset_path "data/manifest_merged.csv" \
    --output_path "data/projection.csv" \
    --projection "{'channels': ['membrane', 'structure', 'dna'], \
                   'masks': {'membrane': 'membrane_segmentation', 'dna': 'nucleus_segmentation'}, \
                   'axis': 'Y', 'method': 'max', \
                   'output': '/allen/aics/modeling/spanglry/data/mitotic-classifier/projections/'}" \
    --path_3d_column "CellImage3DPath" \
    --chosen_projection "Chosen2DProjectionPath" \
    --chosen_class "ChosenMitoticClass" \
    --label "Draft mitotic state resolved"
```

### To train a model:

```python
python -m serotiny.steps.train_model \
    --datamodule 'ACTK2DDataModule' \
    --datasets_path 'data/draft-mitotic-state/split/Z.mean.membrane_segmentation.nucleus_segmentation.brightfield.brightfield-membrane_segmentation/' \
    --output_path 'data/draft-mitotic-state/models/basic/Z.mean.membrane_segmentation.nucleus_segmentation.brightfield.brightfield-membrane_segmentation/membrane_segmentation-nucleus_segmentation/adam' \
    --data_config '{"classes": ["M0", "M1/M2", "M3", "M4/M5", "M6/M7"], "channel_indexes": ["membrane_segmentation", "nucleus_segmentation"], "id_fields": ["CellId", "CellIndex", "FOVId"], "channels": ["membrane_segmentation", "nucleus_segmentation", "brightfield"], "projection_path"
: "data/draft-mitotic-state/projections/Z.mean.membrane_segmentation.nucleus_segmentation.brightfield.brightfield-membrane_segmentation.csv"}' \
    --model 'basic' \
    --batch_size 10 \
    --num_gpus 1 \
    --num_workers 20 \
    --num_epochs 1 \
    --lr 0.001 \
    --optimizer 'adam' \
    --scheduler 'reduce_lr_plateau' \
    --tune_bool False \
    --test True \
    --x_label 'projection_image' \
    --y_label 'ChosenMitoticClass'
``` -->

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

![SEROTINY](https://github.com/AllenCellModeling/serotiny/blob/master/resources/serotiny.png)

Serotiny (n) - when fire triggers the release of a seed

**MIT license**
