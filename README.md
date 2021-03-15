# serotiny

[![Build Status](https://github.com/AllenCellModeling/serotiny/workflows/Build%20Main/badge.svg)](https://github.com/AllenCellModeling/serotiny/actions)
[![Documentation](https://github.com/AllenCellModeling/serotiny/workflows/Documentation/badge.svg)](https://AllenCellModeling.github.io/serotiny/)
[![Code Coverage](https://codecov.io/gh/AllenCellModeling/serotiny/branch/main/graph/badge.svg)](https://codecov.io/gh/AllenCellModeling/serotiny)

Pytorch/pytorch-lightning based library and commands for deep learning workflows, modularized as a set of steps to facilitate integration into existing workflow pipelines and packages.

![SEROTINY](https://github.com/AllenCellModeling/serotiny/blob/master/resources/serotiny.png)

Serotiny (n) - when fire triggers the release of a seed

---

## Features

- Provides an array of commonly-used functionality for pytorch/pytorch-lightning based deep learning tasks and associated data processing tasks. 
- Structured as a set of modular "steps" that act as commands which can be easily assembled into a larger machine learning pipeline, including steps that:
  - Split data into training, validation, and test sets
  - Apply one-hot encoding to class labels
  - Change resolution of input images
  - Apply 2D projections for 3D input images
  - Train classifier and autoencoder models
    

## Installation

**Stable Release:** `pip install serotiny`<br>
**Development Head:** `pip install git+https://github.com/AllenCellModeling/serotiny.git`

## Quick Start

### To change the resolution of input images:

```python
python -m  serotiny.steps.change_resolution \
    --manifest_in "data/manifest_merged.csv" \
    --path_3d_column "CellImage3DPath" \
    --manifest_out "/allen/aics/modeling/spanglry/data/mitotic-classifier/sampled_output/manifest.csv" \
    --path_3d_resized_column "CellSampledImage3DPath" \
    --path_out "/allen/aics/modeling/spanglry/data/mitotic-classifier/sampled_output/" \
    --resolution [10,20,50]
```

### To apply 2D projections to 3D images:

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
```

## Documentation

For full package documentation please visit [AllenCellModeling.github.io/serotiny](https://AllenCellModeling.github.io/serotiny).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT license**
