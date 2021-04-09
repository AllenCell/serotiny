import pytest
from serotiny_steps.train_model import train_model

def test_unet_step():
    
    # batch_size = 8, depth = 3, channel_fan_top = 64 (x)
    # 16-bit precision
    # batch_size = 8, depth = 3, channel_fan_top = 64 (finished 10 epochs) - 1 (version_31-03-2021--14-33-31)
    # batch_size = 8, depth = 4, channel_fan_top = 64 (x)
    # batch_size = 1, depth = 4, channel_fan_top = 64
    # batch_size = 2, depth = 4, channel_fan_top = 64 (x)
    # batch_size = 2, depth = 4, channel_fan_top = 32
    # batch_size = 8, depth = 4, channel_fan_top = 32 (finished 10 epochs) - 2 (version_31-03-2021--17-04-37)
    # batch_size = 16, depth = 4, channel_fan_top = 32 (memory error)
    # batch_size = 16, depth = 4, channel_fan_top = 16 (finished 10 epochs) - 3 (version_01-04-2021--16-22-11)
    
    input_channels = ['dna']
    output_channels = ['nucleus_segmentation']

    x_label = 'x_label'
    y_label = 'y_label'

    training_params = {
        'output_path': "data/output",
        'num_gpus': [0],
        'num_epochs': 1,
        'datamodule': "DummyImageDatamodule",
        'network': "Unet3d",
        'loss': "BatchMSELoss",
        'model': "UnetModel",
        'test': True,
    }

    model_params = {
        'lr': 1e-3,
        'optimizer': "Adam",
        'x_label': x_label,
        'y_label': y_label,
        'input_channels': input_channels,
        'output_channels': output_channels,
        'auto_padding': True,
        'test_image_output': True,
        # TODO: add scheduler to model
        # 'scheduler': "ReduceLROnPlateau",
    }

    loss_params = {}

    network_params = {
        'num_input_channels': len(input_channels),
        'num_output_channels': len(output_channels),
        'depth': 2,  # Unet-specific
        'channel_fan_top': 64,  # Unet-specific
    }

    datamodule_params = {
        'datamodule': "DummyImageDatamodule",
        'data_dir': "data", # "/allen/aics/modeling/VariancePlayground/manifests/",
        'batch_size': 2,
        'id_fields': ['CellId', 'CellIndex', 'FOVId'],
        'num_workers': 4,
        'x_label': x_label,
        'y_label': y_label,
        'input_column': "actk_rawseg",
        'output_column': "actk_rawseg",
        'input_channels': input_channels,
        'output_channels': output_channels,
        # kwargs for dummy image data module
        'length': 10,
        #input_dims=[572, 572, 60],
        #output_dims=[572, 572, 60],
        'input_dims': [36, 38, 12],
        'output_dims': [36, 38, 12],
    }

    try:
        train_model(
            training=training_params,
            model=model_params,
            network=network_params,
            loss=loss_params,
            datamodule=datamodule_params,
        )


        #     data_dir="data", # "/allen/aics/modeling/VariancePlayground/manifests/",
        #     output_path="data/output",
        #     # output_path="/allen/aics/modeling/caleb/runs/image2image/variance_playground_3d_test/",
        #     # datamodule="DummyImageDatamodule",
        #     datamodule=datamodule_params,
        #     # datamodule="ImageImage",
        #     batch_size=2,
        #     id_fields=['CellId', 'CellIndex', 'FOVId'],
        #     num_gpus=[0],
        #     # num_gpus=[2],
        #     num_workers=4,
        #     num_epochs=1,
        #     lr=1e-3,
        #     optimizer="Adam",
        #     scheduler="Adam",
        #     loss="BatchMSELoss",
        #     test=True,
        #     x_label="x_label",
        #     y_label="y_label",
        #     input_column="actk_rawseg",
        #     output_column="actk_rawseg",
        #     input_channels=['dna'],
        #     output_channels=['nucleus_segmentation'],
        #     depth=2,  # Unet-specific
        #     channel_fan_top=64,  # Unet-specific
        #     auto_padding=True,  # Unet-specific
            
        #     # kwargs for dummy image data module
        #     length=10,
        #     #input_dims=[572, 572, 60],
        #     #output_dims=[572, 572, 60],
        #     input_dims=[36, 38, 12],
        #     output_dims=[36, 38, 12],
        # )
        
    except Exception as e:
        pytest.fail(f"`train_unet` failed to run.")

if __name__ == "__main__":
    test_unet_step()
