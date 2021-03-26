import pytest
from serotiny_steps.train_unet import train_unet

def test_unet_step():
    input_channels = ['dna']
    output_channels = ['nucleus_segmentation']
    try:
        train_unet(
            data_dir="/allen/aics/modeling/VariancePlayground/manifests/",
            output_path="./tmp",
            #datamodule="DummyImageDatamodule",
            datamodule="ImageImage",
            batch_size=1,
            num_gpus=[1],
            num_workers=4,
            id_fields=['CellId', 'CellIndex', 'FOVId'],
            num_epochs=1,
            lr=1e-3,
            optimizer="Adam",
            loss="BatchMSELoss",
            test=True,
            x_label="dummy_x",
            y_label="dummy_y",
            input_column="actk_rawseg",
            output_column="actk_rawseg",
            input_channels=input_channels,
            output_channels=output_channels,
            depth=3,  # Unet-specific
            auto_padding=True,  # Unet-specific
            
            # TODO: Think clearly about the relationship between input/output_dims and
            #       input/output_channels
            length=1, #kwarg for dummy image datamodule
            #input_dims=[572, 572, 60], #kwarg for dummy image datamodule
            #output_dims=[572, 572, 60], #kwarg for dummy image datamodule
            input_dims=[36, 38, 12], #kwarg for dummy image datamodule
            output_dims=[36, 38, 12], #kwarg for dummy image datamodule
        )
    except Exception as e:
        pytest.fail(f"`train_unet` failed to run.")

if __name__ == "__main__":
    test_unet_step()
