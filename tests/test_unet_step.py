import pytest
from serotiny_steps.train_unet import train_unet

def test_unet_step():
    
    try:
        train_unet(
            data_dir="/allen/aics/modeling/VariancePlayground/manifests/",
            output_path="output",
            # output_path="/allen/aics/modeling/caleb/runs/image2image/variance_playground_3d_test/",
            datamodule="DummyImageDatamodule",
            # datamodule="ImageImage",
            batch_size=2,
            num_gpus=[2],
            num_workers=4,
            id_fields=['CellId', 'CellIndex', 'FOVId'],
            num_epochs=10,
            lr=1e-3,
            optimizer="Adam",
            loss="BatchMSELoss",
            test=True,
            x_label="x_label",
            y_label="y_label",
            input_column="actk_rawseg",
            output_column="actk_rawseg",
            input_channels=['dna'],
            output_channels=['nucleus_segmentation'],
            # depth=3,  # Unet-specific
            depth=2,  # Unet-specific
            auto_padding=True,  # Unet-specific
            channel_fan_top=64,
            # kwargs for dummy image data module
            length=10,
            #input_dims=[572, 572, 60],
            #output_dims=[572, 572, 60],
            input_dims=[36, 38, 12],
            output_dims=[36, 38, 12],
        )
        
    except Exception as e:
        pytest.fail(f"`train_unet` failed to run.")

if __name__ == "__main__":
    test_unet_step()
