import pytest
from serotiny_steps.train_unet import train_unet

def test_unet_step():
    input_channels = [1, 2]
    output_channels = [0]
    try:
        train_unet(
            data_dir="./tmp",
            output_path="./tmp",
            datamodule="DummyImageDatamodule",
            #datamodule="ImageImage",
            batch_size=1,
            num_gpus=[1],
            num_workers=4,
            num_epochs=1,
            lr=1e-3,
            optimizer="Adam",
            loss="BatchMSELoss",
            test=True,
            x_label="dummy_x",
            y_label="dummy_y",
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
