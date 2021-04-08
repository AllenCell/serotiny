import pytest
from serotiny_steps.train_unet import train_unet

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
    
    try:
        train_unet(
            data_dir="data", # "/allen/aics/modeling/VariancePlayground/manifests/",
            output_path="data/output",
            # output_path="/allen/aics/modeling/caleb/runs/image2image/variance_playground_3d_test/",
            datamodule="DummyImageDatamodule",
            # datamodule="ImageImage",
            batch_size=2,
            id_fields=['CellId', 'CellIndex', 'FOVId'],
            num_gpus=[0],
            # num_gpus=[2],
            num_workers=4,
            num_epochs=10,
            lr=1e-3,
            optimizer="Adam",
            scheduler="Adam",
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
            channel_fan_top=64,  # Unet-specific
            auto_padding=True,  # Unet-specific
            
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
