import pytest
from serotiny_steps.train_vae import train_vae

def test_vae_step():
    for dimensionality in [2, 3]:
        try:
            train_vae(
                data_dir="/tmp",
                output_path="/tmp",
                datamodule="DummyDatamodule",
                batch_size=4,
                num_gpus=1,
                num_workers=4,
                num_epochs=1,
                lr=1e-3,
                optimizer_encoder="Adam",
                optimizer_decoder="Adam",
                crit_recon="BatchMSELoss",
                test=True,
                x_label="dummy_x",
                class_label="dummy_y",
                n_latent_dim=32,
                n_classes=10,
                activation="relu",
                activation_last="sigmoid",
                conv_channels_list=[8, 16, 32],
                input_dims=[32]*dimensionality,
                target_channels=[1, 2],
                reference_channels=[0],
                beta=0.5,
                dimensionality=dimensionality,
                length=20, #kwarg for dummy datamodule
                dims=[32]*dimensionality, #kwarg for dummy datamodule
                channels=[0, 0, 0] #kwarg for dummy datamodule
            )
        except Exception as e:
            pytest.fail(f"`train_vae` failed to run.")

if __name__ == "__main__":
    test_vae_step()
