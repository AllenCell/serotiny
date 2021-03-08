import pytest
from serotiny_steps.train_vae import train_vae

def test_vae_step():
    train_vae(
        data_dir="/tmp",
        output_path="/tmp",
        datamodule="DummyDatamodule",
        batch_size=16,
        num_gpus=1,
        num_workers=4,
        num_epochs=1,
        lr=1e-3,
        optimizer_encoder="adam",
        optimizer_decoder="adam",
        crit_recon="BatchMSELoss",
        test=True,
        x_label="dummy_x",
        class_label="dummy_y",
        n_latent_dim=32,
        n_classes=10,
        n_ch_target=2,
        n_ch_ref=1,
        conv_channels_list=[8, 16, 32],
        input_dims=[28, 28],
        target_channels=[1, 2],
        reference_channels=[0],
        beta=0.5,
        is_2d_or_3d=2,
        length=100, #kwarg for dummy datamodule
        dims=[28, 28], #kwarg for dummy datamodule
        channels=[0, 0, 0] #kwarg for dummy datamodule
    )

if __name__ == "__main__":
    test_vae_step()
