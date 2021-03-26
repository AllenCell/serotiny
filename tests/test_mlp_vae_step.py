from serotiny_steps.train_mlp_vae import train_mlp_vae


def test_mlp_vae_step():
    train_mlp_vae(
        data_dir="./test_results3/",
        output_path="./test_results3/",
        datamodule="GaussianDataModule",
        batch_size=64,
        num_gpus=1,
        num_workers=4,
        num_epochs=30,
        lr=1e-3,
        optimizer="adam",
        scheduler="reduce_lr_plateau",
        x_label="gaussian_x",
        c_label="gaussian_c",
        c_label_ind="gaussian_c_ind",
        x_dim=2,
        c_dim=4,
        enc_layers=[2, 256, 256, 64],
        dec_layers=[64, 256, 256, 2],
        beta=1,
        length=64 * 2000,
        corr=False,
    )


if __name__ == "__main__":
    test_mlp_vae_step()
