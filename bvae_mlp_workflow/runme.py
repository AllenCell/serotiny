import os
from omegaconf import DictConfig, OmegaConf
import hydra
from serotiny_steps.train_mlp_vae import train_mlp_vae


@hydra.main(config_path="conf", config_name="config")
def runme(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"{os.getcwd()}/results/")
    print(cfg["network"]["hidden_layers"])

    train_mlp_vae(
        source_path=cfg["data"]["dir"]["path"],
        modified_source_save_dir=cfg["data"]["modified_source_save_dir"],
        output_path=f"{os.getcwd()}/results/",
        datamodule=cfg["data"]["datamodule"],
        batch_size=int(cfg["training"]["batch_size"]),
        gpu_id=cfg["training"]["gpu_id"],
        num_workers=int(cfg["training"]["num_workers"]),
        num_epochs=int(cfg["training"]["num_epochs"]),
        lr=float(cfg["training"]["lr"]),
        optimizer=cfg["training"]["optimizer"],
        scheduler=cfg["training"]["scheduler"],
        x_label=cfg["data"]["input"]["x_label"],
        c_label=cfg["data"]["condition"]["c_label"],
        c_label_ind=cfg["data"]["condition"]["c_label_ind"],
        x_dim=int(cfg["data"]["input"]["x_dim"]),
        c_dim=int(cfg["data"]["condition"]["c_dim"]),
        hidden_layers=list(cfg["network"]["hidden_layers"]),
        latent_dims=int(cfg["network"]["latent_dims"]),
        beta=float(cfg["network"]["beta"]),
        cvapipe_analysis_config_path=cfg["cvapipe_analysis_config"]["path"],
        set_zero=cfg["data"]["set_zero"]["bool"],  # kwarg for spharm
        overwrite=False,  # kwarg for spharm
        id_fields=cfg["data"]["id_fields"],  # kwarg for spharm
        values=list(cfg["callbacks"]['values']),
        latent_walk_range = list(cfg["callbacks"]['latent_walk_range']),
        n_cells = int(cfg["callbacks"]['n_cells']),  # No of closets cells to find per location
    )

    print("Done!")


if __name__ == "__main__":
    runme()