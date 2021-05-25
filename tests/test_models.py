import yaml
import tempfile

from pathlib import Path

from serotiny_steps.train_model import train_model

def test_models():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmppath = Path(tmpdirname)
        main_config = yaml.load(f"""
        training:
          callbacks:
            pytorch_lightning.callbacks.ProgressBar:
              refresh_rate: 2

          loggers:
            pytorch_lightning.loggers.TensorBoardLogger:
              save_dir: "{tmppath / 'tboard_logs'}"

          gpu_ids: [5]
          trainer_config:
            max_epochs: 1

        model_zoo_config:
          path: "{tmppath / 'model_zoo'}"
          store_config: True
          checkpoint_monitor: "val_loss"
          checkpoint_mode: "min"
        """)

        model_configs_paths = Path("./model_configs").glob("*.yaml")
        for config in model_configs_paths:
            with open(config) as f:
                config = yaml.safe_load(f)
                model_config = config["model"]
                datamodule_config = config["datamodule"]

                train_model(
                    model_name=model_config["name"],
                    model_config=model_config["config"],
                    datamodule_name=datamodule_config["name"],
                    datamodule_config=datamodule_config["config"],
                    model_zoo_config=main_config["model_zoo_config"],
                    **main_config["training"],
                )

if __name__ == "__main__":
    test_models()
