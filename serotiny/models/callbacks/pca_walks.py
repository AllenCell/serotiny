from pathlib import Path

from pytorch_lightning import Callback, LightningModule, Trainer
from cvapipe_analysis.steps.shapemode.shapemode_tools import ShapeModeCalculator
from cvapipe_analysis.tools import controller


class PCAWalks(Callback):
    def __init__(self, df, config):
        self.df = df
        self.config = config

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        dir_path = Path(trainer.logger[1].save_dir)

        subdir = dir_path / "pca_walks"
        subdir.mkdir(parents=True, exist_ok=True)

        subdir_necessary_for_cvapipe_analysis = subdir / "shapemode/avgshape"
        subdir_necessary_for_cvapipe_analysis.mkdir(parents=True, exist_ok=True)

        self.config["project"][
            "local_staging"
        ] = subdir  # This is where plots get saved

        control = controller.Controller(self.config)
        calculator = ShapeModeCalculator(control)
        calculator.set_data(self.df)
        calculator.space.execute(calculator.df)
        calculator.compute_shcoeffs_for_all_shape_modes()
        calculator.recontruct_meshes()
        calculator.generate_and_save_animated_2d_contours()
        calculator.plot_maker_sm.combine_and_save_animated_gifs()
