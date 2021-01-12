import importlib.util
import sys

from pytorch_lightning.callbacks.progress import ProgressBarBase

# check if ipywidgets is installed before importing tqdm.auto
# to ensure it won't fail and a progress bar is displayed
if importlib.util.find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm


class GlobalProgressBar(ProgressBarBase):
    def __init__(self, process_position: int = 0):
        super().__init__()
        self._process_position = process_position
        self._enabled = True
        self.main_progress_bar = None

    def __getstate__(self):
        # can't pickle the tqdm objects
        state = self.__dict__.copy()
        state["main_progress_bar"] = None
        return state

    @property
    def process_position(self) -> int:
        return self._process_position

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.main_progress_bar = tqdm(
            desc="Total Epochs",
            initial=trainer.current_epoch,
            total=trainer.max_epochs,
            position=(2 * self.process_position),
            disable=False,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )

    def on_train_end(self, trainer, pl_module):
        self.main_progress_bar.close()

    def on_epoch_end(self, trainer, pl_module):
        self.main_progress_bar.update(1)
