from torch.utils.data import DataLoader


class ModeDataLoader(DataLoader):
    def __init__(self, mode, *args, **kwargs):
        self.mode = mode
        super(ModeDataLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        self.dataset.set_mode(self.mode)
        return super().__iter__()
