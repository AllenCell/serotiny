from torch.utils.data import DataLoader


class TrainDataLoader(DataLoader):
    def __iter__(self):
        self.dataset.train()
        return super().__iter__()

class EvalDataLoader(DataLoader):
    def __iter__(self):
        self.dataset.eval()
        return super().__iter__()
