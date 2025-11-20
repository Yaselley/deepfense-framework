from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Abstract base class for all datasets.
    Provides a shared structure but leaves implementation to subclasses.
    """

    def __init__(self):
        super().__init__()

    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__().")

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__().")
