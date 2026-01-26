import torch
from torch.utils.data import Dataset, DataLoader

class FraudDataset(Dataset):
    """
    - X float32
    - y float32 with shape (n, 1)
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_loader(X, y, batch_size: int, shuffle: bool):
    ds = FraudDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
