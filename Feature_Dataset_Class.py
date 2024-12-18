# Code for Feature Dataset

from torch.utils.data import Dataset
import torch

class FeatureDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx],weights_only=False)
        # Ensure features are cast to float32
        features = data["features"].to(dtype=torch.float32)
        label = torch.tensor(data["label"]).float()
        return features, label