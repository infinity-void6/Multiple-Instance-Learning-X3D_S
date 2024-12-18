import os
import random
import shutil
import torch
from torch.utils.data import Dataset, DataLoader

# Define paths
normal_dir = "normal_features"
anomaly_dir = "anomaly_features"

# Create output directories
output_dirs = {
    "train": {"normal": "split/train/normal", "anomaly": "split/train/anomalous"},
    "val": {"normal": "split/val/normal", "anomaly": "split/val/anomalous"},
    "test": {"normal": "split/test/normal", "anomaly": "split/test/anomalous"},
}

# Create the directories
for split, dirs in output_dirs.items():
    for key, path in dirs.items():
        os.makedirs(path, exist_ok=True)

# Load all files
normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith(".pt")]
anomaly_files = [os.path.join(anomaly_dir, f) for f in os.listdir(anomaly_dir) if f.endswith(".pt")]

# Shuffle the files
random.seed(42)
random.shuffle(normal_files)
random.shuffle(anomaly_files)

# Compute splits
split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

def split_files(files, ratios):
    n_train = int(len(files) * ratios["train"])
    n_val = int(len(files) * ratios["val"])
    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]
    return train, val, test

normal_train, normal_val, normal_test = split_files(normal_files, split_ratios)
anomaly_train, anomaly_val, anomaly_test = split_files(anomaly_files, split_ratios)

# Copy files to respective directories
for split, files in [("train", normal_train), ("val", normal_val), ("test", normal_test)]:
    for file in files:
        shutil.copy(file, output_dirs[split]["normal"])

for split, files in [("train", anomaly_train), ("val", anomaly_val), ("test", anomaly_test)]:
    for file in files:
        shutil.copy(file, output_dirs[split]["anomaly"])

print("Dataset split completed.")
print(f"Train: Normal = {len(normal_train)}, Anomaly = {len(anomaly_train)}")
print(f"Validation: Normal = {len(normal_val)}, Anomaly = {len(anomaly_val)}")
print(f"Test: Normal = {len(normal_test)}, Anomaly = {len(anomaly_test)}")

def pad_to_max_segments(feature_tensor, max_segments):
    """
    Pads the feature tensor to the max_segments along the batch/segment dimension.
    
    Args:
        feature_tensor (torch.Tensor): Input tensor of shape [B, ...].
        max_segments (int): Maximum segments to pad to.
        
    Returns:
        torch.Tensor: Padded tensor of shape [max_segments, ...].
    """
    current_segments = feature_tensor.size(0)
    if current_segments < max_segments:
        padding = torch.zeros((max_segments - current_segments, *feature_tensor.shape[1:]), device=feature_tensor.device)
        feature_tensor = torch.cat([feature_tensor, padding], dim=0)
    return feature_tensor

class FeatureDataset(Dataset):
    def __init__(self, normal_files, anomaly_files, max_segments):
        self.normal_files = normal_files
        self.anomaly_files = anomaly_files
        self.max_segments = max_segments  # Max segments for padding

    def __len__(self):
        return max(len(self.normal_files), len(self.anomaly_files))

    def __getitem__(self, idx):
        normal_idx = idx % len(self.normal_files)
        anomaly_idx = idx % len(self.anomaly_files)

        normal_feature = torch.load(self.normal_files[normal_idx])["features"]
        anomaly_feature = torch.load(self.anomaly_files[anomaly_idx])["features"]

        # Apply padding
        normal_feature = pad_to_max_segments(normal_feature, self.max_segments)
        anomaly_feature = pad_to_max_segments(anomaly_feature, self.max_segments)

        return normal_feature, anomaly_feature

# Create DataLoaders for training, validation, and testing
def create_dataloader(normal_dir, anomaly_dir, batch_size,max_segments):
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith(".pt")]
    anomaly_files = [os.path.join(anomaly_dir, f) for f in os.listdir(anomaly_dir) if f.endswith(".pt")]
    dataset = FeatureDataset(normal_files, anomaly_files,max_segments)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Directories for train, val, and test splits
train_normal_dir = "split/train/normal"
train_anomalous_dir = "split/train/anomalous"
val_normal_dir = "split/val/normal"
val_anomalous_dir = "split/val/anomalous"
test_normal_dir = "split/test/normal"
test_anomalous_dir = "split/test/anomalous"

# Max segments for each split
max_segments_train = 887  # Max for training
max_segments_val = 675    # Max for validation
max_segments_test = 271   # Max for testing

# Batch size
batch_size = 4

# Create DataLoaders
train_loader = create_dataloader(train_normal_dir, train_anomalous_dir, batch_size, max_segments_train)
val_loader = create_dataloader(val_normal_dir, val_anomalous_dir, batch_size, max_segments_val)
test_loader = create_dataloader(test_normal_dir, test_anomalous_dir, batch_size, max_segments_test)

# Print DataLoader sizes
print(f"Train Loader: {len(train_loader)} batches")
print(f"Validation Loader: {len(val_loader)} batches")
print(f"Test Loader: {len(test_loader)} batches")

if __name__=='__main__':
    normal_files=[r"E:\MIL\Code\normal_features\Normal_Videos_003_x264_features_aug.pt",r"E:\MIL\Code\normal_features\Normal_Videos_866_x264_features.pt",r"E:\MIL\Code\normal_features\Normal_Videos_656_x264_features.pt"]
    anomaly_files=[r"E:\MIL\Code\anomaly_features\Abuse001_x264_features.pt",r"E:\MIL\Code\anomaly_features\Abuse003_x264_features.pt",r"E:\MIL\Code\anomaly_features\Fighting041_x264_features.pt"]
        # Define parameters
    max_segments_train = 198  # Max segments for training
    batch_size = 2  # Batch size for testing
    def create_dataloader(normal_files, anomaly_files, batch_size, max_segments):
        """
        Creates a DataLoader using the provided lists of file paths.

        Args:
            normal_files (list): List of file paths for normal features.
            anomaly_files (list): List of file paths for anomalous features.
            batch_size (int): Batch size for DataLoader.
            max_segments (int): Maximum segments to pad to.

        Returns:
            DataLoader: A PyTorch DataLoader instance.
        """
        dataset = FeatureDataset(normal_files, anomaly_files, max_segments)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create a DataLoader
    train_loader = create_dataloader(normal_files, anomaly_files, batch_size, max_segments_train)
    # Test the DataLoader
    for batch_idx, (normal_batch, anomaly_batch) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        reshaped = normal_batch.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove [1, 1, 1] at the end
        reshaped = reshaped.view(-1, reshaped.size(-1))   
        print(f"Normal batch shape: {reshaped.shape}")
        reshaped = anomaly_batch.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove [1, 1, 1] at the end
        reshaped = reshaped.view(-1, reshaped.size(-1)) 
        print(f"Anomalous batch shape: {reshaped.shape}")
        # Only display the first batch for testing
    



'''
# TRAIN AND TEST

from Feature_Dataset_Class import FeatureDataset  # Import FeatureDataset class
from Sequential_Model import SequentialMILModel   # Import SequentialMILModel class

import os 
import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.amp import autocast 
# Automatically switches between float16 and float32 precision during 
# training for faster computation.
from torch.amp import GradScaler
# Scales gradients to prevent underflow when using float16. not go to 0
from tqdm import tqdm

# Import custom classes
from Feature_Dataset_Class import FeatureDataset
from Sequential_Model import SequentialMILModel

# data_preparation.py

import os
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from Feature_Dataset_Class import FeatureDataset


class DataPreparation:
    """
    # Handles dataset splitting and DataLoader creation for training, validation, and testing datasets.
    """

    def __init__(self, processed_features_dir, random_seed=42):
        """
        # Initializes the DataPreparation class.
        # Args:
            p# rocessed_features_dir (str): Directory containing the processed feature files.
            # andom_seed (int): Random seed for reproducibility.
        """
        self.processed_features_dir = processed_features_dir
        self.random_seed = random_seed
        self.train_files = []
        self.val_files = []
        self.test_files = []

    def split_dataset(self, test_size=0.4, val_size=0.5):
        """
        Splits the dataset into training, validation, and test sets.
        Args:
            test_size (float): Proportion of the dataset to include in the test set.
            val_size (float): Proportion of the temporary test set to include in the validation set.
        """
        all_files = [
            os.path.join(self.processed_features_dir, f)
            for f in os.listdir(self.processed_features_dir)
            if f.endswith(".pt")
        ]
        print(f"Total files: {len(all_files)}")

        random.seed(self.random_seed)
        random.shuffle(all_files)
        train_files, temp_files = train_test_split(all_files, test_size=test_size, random_state=self.random_seed)
        val_files, test_files = train_test_split(temp_files, test_size=val_size, random_state=self.random_seed)

        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files

        print(f"Train files: {len(self.train_files)}")
        print(f"Validation files: {len(self.val_files)}")
        print(f"Test files: {len(self.test_files)}")

    def get_test_loader(self, batch_size=1, shuffle=False):
        """
        Creates a DataLoader for the test dataset.
        Args:
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the test data.
        Returns:
            DataLoader: DataLoader for the test dataset.
        """
        test_dataset = FeatureDataset(self.test_files)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return test_loader

    def get_data_loaders(self, batch_size=1, shuffle_train=True):
        """
        Creates DataLoaders for training, validation, and testing datasets.
        Args:
            batch_size (int): Batch size for the DataLoader.
            shuffle_train (bool): Whether to shuffle the training data.
        Returns:
            tuple: DataLoaders for train, validation, and test datasets.
        """
        train_dataset = FeatureDataset(self.train_files)
        val_dataset = FeatureDataset(self.val_files)
        test_dataset = FeatureDataset(self.test_files)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader




if __name__=='__main__':
    # Split Dataset
    processed_features_dir = "processed_features"
    all_files = [os.path.join(processed_features_dir, f) for f in os.listdir(processed_features_dir) if f.endswith(".pt")]
    print(f'len of all files {len(all_files)}')
    random.seed(42)
    random.shuffle(all_files)
    train_files, temp_files = train_test_split(all_files, test_size=0.4, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    print(f'len of train_files: {len(train_files)}')
    print(f'len of val_files: {len(val_files)}')
    print(f'len of test_files: {len(test_files)}')

    def count_normal(files):
            # Example: Assume normal files have "Normal" in their filename
            return sum(1 for file in files if "Normal" in os.path.basename(file))

    num_normals_train = count_normal(train_files)
    num_normals_val = count_normal(val_files)
    num_normals_test = count_normal(test_files)

    print(f"Normal files in training set: {num_normals_train}")
    print(f"Normal files in validation set: {num_normals_val}")
    print(f"Normal files in test set: {num_normals_test}")



    # Create Datasets and Dataloaders
    train_dataset = FeatureDataset(train_files)
    val_dataset = FeatureDataset(val_files)
    test_dataset = FeatureDataset(test_files)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print(f"length_of_train loader: {len(train_loader)}")
    for feature,label in train_loader:
        print(f'train label:{label}')
        print(f'train feature_shape: {feature.shape}')
        break

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"length of test loader: {len(test_loader)}")

    for feature,label in test_loader:
        print(f'test label:{label}')
        print(f'test feature.shape{feature.shape}')
        break
    # Initialize Model, Optimizer, and Scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequentialMILModel(input_dim=2048).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scaler = GradScaler()  # Gradient scaler for mixed precision""'''

    


'''

# Initialize GradScaler for mixed precision training
scaler = GradScaler()

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    train_loss = 0.0

    # Training Progress Bar
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training", leave=False)
    for features, label in train_progress:
        features = features.squeeze(0).to(device)  # [num_segments, feature_dim] [Segments, 2048]
        label = label.to(device)  # [1]

        optimizer.zero_grad()  # Reset gradients

        # Forward pass with autocast for mixed precision
        with autocast(device_type='cuda', dtype=torch.float32):
            scores = model(features)  # Get raw logits from the model
            scores=torch.mean(scores).unsqueeze(0)
            loss = combined_loss(scores, label) 


        # Backward pass with gradient scaling
        scaler.scale(loss).backward()  # Scale gradients for mixed precision
        scaler.step(optimizer)  # Update model parameters
        scaler.update()  # Update gradient scaler for next step

        train_loss += loss.item()  # Accumulate training loss

        # Update training progress bar with current loss
        train_progress.set_postfix(loss=loss.item())

    # Validation Loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1} - Validation", leave=False)
    with torch.no_grad():  # Disable gradient calculations during validation
        i=0
        for features, label in val_progress:
            
            features = features.squeeze(0).to(device).to(dtype=torch.float32)  # Convert to float32
            label = label.to(device)

            # Forward pass
            scores = model(features)
            scores=torch.mean(scores).unsqueeze(0)
            loss = combined_loss(scores, label)  # Calculate validation loss
            print(f'validation_loss at iteration {i} is : {loss.item()}')
            val_loss += loss.item()
            i+=1
            print(i)

            # Update validation progress bar with current loss
            val_progress.set_postfix(loss=loss.item())

    # Print epoch summary
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


# Test Loop
model.eval()
test_scores = []
test_labels = []
test_progress = tqdm(test_loader, desc="Testing")

with torch.no_grad():
    for features, label in test_progress:
        features = features.squeeze(0).to(device).to(dtype=torch.float32)
        label = label.to(device)
        scores = model(features)
        scores=torch.mean(scores).unsqueeze(0)
        test_scores.append(scores.item())
        test_labels.append(label.item())

# Calculate Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score

# print(test_scores)

predictions = [1 if score > 0.5 else 0 for score in test_scores]
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)

print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
'''