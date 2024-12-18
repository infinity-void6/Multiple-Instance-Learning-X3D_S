# define model
import torch
import torch.nn as nn
from Feature_Dataset_Class import FeatureDataset
from torch.utils.data import DataLoader,Dataset
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler 
from loss_function import ranking_loss
def compute_threshold(anomaly_score, normal_score, margin=5e-17):
    """
    Computes a dynamic threshold based on anomaly and normal scores.
    Args:
    - anomaly_score (float): Mean or max score of the anomaly video.
    - normal_score (float): Mean or max score of the normal video.
    - margin (float): Margin to separate normal and anomalous scores.
    Returns:
    - float: The computed threshold.
    """
    return (anomaly_score + normal_score) / 2  # margin Midpoint + margi

def sharpened_sigmoid(x, alpha=25):
    return torch.sigmoid(alpha * x)
import torch

def cos_transformation_0_1(x, k=50):
    """
    Amplifies differences using a sine transformation, mapping outputs to [0, 1].
    Args:
    - x (torch.Tensor): Input values in any range.
    - k (float): Scaling factor to control sensitivity.
    Returns:
    - torch.Tensor: Transformed values in range [0, 1].
    """
    return (torch.cos(k*x)+1)/2

'''class SequentialMILModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512):
        """
        Initializes a Sequential MIL Model for anomaly detection.

        Parameters:
        - input_dim (int): Input feature dimension (default: 2048).
        - hidden_dim (int): Hidden layer dimension (default: 512).
        """
        super(SequentialMILModel, self).__init__()
        print("SequentialMILModel Initialized")  # Debug print
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_dim, 32)         # Second fully connected layer
        self.fc3 = nn.Linear(32, 1)                 # Final output layer
        self.dropout = nn.Dropout(0.6)              # Dropout with probability 0.6
        self.relu = nn.ReLU()                       # ReLU activation
        self.sigmoid = nn.Sigmoid()                 # Sigmoid activation for output
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.tanh=nn.Tanh()
        self.silu=nn.SiLU()
        self.softmax=nn.Softmax()

    def forward(self, x):
        """
        Forward pass for the model.

        Parameters:
        - x (torch.Tensor): Input features of shape [num_segments, input_dim].

        Returns:
        - torch.Tensor: Raw logits for each segment. Shape: [num_segments, 1].
        """
        if x.dim() > 2:  # Check if the input tensor has more than 2 dimensions
            # Reshape the tensor to collapse all dimensions except the batch size and feature dimension
            # For example, input shape (B, C, T, H, W) -> (B, C * T * H * W)
            # This ensures the input is compatible with fully connected (linear) layers.
            x = x.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove [1, 1, 1] at the end
            x = x.view(-1, x.size(-1))
        # Normalize the input features
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

        # Pass the reshaped tensor through the first fully connected layer
        # This layer reduces the input feature dimension (e.g., 2048) to the hidden dimension (e.g., 512),
        # allowing the model to learn more compact representations.
        # x=self.relu (self.fc1(x))
        # x = self.sigmoid(self.fc1(x))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

        # Apply dropout to randomly zero out some of the activations, reducing overfitting
        # Dropout helps regularize the model by preventing it from relying too heavily on specific neurons.
        # x = self.dropout(x)

        # Pass the tensor through the second fully connected layer
        # This further reduces the dimension (e.g., from 512 to 32) to extract more abstract features.
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)


        # Apply another dropout layer for additional regularization
        # x = self.dropout(x)

        # Pass the tensor through the final fully connected layer
        # This layer maps the feature vector to a single logit (raw prediction value) for each segment.
        x = self.fc3(x)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
        # x = self.dropout(x)

        # Apply the sigmoid activation function to map the logits to probabilities in the range [0, 1]
        # This is useful for binary classification tasks (e.g., anomaly detection).
        x = self.sigmoid(x)


        # Squeeze the last dimension to simplify the output shape
        # For example, output shape changes from [num_segments, 1] to [num_segments].
        return x.squeeze(-1)'''

class SequentialMILModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024):
        super(SequentialMILModel, self).__init__()
        print("Complex SequentialMILModel Initialized")
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)

        self.fc4 = nn.Linear(hidden_dim // 4, 32)
        self.fc5 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() > 2:
            x = x.squeeze(-1).squeeze(-1).squeeze(-1)
            x = x.view(-1, x.size(-1))

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        #x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.silu(x)
        #x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        #x = self.dropout(x)

        x = self.fc4(x)
        x = self.relu(x)

        x = self.fc5(x)
        x = self.sigmoid(x)

        return x.squeeze(-1)

if __name__ == "__main__":
    model = SequentialMILModel(input_dim=2048, hidden_dim=512)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion =ranking_loss
    scaler=GradScaler
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normal_files = [
        r"E:/MIL/Code/normal_features/Normal_Videos_003_x264_features_aug.pt",
        r"E:/MIL/Code/normal_features/Normal_Videos_866_x264_features.pt",
        r"E:/MIL/Code/normal_features/Normal_Videos_656_x264_features.pt"
    ]

    anomaly_files = [
        r"E:/MIL/Code/anomaly_features/Abuse001_x264_features.pt",
        r"E:/MIL/Code/anomaly_features/Abuse003_x264_features.pt",
        r"E:/MIL/Code/anomaly_features/Fighting041_x264_features.pt"]

        # Define parameters
    max_segments_train = 198  # Max segments for training
    batch_size = 2  # Batch size for testing

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

    # Initialize the model and move it to the device
    model = SequentialMILModel(input_dim=2048, hidden_dim=512).to(device)  # Move model to the correct device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = ranking_loss  # Loss function
    scaler = GradScaler()  # Correctly initialized GradScaler

    def train_epoch(train_loader, model, optimizer, criterion, device, scaler, batch_size):
        model.train()  # Set model to training mode
        total_loss = 0.0

        for normal_features, anomalous_features in tqdm(train_loader, desc="Training"):
            # Combine features
            if normal_features.dim() > 2:            
                normal_features = normal_features.squeeze(-1).squeeze(-1).squeeze(-1)
                normal_features = normal_features.view(-1, normal_features.size(-1))
            if anomalous_features.dim() > 2:
                anomalous_features = anomalous_features.squeeze(-1).squeeze(-1).squeeze(-1)
                anomalous_features = anomalous_features.view(-1, anomalous_features.size(-1))

            # Move features to the device
            normal_features = normal_features.to(device)
            anomalous_features = anomalous_features.to(device)
            inputs = torch.cat((normal_features, anomalous_features), dim=0)

            # Move labels to the device
            labels = torch.cat((
                torch.zeros(len(normal_features), 1, device=device),
                torch.ones(len(anomalous_features), 1, device=device)
            ), dim=0)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast(device_type=device.type, enabled=True):
                outputs = model(inputs)  # Pass inputs through model
                loss = criterion(outputs, labels, batch_size)  # Compute loss

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_loss:.4f}")
        return avg_loss
    for i in range(10):
        train_loss = train_epoch(train_loader, model, optimizer, criterion, device, scaler, batch_size)
