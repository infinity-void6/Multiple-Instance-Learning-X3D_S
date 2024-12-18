import torch
from torch import nn, optim
from torch.amp import autocast, GradScaler
from split_data_train_val_test import create_dataloader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Assuming SequentialMILModel is already implemented
from Sequential_Model import SequentialMILModel
from loss_function import ranking_loss

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize Model
model = SequentialMILModel(input_dim=2048).to(device)

# 2. Define Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Scheduler: Reduces learning rate when validation loss plateaus
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

# 3. Define Loss Function
criterion = ranking_loss  # Replace with your ranking loss function

# 4. Gradient Scaler for Mixed Precision
scaler = GradScaler()

# Data Loaders
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
test_loader=create_dataloader(test_normal_dir,test_anomalous_dir,batch_size, max_segments_test)

def print_model_weights(model, epoch):
    """
    Prints the weights of the model's layers.
    
    Args:
        model (torch.nn.Module): Trained model.
        epoch (int): Current epoch number.
    """
    print(f"\nWeights after Epoch {epoch + 1}:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}")
            print(f"Weight: {param.data}")
            print(f"Weight Shape: {param.data.shape}")
            print(f"Weight Mean: {param.data.mean():.6f}, Weight Std: {param.data.std():.6f}\n")

# Training Loop
def train_epoch(train_loader, model, optimizer, criterion, device, scaler, batch_size):
    model.train()  # Set model to training mode
    total_loss = 0.0

    for normal_features, anomalous_features in tqdm(train_loader, desc="Training"):
        # Combine features
        if normal_features.dim() > 2:            
            normal_features = normal_features.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove [1, 1, 1] at the end
            normal_features = normal_features.view(-1, normal_features.size(-1))          # Reshape to [n * pad seg, 2048]
            #print(f"Normal features reshaped: {normal_features.shape}")
        if anomalous_features.dim() > 2:
            anomalous_features=anomalous_features.squeeze(-1).squeeze(-1).squeeze(-1)
            anomalous_features = anomalous_features.view(-1,anomalous_features.size(-1))
            #print(f"Anomalous features reshaped: {anomalous_features.shape}")
        
        normal_features = normal_features.to(device)
        anomalous_features = anomalous_features.to(device)
        inputs = torch.cat((normal_features, anomalous_features), dim=0)

        # Labels: 0 for normal, 1 for anomalous
        labels = torch.cat((
            torch.zeros(len(normal_features), 1).to(device),  # Normal
            torch.ones(len(anomalous_features), 1).to(device)  # Anomalous
        ), dim=0)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, labels, batch_size)

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss

# Validation Loop
def validate_epoch(val_loader, model, criterion, device, batch_size):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():  # No gradient calculation
        for normal_features, anomalous_features in tqdm(val_loader, desc="Validation"):
            # Combine features
            if normal_features.dim() > 2:            
                normal_features = normal_features.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove [1, 1, 1] at the end
                normal_features = normal_features.view(-1, normal_features.size(-1))   # Reshape to [n * pad seg, 2048]
                #print(f"Normal features reshaped: {normal_features.shape}")
            if anomalous_features.dim() > 2:
                anomalous_features=anomalous_features.squeeze(-1).squeeze(-1).squeeze(-1)
                anomalous_features = anomalous_features.view(-1,anomalous_features.size(-1))
                #print(f"Anomalous features reshaped: {anomalous_features.shape}")
            
            normal_features = normal_features.to(device)
            anomalous_features = anomalous_features.to(device)
            inputs = torch.cat((normal_features, anomalous_features), dim=0)

            # Labels: 1 for normal, 0 for anomalous
            labels = torch.cat((
                torch.ones(len(normal_features), 1).to(device),  # Normal
                torch.zeros(len(anomalous_features), 1).to(device)  # Anomalous
            ), dim=0)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels,batch_size)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def test_epoch(test_loader, model, criterion, device, batch_size):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():  # No gradient calculation
        for normal_features, anomalous_features in tqdm(test_loader, desc="Validation"):
            # Combine features
            if normal_features.dim() > 2:            
                normal_features = normal_features.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove [1, 1, 1] at the end
                normal_features = normal_features.view(-1, normal_features.size(-1))   # Reshape to [n * pad seg, 2048]
                #print(f"Normal features reshaped: {normal_features.shape}")
            if anomalous_features.dim() > 2:
                anomalous_features=anomalous_features.squeeze(-1).squeeze(-1).squeeze(-1)
                anomalous_features = anomalous_features.view(-1,anomalous_features.size(-1))
                #print(f"Anomalous features reshaped: {anomalous_features.shape}")
            
            normal_features = normal_features.to(device)
            anomalous_features = anomalous_features.to(device)
            inputs = torch.cat((normal_features, anomalous_features), dim=0)

            # Labels: 1 for normal, 0 for anomalous
            labels = torch.cat((
                torch.ones(len(normal_features), 1).to(device),  # Normal
                torch.zeros(len(anomalous_features), 1).to(device)  # Anomalous
            ), dim=0)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels,batch_size)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss
# Training and Validation
num_epochs = 10
batch_size = 4  # Keep consistent with DataLoader



for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train_epoch(train_loader, model, optimizer, criterion, device, scaler, batch_size)
    val_loss = validate_epoch(val_loader, model, criterion, device, batch_size)
    test_loss=test_epoch(test_loader,model, criterion, device, batch_size)
    scheduler.step(val_loss)
    
    # Print model weights
    print_model_weights(model, epoch)



'''
from Sequential_Model import SequentialMILModel
import torch
from torch import nn
from split_data_train_val_test import DataPreparation
from loss_function import combined_loss
from torch.amp import autocast 
# Automatically switches between float16 and float32 precision during 
# training for faster computation.
from torch.amp import GradScaler
# Scales gradients to prevent underflow when using float16. not go to 0
# Initialize Model, Optimizer, and Scaler
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SequentialMILModel(input_dim=2048).to(device)
# Initialize optimizer and scheduler
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",         # Minimize the monitored metric (e.g., loss)
    factor=0.1,         # Reduce LR by a factor of 0.1
    patience=5,         # Wait 5 epochs for improvement before reducing LR
    verbose=True        # Print updates when LR changes
)
scaler = GradScaler()  # Gradient scaler for mixed precision

# Path to the directory containing processed feature files
processed_features_dir = "processed_features"

# Initialize the DataPreparation class
data_prep = DataPreparation(processed_features_dir)
# Split the dataset
data_prep.split_dataset(test_size=0.4, val_size=0.5)
# Create DataLoaders
train_loader, val_loader,test_loader = data_prep.get_data_loaders(batch_size=1)


# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    train_loss = 0.0

    # Training Progress Bar
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training", leave=False)
    ce_loss=0
    sparsity_loss=0
    smoothness_loss=0
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
        ce_loss+= nn.BCEWithLogitsLoss()(scores, label)
        sparsity_loss+= 1e-6 * torch.sum(torch.sigmoid(scores))
        smoothness_loss+= 1e-6 * torch.sum((torch.sigmoid(scores[1:]) - torch.sigmoid(scores[:-1])) ** 2)
    print(f"BCE={ce_loss.item()}, Sparsity={sparsity_loss.item()}, Smoothness={smoothness_loss.item()}")
    print(f'train loss - epoch:{train_loss} - {epoch}')
    
print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")'''


'''
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
'''
    # Print epoch summary
    # print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")
    # print(f"Val Loss: {val_loss:.4f}")

'''
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