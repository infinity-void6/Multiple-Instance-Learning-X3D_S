# Feature Extractor
from pytorchvideo.models.hub import x3d_s
import torch
from torch import nn
from extract_segments import extract_segments_from_video

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained x3d_s model
model = x3d_s(pretrained=True).to(device)

# Modify block 5 to include explicit global average pooling
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.backbone = torch.nn.Sequential(*model.blocks[:-1])  # Blocks 0-4
        self.final_block = model.blocks[5].pool  # Block 5 (unchanged)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Define global average pooling layer

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # Debug input shape
        x = self.backbone(x)  # Pass through blocks 0-4
        #print(f"After backbone shape: {x.shape}")  # Debug backbone output shape
        x=self.final_block(x)
        print(f"After final_block shape: {x.shape}")  # Debug backbone output shape
        x = self.pool(x)  # Pass through block 5 pooling
        print(f"After pooling shape: {x.shape}")  # Debug pooling output shape

        x = torch.mean(x, dim=[-3, -2, -1], keepdim=True)  # Apply global average pooling
        #print(f"After global average pooling shape: {x.shape}")  # Debug GAP output shape

        return x

# Initialize the feature extractor
feature_extractor = FeatureExtractor(model)

# Define feature extraction function
def extract_features(segments, device, model=feature_extractor):
    """
    Extract features from video segments using the modified model.

    Args:
        model (torch.nn.Module): Pretrained and modified model.
        segments (list): List of video segments (tensors).
        device (torch.device): Device for computation.

    Returns:
        list: List of feature tensors.
    """
    features = []
    model.eval()
    with torch.no_grad():
        for segment in segments:
            segment = segment.unsqueeze(0).to(device)  # Add batch dimension and move to device
            feature = model(segment)  # Extract features
            features.append(feature.squeeze(0).cpu())  # Remove batch dimension and move to CPU
    return features

if __name__ == "__main__":
    file = r"E:\MIL\Dataset\Anomaly-Videos\Anomaly-Videos-Part-2\Explosion\Explosion001_x264.mp4"
    list_segment = extract_segments_from_video(file, segment_size=16, target_shape=(240, 320), frame_skip=2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features=extract_features([list_segment[0]],device)
    print(features)
    for idx, segment in enumerate(list_segment):
        print(f"Segment {idx + 1} shape: {segment.shape}")
        break  # Only print the first segment for testing
'''
FIRST MODEL 
________________
from pytorchvideo.models.hub import x3d_s
import torch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=x3d_s(pretrained=True).to(device)
model.blocks[5]=torch.nn.Identity()

def extract_features(segments,device,model=model):
    """
    Extract features from video segments using a pretrained model.

    Args:
        model (torch.nn.Module): Pretrained model.
        segments (list): List of video segments (tensors).
        device (torch.device): Device for computation.

    Returns:
        list: List of feature tensors.
    """
    features=[]
    model.eval()
    with torch.no_grad():
        for segment in segments:
            segment=segment.unsqueeze(0).to(device)
            feature=model(segment)
            features.append(feature.squeeze(0).cpu())
    return features
'''