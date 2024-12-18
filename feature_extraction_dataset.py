# Import necessary libraries
from extract_segments import extract_segments_from_video  # Function to divide videos into segments
from extract_features import extract_features  # Function to extract features from segments
import torch  # PyTorch for tensor operations and GPU acceleration
import os  # For file and directory management
import pandas as pd  # For handling metadata
from tqdm import tqdm  # For progress bar
from torch.amp import autocast  # For mixed-precision acceleration
import cv2  # OpenCV for video processing

# Define file paths and output directories
file_path = r"E:\MIL\metadata.csv"  # Path to the metadata CSV file
normal_output_dir = 'normal_features'  # Directory to save normal video features
anomaly_output_dir = 'anomaly_features'  # Directory to save anomaly video features

# Create the directories if they don't exist
os.makedirs(normal_output_dir, exist_ok=True)
os.makedirs(anomaly_output_dir, exist_ok=True)

# Check if the metadata file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f'Metadata file not found at {file_path}')  # Raise an error if the file is missing

# Load metadata into a Pandas DataFrame
metadata = pd.read_csv(file_path)

# Check if CUDA (GPU) is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, otherwise fallback to CPU

# Define batch size for segment processing
batch_size = 4  # Start small to avoid out-of-memory (OOM) errors

# Function to determine frame_skip based on video length
def determine_frame_skip(video_path):
    cap = cv2.VideoCapture(video_path)  # Open video file
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    duration = frame_count / fps  # Calculate video duration in seconds
    cap.release()  # Release the video file

    if duration <= 300:  # Video length <= 5 minutes
        return 2
    elif duration <= 600:  # Video length <= 10 minutes
        return 5
    else:  # Video length > 10 minutes
        return 10

# Loop through each video in the metadata file
for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing Videos"):
    video_path = row['file_path']  # Path to the video file
    label = row['label']  # Label associated with the video (e.g., 0 for normal, 1 for anomaly)

    try:
        # Check if the video file exists
        if not os.path.exists(video_path):
            tqdm.write(f"Warning: Video file not found: {video_path}")
            continue

        # Determine frame_skip based on video length
        frame_skip = determine_frame_skip(video_path)
        tqdm.write(f"Using frame_skip={frame_skip} for video: {video_path}")

        # Step 1: Extract segments from the video with frame skipping
        segments = extract_segments_from_video(video_path, frame_skip=frame_skip)  # Pass frame_skip to function

        # Step 2: Ensure all segments have consistent temporal size
        processed_segments = []
        for segment in segments:
            if segment.size(1) < 16:  # If the segment has fewer than 16 frames
                pad_size = 16 - segment.size(1)  # Calculate the number of frames to pad
                padding = torch.zeros((segment.size(0), pad_size, segment.size(2), segment.size(3)))  # Create padding
                segment = torch.cat([segment, padding], dim=1)  # Add padding to the temporal dimension
            elif segment.size(1) > 16:  # If the segment has more than 16 frames
                segment = segment[:, :16, :, :]  # Truncate to the first 16 frames
            processed_segments.append(segment)
        segments = processed_segments  # Replace original segments with the processed ones

        # Step 3: Divide segments into smaller batches
        batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]

        # Step 4: Extract features for each batch
        features = []
        for batch in batches:
            batch = torch.stack(batch).to(device)  # Stack batch into a tensor and move to GPU
            with autocast("cuda"):  # Use mixed precision to save memory and accelerate processing
                batch_features = extract_features(batch, device)  # Extract features for the batch

            # Ensure batch_features is a tensor
            if isinstance(batch_features, list):  
                batch_features = torch.stack(batch_features)  # Convert list to tensor if needed

            features.append(batch_features.cpu())  # Move features to CPU to free up GPU memory

        # Combine features from all batches into a single tensor
        features = torch.cat(features, dim=0)

        # Step 5: Save the features and label to a .pt file
        video_name = os.path.splitext(os.path.basename(video_path))[0]  # Extract the video name (without extension)
        if label == 1:
            output_path = os.path.join(normal_output_dir, f"{video_name}_features.pt")
        else:
            output_path = os.path.join(anomaly_output_dir, f"{video_name}_features.pt")

        torch.save({
            "features": features,  # Tensor containing extracted features
            "label": label  # Label associated with the video
        }, output_path)  # Save to the appropriate directory

        # Clear unused GPU memory
        torch.cuda.empty_cache()

        # Log the successful processing of the video
        tqdm.write(f"Processed video: {video_name}, Label: {label}")

    # Handle out-of-memory errors
    except RuntimeError as e:
        if "out of memory" in str(e):
            tqdm.write(f"CUDA out of memory error while processing {video_path}. Trying smaller batch size...")
            torch.cuda.empty_cache()  # Clear GPU memory to recover
        else:
            tqdm.write(f"Error processing video {video_path}: {e}")
        continue  # Skip this video and move to the next

    # Handle any other exceptions
    except Exception as e:
        tqdm.write(f"Error processing video {video_path}: {e}")
        torch.cuda.empty_cache()  # Clear GPU memory for recovery
        continue  # Skip this video and move to the next

'''
# Import necessary libraries
from extract_segments import extract_segments_from_video  # Function to divide videos into segments
from extract_features import extract_features  # Function to extract features from segments
import torch  # PyTorch for tensor operations and GPU acceleration
import os  # For file and directory management
import pandas as pd  # For handling metadata
from tqdm import tqdm  # For progress bar
from torch.amp import autocast  # For mixed-precision acceleration
import cv2  # OpenCV for video processing

# Define file paths and output directory
file_path = r"E:\MIL\weakly_supervised_metadata.csv"  # Path to the metadata CSV file
output_dir = 'processed_features'  # Directory to save processed features
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Check if the metadata file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f'Metadata file not found at {file_path}')  # Raise an error if the file is missing

# Load metadata into a Pandas DataFrame
metadata = pd.read_csv(file_path)

# Check if CUDA (GPU) is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, otherwise fallback to CPU

# Define batch size for segment processing
batch_size = 4  # Start small to avoid out-of-memory (OOM) errors

# Function to determine frame_skip based on video length
def determine_frame_skip(video_path):
    cap = cv2.VideoCapture(video_path)  # Open video file
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    duration = frame_count / fps  # Calculate video duration in seconds
    cap.release()  # Release the video file

    if duration <= 300:  # Video length <= 5 minutes
        return 2
    elif duration <= 600:  # Video length <= 10 minutes
        return 5
    else:  # Video length > 10 minutes
        return 10

# Loop through each video in the metadata file
for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing Videos"):
    video_path = row['file_path']  # Path to the video file
    label = row['label']  # Label associated with the video (e.g., 0 for normal, 1 for anomaly)

    try:
        # Check if the video file exists
        if not os.path.exists(video_path):
            tqdm.write(f"Warning: Video file not found: {video_path}")
            continue

        # Determine frame_skip based on video length
        frame_skip = determine_frame_skip(video_path)
        tqdm.write(f"Using frame_skip={frame_skip} for video: {video_path}")

        # Step 1: Extract segments from the video with frame skipping
        segments = extract_segments_from_video(video_path, frame_skip=frame_skip)  # Pass frame_skip to function

        # Step 2: Ensure all segments have consistent temporal size
        processed_segments = []
        for segment in segments:
            if segment.size(1) < 16:  # If the segment has fewer than 16 frames
                pad_size = 16 - segment.size(1)  # Calculate the number of frames to pad
                padding = torch.zeros((segment.size(0), pad_size, segment.size(2), segment.size(3)))  # Create padding
                segment = torch.cat([segment, padding], dim=1)  # Add padding to the temporal dimension
            elif segment.size(1) > 16:  # If the segment has more than 16 frames
                segment = segment[:, :16, :, :]  # Truncate to the first 16 frames
            processed_segments.append(segment)
        segments = processed_segments  # Replace original segments with the processed ones

        # Step 3: Divide segments into smaller batches
        batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]

        # Step 4: Extract features for each batch
        features = []
        for batch in batches:
            batch = torch.stack(batch).to(device)  # Stack batch into a tensor and move to GPU
            with autocast("cuda"):  # Use mixed precision to save memory and accelerate processing
                batch_features = extract_features(batch, device)  # Extract features for the batch

            # Ensure batch_features is a tensor
            if isinstance(batch_features, list):  
                batch_features = torch.stack(batch_features)  # Convert list to tensor if needed

            features.append(batch_features.cpu())  # Move features to CPU to free up GPU memory

        # Combine features from all batches into a single tensor
        features = torch.cat(features, dim=0)

        # Step 5: Save the features and label to a .pt file
        video_name = os.path.splitext(os.path.basename(video_path))[0]  # Extract the video name (without extension)
        torch.save({
            "features": features,  # Tensor containing extracted features
            "label": label  # Label associated with the video
        }, os.path.join(output_dir, f"{video_name}_features.pt"))  # Save as a .pt file

        # Clear unused GPU memory
        torch.cuda.empty_cache()

        # Log the successful processing of the video
        tqdm.write(f"Processed video: {video_name}, Label: {label}")

    # Handle out-of-memory errors
    except RuntimeError as e:
        if "out of memory" in str(e):
            tqdm.write(f"CUDA out of memory error while processing {video_path}. Trying smaller batch size...")
            torch.cuda.empty_cache()  # Clear GPU memory to recover
        else:
            tqdm.write(f"Error processing video {video_path}: {e}")
        continue  # Skip this video and move to the next

    # Handle any other exceptions
    except Exception as e:
        tqdm.write(f"Error processing video {video_path}: {e}")
        torch.cuda.empty_cache()  # Clear GPU memory for recovery
        continue  # Skip this video and move to the next
'''


