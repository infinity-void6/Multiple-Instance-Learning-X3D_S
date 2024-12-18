import cv2
import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.amp import autocast
from extract_features import extract_features

import cv2
import torch

def extract_segments_from_video(video_path, segment_size=16, target_shape=(128, 128), frame_skip=10):
    """
    Extracts video segments with frame skipping and efficient memory usage.

    Args:
        video_path (str): Path to the video file.
        segment_size (int): Number of frames per segment.
        target_shape (tuple): Target resolution (height, width) for frames.
        frame_skip (int): Number of frames to skip during extraction.

    Returns:
        list: List of segments, where each segment is a tensor of shape [C, T, H, W].
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        segments = []
        frames = []
        frame_count = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Skip frames based on frame_skip value
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            # Resize frame and convert to tensor
            frame = cv2.resize(frame, target_shape)
            frame_tensor = torch.tensor(frame).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
            frames.append(frame_tensor)

            # Create a segment if enough frames are collected
            if len(frames) == segment_size:
                segments.append(torch.stack(frames, dim=1))  # Stack frames [C, T, H, W]
                frames = []

            frame_count += 1

        # Handle leftover frames (pad to segment size)
        if frames:
            while len(frames) < segment_size:
                frames.append(torch.zeros_like(frames[0]))  # Pad with black frames
            segments.append(torch.stack(frames, dim=1))

        cap.release()
        return segments

    except Exception as e:
        cap.release()
        print(f"Error processing video {video_path}: {e}")
        raise e

# File paths
file_path = r"E:\MIL\weakly_supervised_metadata.csv"  # Path to metadata
output_dir = 'processed_features'  # Directory for saved features
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Load metadata
if not os.path.exists(file_path):
    raise FileNotFoundError(f'Metadata file not found at {file_path}')
metadata = pd.read_csv(file_path)

# Get list of already processed files
processed_files = set(f.split('_features.pt')[0] for f in os.listdir(output_dir) if f.endswith('_features.pt'))

# Filter unprocessed videos
unprocessed_metadata = metadata[~metadata['file_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0]).isin(processed_files)]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Batch size for processing
batch_size = 4

# Process unprocessed videos
for idx, row in tqdm(unprocessed_metadata.iterrows(), total=len(unprocessed_metadata), desc="Processing Unprocessed Videos"):
    video_path = row['file_path']
    label = row['label']

    try:
        if not os.path.exists(video_path):
            tqdm.write(f"Warning: Video file not found: {video_path}")
            continue

        tqdm.write(f"Starting video: {video_path}")

        # Extract segments
        segments = extract_segments_from_video(video_path)

        # Ensure all segments have consistent temporal size
        processed_segments = []
        for segment in segments:
            if segment.size(1) < 16:  # Pad if necessary
                pad_size = 16 - segment.size(1)
                padding = torch.zeros((segment.size(0), pad_size, segment.size(2), segment.size(3)))
                segment = torch.cat([segment, padding], dim=1)
            elif segment.size(1) > 16:  # Truncate if necessary
                segment = segment[:, :16, :, :]
            processed_segments.append(segment)
        segments = processed_segments

        # Divide into batches and extract features
        batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]
        features = []
        for batch in batches:
            batch = torch.stack(batch).to(device)
            with autocast("cuda"):
                batch_features = extract_features(batch, device)
            if isinstance(batch_features, list):
                batch_features = torch.stack(batch_features)
            features.append(batch_features.cpu())

        features = torch.cat(features, dim=0)

        # Save features
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        torch.save({
            "features": features,
            "label": label
        }, os.path.join(output_dir, f"{video_name}_features.pt"))

        torch.cuda.empty_cache()
        tqdm.write(f"Processed video: {video_name}, Label: {label}")

    except RuntimeError as e:
        if "out of memory" in str(e):
            tqdm.write(f"CUDA out of memory error on {video_path}. Skipping...")
            torch.cuda.empty_cache()
        else:
            tqdm.write(f"Runtime error on {video_path}: {e}")
        continue

    except Exception as e:
        tqdm.write(f"Error processing video {video_path}: {e}")
        continue
