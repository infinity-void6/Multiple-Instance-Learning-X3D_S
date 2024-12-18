import os
import pandas as pd

def create_weakly_supervised_metadata(normal_dir, anomaly_dir, output_csv):
    """
    Create metadata for weakly supervised learning with binary labels (0 for anomalous, 1 for normal).
    
    Args:
        normal_dir (str): Path to the directory containing normal videos.
        anomaly_dir (str): Path to the directory containing anomalous videos (subdirectories for each class).
        output_csv (str): Path to save the metadata CSV file.
    """
    data = []

    # Process normal videos (Label 1)
    for video in os.listdir(normal_dir):
        if video.endswith(('.mp4', '.avi', '.mkv')):  # Include only video files
            data.append({
                "file_path": os.path.join(normal_dir, video),
                "label": 1
            })

    # Process anomalous videos (Label 0)
    for root, _, files in os.walk(anomaly_dir):
        for video in files:
            if video.endswith(('.mp4', '.avi', '.mkv')):  # Include only video files
                data.append({
                    "file_path": os.path.join(root, video),
                    "label": 0  # Label for anomalous videos
                })

    # Save metadata to CSV
    df = pd.DataFrame(data)
    print(df)
    print(len(df))
    df.to_csv(output_csv, index=False)
    print(f"Metadata saved to {output_csv}")

# Example usage
normal_dir = r"E:\MIL\Dataset\Normal-Videos-Part-1"
anomaly_dir = r"E:\MIL\Dataset\Anomaly-Videos"
output_csv = r"E:\MIL\metadata.csv"

create_weakly_supervised_metadata(normal_dir, anomaly_dir, output_csv)

