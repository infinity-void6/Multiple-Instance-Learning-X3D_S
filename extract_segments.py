# extract segments
import torch
import cv2
def extract_segments_from_video(video_path, segment_size=16, target_shape=(240, 320), frame_skip=10):
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

if __name__ == "__main__":
    file = r"E:\MIL\Dataset\Anomaly-Videos\Anomaly-Videos-Part-2\Explosion\Explosion001_x264.mp4"
    file2=r"E:\MIL\Dataset\Anomaly-Videos\Anomaly-Videos-Part-1\Abuse\Abuse001_x264.mp4"
    list_segment = extract_segments_from_video(file, segment_size=16, target_shape=(240, 320), frame_skip=2)
    list_segment_2 = extract_segments_from_video(file2, segment_size=16, target_shape=(240, 320), frame_skip=2)
    print(len(list_segment))
    for idx, segment in enumerate(list_segment_2):
        print(f"Segment {idx + 1} shape: {segment.shape}")
        break  # Only print the first segment for testing


