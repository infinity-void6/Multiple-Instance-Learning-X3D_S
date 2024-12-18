import torch
import torch.nn.functional as F

def ranking_loss(scores, labels, batch_size, lamda_sparsity=1e-6, lamda_smooth=1e-6, margin=0.5):
    """
    Ranking loss for weakly-supervised MIL anomaly detection.

    Parameters:
    - scores (torch.Tensor): Predicted scores for all segments. Shape: [batch_size * num_segments].
    - labels (torch.Tensor): Binary labels for videos. Shape: [batch_size].
    - batch_size (int): Number of videos per batch.
    - lamda_sparsity (float): Weight for sparsity loss. Default: 1e-6.
    - lamda_smooth (float): Weight for smoothness loss. Default: 1e-6.
    - margin (float): Margin for ranking loss. Default: 1.0.

    Returns:
    - torch.Tensor: The combined ranking loss.
    """
    num_segments = scores.shape[0] // batch_size  # Segments per video
    total_loss = 0.0  # Initialize cumulative loss

    for i in range(batch_size):
        # Extract scores for the current video
        video_scores = scores[i * num_segments : (i + 1) * num_segments]
        video_label = labels[i]
        # print(f'video_label:{video_label}')

        # Compute max scores for anomaly and normal cases
        max_anomalous = torch.tensor(float("-inf"), device=scores.device)
        max_normal = torch.tensor(float("-inf"), device=scores.device)

        if video_label == 0:  # Anomalous video
            max_anomalous = torch.max(video_scores)
            #print(f'max_anomalous:{max_anomalous}')
        elif video_label == 1:  # Normal video
            max_normal = torch.max(video_scores)
            #print(f'max_normal:{max_normal}')

        # Compute ranking loss: Ensuring valid conditions
        if max_anomalous != float("-inf") and max_normal != float("-inf"):
            rank_loss = F.relu(margin - max_anomalous + max_normal)
            total_loss += rank_loss

        # Sparsity loss: Encourage sparsity in scores
        sparsity_loss = lamda_sparsity * torch.sum(torch.sigmoid(video_scores))

        # Smoothness loss: Penalize abrupt changes between adjacent segments
        smoothness_loss = lamda_smooth * torch.sum(
            (torch.sigmoid(video_scores[1:]) - torch.sigmoid(video_scores[:-1])) ** 2
        )

        total_loss += sparsity_loss + smoothness_loss

    # Normalize by batch size
    return total_loss / batch_size


'''import torch
import torch.nn.functional as F

def ranking_loss(scores, labels, batch_size, lamda_sparsity=1e-6, lamda_smooth=1e-6, margin=1.0):
    """
    Ranking loss for weakly-supervised MIL anomaly detection.

    Parameters:
    - scores (torch.Tensor): Predicted scores for all segments. Shape: [batch_size * num_segments].
    - labels (torch.Tensor): Binary labels for videos. Shape: [batch_size].
    - batch_size (int): Number of videos per batch.
    - lamda_sparsity (float): Weight for sparsity loss. Default: 1e-6.
    - lamda_smooth (float): Weight for smoothness loss. Default: 1e-6.
    - margin (float): Margin for ranking loss. Default: 1.0.

    Returns:
    - torch.Tensor: The combined ranking loss.
    """
    num_segments = scores.shape[0] // batch_size  # Assume all videos have the same number of segments
    loss = torch.tensor(0.0, device=scores.device, requires_grad=True)  # Initialize loss

    for i in range(batch_size):
        # Extract scores for the current video
        video_scores = scores[i * num_segments : (i + 1) * num_segments]
        video_label = labels[i]

        # Compute the max score for the video
        max_score = torch.max(video_scores)

        if video_label == 0:  # Anomalous video
            max_anomalous = max_score
        elif video_label == 1:  # Normal video
            max_normal = max_score

        # Ranking loss: max_anomalous > max_normal + margin
        ranking_loss = F.relu(margin - max_anomalous + max_normal)
        loss += ranking_loss

        # Add sparsity and smoothness losses
        sparsity_loss = lamda_sparsity * torch.sum(torch.sigmoid(video_scores))
        smoothness_loss = lamda_smooth * torch.sum((torch.sigmoid(video_scores[1:]) - torch.sigmoid(video_scores[:-1])) ** 2)

        loss += sparsity_loss + smoothness_loss

    # Normalize by batch size
    return loss / batch_size
'''

'''import torch
from torch import nn
def combined_loss(scores, labels, lamda_sparsity=1e-6, lamda_smooth=1e-6):
    """
    Calculates the combined loss function for weakly-supervised MIL anomaly detection.

    The combined loss consists of:
    1. Binary Cross-Entropy Loss with Logits:
       - This is the classification loss that measures how well the model predicts the 
         anomaly scores for each segment. It internally applies the sigmoid function 
         to convert logits into probabilities before calculating the loss.
    
    2. Sparsity Loss:
       - This encourages sparsity in anomaly predictions by penalizing the sum of 
         predicted probabilities across all segments. It helps the model focus on 
         a small number of segments likely to contain anomalies.

    3. Smoothness Loss:
       - This penalizes abrupt changes in predictions across adjacent segments 
         to ensure temporal smoothness in the predicted anomaly scores.

    Parameters:
    - scores (torch.Tensor): Predicted logits from the model. Shape: [num_segments].
    - labels (torch.Tensor): Ground truth labels. Shape: [num_segments].
    - lamda_sparsity (float): Weight for sparsity loss. Default: 8e-5.
    - lamda_smooth (float): Weight for smoothness loss. Default: 8e-5.

    Returns:
    - torch.Tensor: The combined loss value.
    """
    # Binary Cross-Entropy Loss with Logits (handles sigmoid internally)
    #ce_loss = nn.BCEWithLogitsLoss()(scores, labels)

    # Sparsity Loss: Penalizes the sum of predicted probabilities (encourages sparse anomaly detection)
    # sparsity_loss = lamda_sparsity * torch.sum(torch.sigmoid(scores))

    # Smoothness Loss: Penalizes abrupt changes in adjacent segment predictions
    # smoothness_loss = lamda_smooth * torch.sum((torch.sigmoid(scores[1:]) - torch.sigmoid(scores[:-1])) ** 2)

    # Combined loss
    device=torch.device('cuda' if torch.cuda.is_available() else "cpu")
    pos_weight = torch.tensor([1.5]).to(device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce_loss = bce_loss(scores, labels)
    sparsity_loss = lamda_sparsity * torch.sum(torch.sigmoid(scores))
    smoothness_loss = lamda_smooth * torch.sum((torch.sigmoid(scores[1:]) - torch.sigmoid(scores[:-1])) ** 2)

    

    return ce_loss + sparsity_loss + smoothness_loss'''