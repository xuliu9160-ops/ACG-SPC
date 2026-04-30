import torch
import torch.nn as nn
import torch.nn.functional as F


def combined_loss(
    frame_logits: torch.Tensor,
    video_logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights_frame: torch.Tensor = None,
    class_weights_video: torch.Tensor = None,
    frame_weight: float = 1.0,
):
    B, S, Fr, C = frame_logits.shape
    num_frames = S * Fr

    frame_logits_flat = frame_logits.view(B * num_frames, C)
    frame_labels = labels.unsqueeze(1).expand(B, num_frames).reshape(B * num_frames)

    if class_weights_frame is not None:
        frame_criterion = nn.CrossEntropyLoss(weight=class_weights_frame, reduction='sum')
    else:
        frame_criterion = nn.CrossEntropyLoss(reduction='sum')

    frame_loss = frame_criterion(frame_logits_flat, frame_labels)
    frame_loss_avg = frame_loss / (B * num_frames)

    if class_weights_video is not None:
        video_criterion = nn.CrossEntropyLoss(weight=class_weights_video, reduction='mean')
    else:
        video_criterion = nn.CrossEntropyLoss(reduction='mean')

    video_loss = video_criterion(video_logits, labels)

    total_loss = frame_weight * frame_loss_avg + video_loss

    return total_loss, frame_loss_avg.item(), video_loss.item()
