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
    frame_loss_mode: str = "all_frame",
    mil_select_level: str = "segment",
    mil_pos_topk_segments: int = 2,
    mil_malignant_class_idx: int = 1,
    mil_selection_detach: bool = True,
):
    B, S, Fr, C = frame_logits.shape
    num_frames = S * Fr
    labels = labels.to(frame_logits.device)

    if class_weights_frame is not None:
        frame_criterion = nn.CrossEntropyLoss(weight=class_weights_frame, reduction='sum')
    else:
        frame_criterion = nn.CrossEntropyLoss(reduction='sum')

    if frame_loss_mode == "all_frame":
        frame_logits_flat = frame_logits.view(B * num_frames, C)
        frame_labels = labels.unsqueeze(1).expand(B, num_frames).reshape(B * num_frames)
        frame_loss = frame_criterion(frame_logits_flat, frame_labels)
        frame_loss_avg = frame_loss / (B * num_frames)
    elif frame_loss_mode == "mil_selective":
        if mil_select_level != "segment":
            raise ValueError(f"Unsupported mil_select_level: {mil_select_level}")

        target_frame_labels = labels.view(B, 1, 1).expand(B, S, Fr)
        selected_frame_mask = torch.zeros((B, S, Fr), dtype=torch.bool, device=frame_logits.device)

        neg_mask = labels == 0
        if neg_mask.any():
            selected_frame_mask[neg_mask] = True

        pos_mask = labels == 1
        if pos_mask.any():
            malignant_probs = F.softmax(frame_logits, dim=-1)[..., mil_malignant_class_idx]
            seg_risk = malignant_probs.mean(dim=-1)
            seg_risk_for_select = seg_risk.detach() if mil_selection_detach else seg_risk

            topk = max(1, min(int(mil_pos_topk_segments), S))
            pos_indices = torch.where(pos_mask)[0]
            for b_idx in pos_indices:
                topk_seg_idx = torch.topk(seg_risk_for_select[b_idx], k=topk, dim=0).indices
                selected_frame_mask[b_idx, topk_seg_idx, :] = True

        selected_count = selected_frame_mask.sum()
        if selected_count.item() > 0:
            selected_logits = frame_logits[selected_frame_mask]
            selected_targets = target_frame_labels[selected_frame_mask]
            frame_loss = frame_criterion(selected_logits, selected_targets)
            frame_loss_avg = frame_loss / selected_count.to(frame_loss.dtype)
        else:
            frame_loss_avg = video_logits.new_tensor(0.0)
    else:
        raise ValueError(f"Unsupported frame_loss_mode: {frame_loss_mode}")

    if class_weights_video is not None:
        video_criterion = nn.CrossEntropyLoss(weight=class_weights_video, reduction='mean')
    else:
        video_criterion = nn.CrossEntropyLoss(reduction='mean')

    video_loss = video_criterion(video_logits, labels)

    total_loss = frame_weight * frame_loss_avg + video_loss

    return total_loss, frame_loss_avg.detach(), video_loss.detach()
