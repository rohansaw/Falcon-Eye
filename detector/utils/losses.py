import numpy as np
import torch
import torch.nn.functional as F


def loss_bce_per_pixel(logits, y, pos_weight):
    batch, num_classes, height, width = y.shape

    device = y.device

    # Create a mask for the background class
    background_y = torch.zeros((num_classes,), device=device)
    background_y[0] = 1.0
    background_loss_mask = background_y[None, :, None, None].repeat(batch, 1, height, width)

    y = y.float()
    logits = logits.float()

    # Calculate the loss for the background
    background_loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    background_loss *= background_loss_mask

    # Calculate the loss for the non-background
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    non_background_loss = F.binary_cross_entropy_with_logits(
        logits,
        y,
        reduction="none",
        pos_weight=pos_weight_tensor,
    )

    # Ensure pos_weight is applied to all classes except the background
    non_background_loss *= (1 - background_loss_mask)

    # Return separate background and non-background losses
    return background_loss, non_background_loss

def loss_bce(logits, y, pos_weight):
    background_loss, non_background_loss = loss_bce_per_pixel(logits, y, pos_weight)
    return (background_loss + non_background_loss).mean()

def focal_loss_with_logits(logits, targets, alpha, gamma):
    """
    Compute focal loss between logits and binary labels, considering class weights.

    Parameters:
        logits (torch.Tensor): Logits from the model.
        targets (torch.Tensor): Ground truth labels, shape [batch, num_classes, height, width]
        alpha (float): Alpha factor to balance positive/negative examples.
        gamma (float): Focusing parameter to smooth the loss for well-classified examples.
        pos_weight (torch.Tensor): Tensor of positive weights per class.

    Returns:
        torch.Tensor: Computed focal loss.
    """
    # Generated with chat gpt
    
    probas = torch.sigmoid(logits)
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probas * targets + (1 - probas) * (1 - targets)
    loss = bce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss


def loss_focal_matrix(logits, y, pos_weight):
    """
    Computes the focal loss for all classes together, incorporating class weights.

    Parameters:
        logits (torch.Tensor): Logits from the model, shape [batch, num_classes, height, width]
        y (torch.Tensor): Ground truth labels, same shape as logits
        pos_weight (torch.Tensor or float): Weights for positive classes.

    Returns:
        float: Mean focal loss over all examples
    """
    y = y.float()
    logits = logits.float()

    # Focal loss parameters
    alpha = 0.25  # Adjust as needed
    gamma = 2  # Adjust as needed

    # Calculate focal loss for all elements
    total_loss = focal_loss_with_logits(logits, y, alpha, gamma)

    # Average the loss over all elements
    return total_loss

def loss_focal(logits, y, pos_weight):
    return loss_focal_matrix(logits, y, pos_weight).mean()

def loss_dist_focal(logits, y, pos_weight, distances):
    focal_loss = loss_focal_matrix(logits, y, pos_weight)
    loss_weighted = focal_loss * distances
    return loss_weighted.mean()

def loss_dist_bce(logits, y, pos_weight, distances):
    background_loss, non_background_loss = loss_bce_per_pixel(logits, y, pos_weight)
    loss_weighted = (background_loss  + non_background_loss) * distances
    return loss_weighted.mean()
