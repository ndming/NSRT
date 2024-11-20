import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms

class Criterion(nn.Module):
    r"""The loss criterion for the NSRT model."""

    def __init__(self):
        r"""See [Temporally Stable Real-Time Joint Neural Denoising and Supersampling](https://doi.org/10.1145/3543870) 
            for more details on the loss function.
        """

        super().__init__()

        # Weights for each task
        self.w_diff = 1.0
        self.w_spec = 1.0

        self.spato_loss = PerceptualLossVGG16()
        self.tempo_loss = TemporalGradientLoss()

    def forward(self, logits, target, warped_logits, warped_target, w_spatial=0.2):
        logits_diff, logits_spec = torch.split(logits, [3, 3], dim=1)
        target_diff, target_spec = torch.split(target, [3, 3], dim=1)
        warped_logits_diff, warped_logits_spec = torch.split(warped_logits, [3, 3], dim=1)
        warped_target_diff, warped_target_spec = torch.split(warped_target, [3, 3], dim=1)

        loss_diff = self._compute_task_loss(logits_diff, target_diff, warped_logits_diff, warped_target_diff, w_spatial)
        loss_spec = self._compute_task_loss(logits_spec, target_spec, warped_logits_spec, warped_target_spec, w_spatial)

        # The total loss is the weighted sum of the spatial and temporal losses for each task
        # See https://doi.org/10.1145/3543870, Equation (3) for more details
        return self.w_diff * loss_diff + self.w_spec * loss_spec
    
    def _compute_task_loss(self, logits, target, warped_logits, warped_target, w_spatial):
        # Each task loss is a weighted sum of the spatial and temporal losses
        loss_spato = self.spato_loss(logits, target)
        loss_tempo = self.tempo_loss(logits, target, warped_logits, warped_target)
        return w_spatial * loss_spato + (1 - w_spatial) * loss_tempo

    
class PerceptualLossVGG16(nn.Module):
    r"""Perceptual loss implementation, based on 
        [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://doi.org/10.48550/arXiv.1603.08155).
    """

    def __init__(self, depth=16):
        r"""This is the drop-in replacement for the spatial loss, which is based on the VGG16 network.
        Here we're only interested in feature loss, not style transfer.

        See [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://doi.org/10.48550/arXiv.1603.08155) 
        for more details.
        """

        super().__init__()

        # For VGG16, input images are expected to be zero-centered with respect to ImageNet's dataset
        self.vgg_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Get the first `depth` layers of the VGG16 network
        self.vgg_extractor = models.vgg16(weights='IMAGENET1K_V1').features[:depth].eval()
        for param in self.vgg_extractor.parameters():
            param.requires_grad = False

    def forward(self, logits, target):
        # Preprocessing is based on Keras documentation: https://keras.io/api/applications/vgg/
        # Images are converted to BGR, but we don't resize the images to the default of 224x224
        y_pred = self.vgg_transform(logits[:, [2, 1, 0], :, :])  # (B, C, H, W)
        y_pred = self.vgg_extractor(y_pred)                      # (B, C, H, W)

        # We don't need gradients for the target
        with torch.no_grad():
            y_true = self.vgg_transform(target[:, [2, 1, 0], :, :])  # (B, C, H, W)
            y_true = self.vgg_extractor(y_true)                      # (B, C, H, W)

        # The original paper use MSE, but we find that L1-norm works better in practice
        return F.l1_loss(y_pred, y_true)
    

class TemporalGradientLoss(nn.Module):
    def __init__(self):
        r"""Temporal loss implementation, based on 
            [Temporally Stable Real-Time Joint Neural Denoising and Supersampling](https://doi.org/10.1145/3543870).
        """
        
        super().__init__()
    
    def forward(self, logits, target, warped_logits, warped_target):
        # The temporal loss is the L1-norm of the temporal gradient
        # See https://doi.org/10.1145/3543870, Equation (5) for more details
        y_pred = logits - warped_logits  # (B, C, H, W)
        y_true = target - warped_target  # (B, C, H, W)
        return F.l1_loss(y_pred, y_true)
    