import torch
import torch.nn.functional as F

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

class Trainer:
    def __init__(self, rank, model, dataloader, optimizer, criterion):
        self.rank  = rank  # determine the device to use in multi-GPU training
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion

        self.context_length = model.module.context_length if isinstance(model, (DDP, DataParallel)) else model.context_length
        self.frame_channels = model.module.frame_channels if isinstance(model, (DDP, DataParallel)) else model.frame_channels

    def step(self, on_loss_update=None):
        self.model.train()

        batch_losses = []

        # Get all chunks in the batch
        for batch, (native, target) in enumerate(self.dataloader):
            self.optimizer.zero_grad()

            native = native.to(self.rank)  # (B, chunk_size, 14, H, W)
            target = target.to(self.rank)  # (B, chunk_size,  8, H * factor, W * factor)
            factor = self.dataloader.dataset.upsampling_factor

            chunk_size = self.dataloader.dataset.chunk_size
            prev_count = self.context_length - 1

            logits = None      # the previously reconstructed diffuse + specular  (B, 6, H * factor, W * factor)
            lstm_state = None  # the hidden and cell states of the ConvLSTM from the previous frame
            losses = []        # the accumulated loss for the current chunk

            # Feed consecutive patches in the chunk to the model and accumulate the loss
            for frame_idx in range(chunk_size):
                # Extract native motion vectors of the current frame
                curr2prev = native[:, frame_idx, -4:-2, ...]  # (B, 2, H, W)

                # Extract target light components of the current frame
                y_target = target[:, frame_idx, :6, ...]      # (B, 6, H * factor, W * factor) 

                # Warp the previously reconstructed logits to the current frame with the upsampled motion vectors
                # We do this step first since curr2prev will change when back warping previous frames
                if logits is None:
                    # We're in the first frame, set warped logits and target to zeros
                    warped_logits = torch.zeros_like(y_target)   # (B, 6, H * factor, W * factor)
                    warped_target = torch.zeros_like(y_target)   # (B, 6, H * factor, W * factor)
                else:
                    logits = logits.detach()  # detach the previous logits to prevent backpropagation
                    scaled_curr2prev = F.interpolate(curr2prev, scale_factor=factor, mode='bilinear', align_corners=True)
                    target_curr2prev = target[:, frame_idx, -2:, ...]         # (B, 2, H * factor, W * factor)
                    warped_logits = warp_frame(logits, scaled_curr2prev)      # (B, 6, H * factor, W * factor)
                    warped_target = warp_frame(target[:, frame_idx - 1, :6, ...], target_curr2prev, y_target)

                # Extract the current frame light and buffer components
                x_curr = native[:, frame_idx, :self.frame_channels, ...]      # (B, 10, H, W)
                b, _, h, w = x_curr.shape

                # The final version of the first input we will send to NSRT
                x = tonemap(x_curr[:, :6, ...])                # (B,  6, H, W)
                x = torch.cat([x, x_curr[:, 6:, ...]], dim=1)  # (B, 10, H, W)

                # Warp all previous frames in the context length to the current frame and concatenate them to x
                # We also collect the motion masks and accummulate relative motions along the way
                masks = []
                prev2curr = torch.zeros_like(curr2prev)        # (B, 2, H, W)
                for i in range(prev_count):
                    if frame_idx - i - 1 < 0:
                        warped_prev = torch.zeros_like(x_curr)
                        masks.append(torch.zeros((b, 1, h, w), device=x.device, dtype=torch.float32))
                    else:
                        # Frame backward warping
                        prev = native[:, frame_idx - i - 1, :self.frame_channels, ...]  # (B, 10, H, W)
                        prev_tonemapped = tonemap(prev[:, :6, ...])
                        prev_tonemapped = torch.cat([prev_tonemapped, prev[:, 6:, ...]], dim=1)  # (B, 10, H, W)
                        warped_prev = warp_frame(prev_tonemapped, curr2prev, x_curr)
                        # Motion masks and relative motions
                        prev2curr = prev2curr + native[:, frame_idx - i - 1, -2:, ...]      # (B, 2, H, W)
                        threshold = self.dataloader.dataset.motion_threshold
                        masks.append(compute_motion_mask(curr2prev, prev2curr, threshold))  # append (B, 1, H, W)
                        curr2prev = curr2prev + native[:, frame_idx - i - 1, -4:-2, ...]

                    x = torch.cat([x, warped_prev], dim=1)
                
                # Concatenate all motion masks
                motion_masks = torch.cat(masks, dim=1)

                # Tonemap target light components
                y_target = tonemap(y_target)            # (B, 6, H * factor, W * factor)
                warped_target = tonemap(warped_target)  # (B, 6, H * factor, W * factor)

                # Forward pass
                logits, lstm_state = self.model(x, motion_masks, warped_logits, lstm_state)
                lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())

                # Accumulate the loss
                w_spatial = 0.2 if frame_idx >= prev_count else 1.0
                losses.append(self.criterion(logits, y_target, warped_logits, warped_target, w_spatial))

            loss = sum(losses) / chunk_size
            loss.backward()
            self.optimizer.step()

            if on_loss_update is not None:
                on_loss_update(batch, loss.item())
            
            batch_losses.append(loss.item())
        
        return sum(batch_losses) / len(batch_losses)
        

class Validator:
    def __init__(self, rank, model, dataloader, criterion):
        self.rank  = rank  # determine the device to use in multi-GPU training
        self.model = model
        self.dataloader = dataloader
        self.criterion  = criterion

        self.context_length = model.module.context_length if isinstance(model, (DDP, DataParallel)) else model.context_length
        self.frame_channels = model.module.frame_channels if isinstance(model, (DDP, DataParallel)) else model.frame_channels

    def step(self, on_metrics_update=None):
        self.model.eval()

        batch_losses = []
        batch_ssims  = []
        batch_psnrs  = []

        best_logits = None
        best_target = None

        with torch.no_grad():
            for batch, (native, target) in enumerate(self.dataloader):
                native = native.to(self.rank)
                target = target.to(self.rank)
                factor = self.dataloader.dataset.upsampling_factor
    
                chunk_size = self.dataloader.dataset.chunk_size
                prev_count = self.context_length - 1
    
                logits = None
                lstm_state = None

                chunk_losses = []
                chunk_ssims  = []
                chunk_psnrs  = []

                for frame_idx in range(chunk_size):
                    curr2prev = native[:, frame_idx, -4:-2, ...]
                    y_target = target[:, frame_idx, :6, ...]

                    if logits is None:
                        warped_logits = torch.zeros_like(y_target)
                        warped_target = torch.zeros_like(y_target)
                    else:
                        scaled_curr2prev = F.interpolate(curr2prev, scale_factor=factor, mode='bilinear', align_corners=True)
                        target_curr2prev = target[:, frame_idx, -2:, ...]
                        warped_logits = warp_frame(logits, scaled_curr2prev)
                        warped_target = warp_frame(target[:, frame_idx - 1, :-2, ...], target_curr2prev, y_target)
                    
                    x_curr = native[:, frame_idx, :self.frame_channels, ...]
                    b, _, h, w = x_curr.shape

                    x = tonemap(x_curr[:, :6, ...])
                    x = torch.cat([x, x_curr[:, 6:, ...]], dim=1)

                    masks = []
                    prev2curr = torch.zeros_like(curr2prev)
                    for i in range(prev_count):
                        if frame_idx - i - 1 < 0:
                            warped_prev = torch.zeros_like(x_curr)
                            masks.append(torch.zeros((b, 1, h, w), device=x.device, dtype=torch.float32))
                        else:
                            prev = native[:, frame_idx - i - 1, :self.frame_channels, ...]
                            prev_tonemapped = tonemap(prev[:, :6, ...])
                            prev_tonemapped = torch.cat([prev_tonemapped, prev[:, 6:, ...]], dim=1)
                            warped_prev = warp_frame(prev_tonemapped, curr2prev, x_curr)
                            prev2curr = prev2curr + native[:, frame_idx - i - 1, -2:, ...]
                            threshold = self.dataloader.dataset.motion_threshold
                            masks.append(compute_motion_mask(curr2prev, prev2curr, threshold))
                            curr2prev = curr2prev + native[:, frame_idx - i - 1, -4:-2, ...]

                        x = torch.cat([x, warped_prev], dim=1)
                    
                    motion_masks = torch.cat(masks, dim=1)
                    y_target = tonemap(y_target)
                    warped_target = tonemap(warped_target)

                    logits, lstm_state = self.model(x, motion_masks, warped_logits, lstm_state)
                    lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())

                    # Val loss
                    w_spatial = 0.2 if frame_idx >= prev_count else 1.0
                    chunk_losses.append(self.criterion(logits, y_target, warped_logits, warped_target, w_spatial).item())

                    # SSIM
                    ssim = StructuralSimilarityIndexMeasure()
                    chunk_ssims.append(ssim(logits, y_target))

                    # PSNR
                    psnr = PeakSignalNoiseRatio()
                    chunk_psnrs.append(psnr(logits, y_target))

                
                loss = sum(chunk_losses) / chunk_size
                ssim = sum(chunk_ssims)  / chunk_size
                psnr = sum(chunk_psnrs)  / chunk_size

                if on_metrics_update is not None:
                    on_metrics_update(batch, loss, ssim, psnr)

                if best_logits is None or loss < min(batch_losses):
                    best_logits = logits[chunk_size // 2, ...]
                    best_target = y_target[chunk_size // 2, ...]

                batch_losses.append(loss)
                batch_ssims.append(ssim)
                batch_psnrs.append(psnr)
        
        avg_loss = sum(batch_losses) / len(batch_losses)
        avg_ssim = sum(batch_ssims)  / len(batch_ssims)
        avg_psnr = sum(batch_psnrs)  / len(batch_psnrs)
        return avg_loss, avg_ssim, avg_psnr, best_logits, best_target



def warp_frame(prev, vector, curr=None):
    r"""Warp the previous frame to the current frame using the motion vector. 
        Vectors map pixels in the current frame to the previous frame.

        If the current frame is provided, the out-of-bounds pixels are replaced with the current frame.
    """

    b, _, h, w = prev.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=prev.device), 
        torch.arange(w, device=prev.device), 
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)
    grid = torch.cat([grid_x, grid_y], dim=1).float()  # (B, 2, H, W)
    next = grid + vector  # the pixel indices to the prev tensor that would give pixel values in the curr tensor
    next = 2.0 * next / torch.tensor([w, h], device=next.device).view(1, 2, 1, 1) - 1.0  # normalize to [-1, 1]

    # Permute grid to match grid_sample input format (batch, H, W, 2)
    next = next.permute(0, 2, 3, 1)
    warp = F.grid_sample(prev, next, align_corners=True, mode='bicubic')  # (B, C, H, W)

    if curr is None:
        return warp
    
    out_of_bounds = ((next < -1) | (next > 1)).any(dim=3, keepdim=True)   # (B, H, W, 1)
    out_of_bounds = out_of_bounds.permute(0, 3, 1, 2)                     # (B, 1, H, W)

    return torch.where(out_of_bounds, curr, warp)

def compute_motion_mask(curr2prev, prev2curr, threshold):
    # See https://zheng95z.github.io/publications/rtg221
    b, _, h, w = curr2prev.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=curr2prev.device), 
        torch.arange(w, device=curr2prev.device), 
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)

    x_curr = torch.cat([grid_x, grid_y], dim=1).float()  # (B, 2, H, W)
    y = x_curr + curr2prev
    z = y + prev2curr
    x_prev = x_curr - z + y
    curr2prev_dual = x_prev - x_curr

    mask = curr2prev_dual - curr2prev
    mask = torch.abs(mask) >= threshold
    return torch.any(mask, dim=1, keepdim=True).float()


def tonemap(x, gamma=2.2):
    # See Equation (1) in https://doi.org/10.1145/3543870
    x = torch.clamp(x, 0, 65535)
    x = torch.log(x + 1).pow(1 / gamma)
    return x