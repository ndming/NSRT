from adabelief_pytorch import AdaBelief
from ranger_adabelief import RangerAdaBelief
from torch.optim import Optimizer, AdamW, SGD
from torch.optim.lr_scheduler import LRScheduler, StepLR, ExponentialLR

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict

import torch
import OpenEXR
import Imath
import random

def get_optimizer(name, rate, model) -> tuple[Optimizer, LRScheduler]:
    r"""Get the optimizer and learning rate scheduler based on the configuration. 
        The optimizer will be initialized with the model's parameters.
    """

    if name == 'SGD':
        # Learning rate is halved every 100 epochs, based on: https://doi.org/10.48550/arXiv.2308.06699
        initial_lr = rate or 5e-4
        optimizer = SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    elif name == 'AdamW':
        # Settings are the same as SGD, but with the AdamW optimizer
        initial_lr = rate or 5e-4
        optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=0)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    elif name == 'AdaBelief':
        # Based on parameters used for ImageNet training:
        # https://github.com/juntang-zhuang/Adabelief-Optimizer?tab=readme-ov-file#table-of-hyper-parameters
        # Exponential decay of learning rate, based on: https://doi.org/10.1145/3543870
        initial_lr = rate or 1e-3
        optimizer = AdaBelief(model.parameters(), lr=initial_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, weight_decouple=True)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif name == 'RangerAdaBelief':
        # Settings are the same as AdaBelief, but with the Ranger optimizer
        initial_lr = rate or 1e-3
        optimizer = RangerAdaBelief(model.parameters(), lr=initial_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, weight_decouple=True)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Options are: SGD, AdamW, AdaBelief, RangerAdaBelief")
    
    return optimizer, scheduler


def write_inference(logits, target, output_path):
    r"""Write the best results to an EXR file.
    """

    logits_diff = logits[:3, :, :].cpu().numpy().transpose(1, 2, 0)
    logits_spec = logits[3:, :, :].cpu().numpy().transpose(1, 2, 0)
    target_diff = target[:3, :, :].cpu().numpy().transpose(1, 2, 0)
    target_spec = target[3:, :, :].cpu().numpy().transpose(1, 2, 0)

    height, width, _ = logits_diff.shape
    header = OpenEXR.Header(width, height)

    logits_name = "Logits"
    target_name = "Target"

    header['channels'] = {
        f"{logits_name}.Diff.B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        f"{logits_name}.Diff.G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        f"{logits_name}.Diff.R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        f"{target_name}.Diff.B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        f"{target_name}.Diff.G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        f"{target_name}.Diff.R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        f"{logits_name}.Spec.B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        f"{logits_name}.Spec.G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        f"{logits_name}.Spec.R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        f"{target_name}.Spec.B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        f"{target_name}.Spec.G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        f"{target_name}.Spec.R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exr_file = OpenEXR.OutputFile(str(output_path), header)
    exr_file.writePixels({
        f"{logits_name}.Diff.B": logits_diff[:, :, 2].tobytes(),
        f"{logits_name}.Diff.G": logits_diff[:, :, 1].tobytes(),
        f"{logits_name}.Diff.R": logits_diff[:, :, 0].tobytes(),
        f"{target_name}.Diff.B": target_diff[:, :, 2].tobytes(),
        f"{target_name}.Diff.G": target_diff[:, :, 1].tobytes(),
        f"{target_name}.Diff.R": target_diff[:, :, 0].tobytes(),
        f"{logits_name}.Spec.B": logits_spec[:, :, 2].tobytes(),
        f"{logits_name}.Spec.G": logits_spec[:, :, 1].tobytes(),
        f"{logits_name}.Spec.R": logits_spec[:, :, 0].tobytes(),
        f"{target_name}.Spec.B": target_spec[:, :, 2].tobytes(),
        f"{target_name}.Spec.G": target_spec[:, :, 1].tobytes(),
        f"{target_name}.Spec.R": target_spec[:, :, 0].tobytes(),
    })


def unpack_model_state(state_dict):
    unpacked_state = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]   # remove 'module.' of DataParallel/DistributedDataParallel
        unpacked_state[name] = v
    return unpacked_state


def write_checkpoint(model, optimizer, scheduler, avg_train_loss, avg_loss, avg_ssim, avg_psnr, train_indices, val_indices, output_path):
    state = {
        'model_latest': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_indices': train_indices,
        'val_indices': val_indices,
        'cpu_checkpoint': not isinstance(model, (DDP, DataParallel)),
        'ddp_checkpoint': isinstance(model, DDP),
        'optim_name': get_optim_name(optimizer)
    }

    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        state['model_best'] = model.state_dict()
        state['val_losses'] = [avg_loss]
        state['train_losses'] = [avg_train_loss]
        state['avg_ssims'] = [avg_ssim]
        state['avg_psnrs'] = [avg_psnr]
    else:
        checkpoint = torch.load(output_path, weights_only=True)
        val_losses = checkpoint['val_losses']
        if avg_loss < min(val_losses):
            state['model_best'] = model.state_dict()
        state['val_losses'] = val_losses + [avg_loss]
        state['train_losses'] = checkpoint['train_losses'] + [avg_train_loss]
        state['avg_ssims'] = checkpoint['avg_ssims'] + [avg_ssim]
        state['avg_psnrs'] = checkpoint['avg_psnrs'] + [avg_psnr]

    torch.save(state, output_path)

def get_optim_name(optimizer):
    if isinstance(optimizer, SGD):
        return 'SGD'
    elif isinstance(optimizer, AdamW):
        return 'AdamW'
    elif isinstance(optimizer, AdaBelief):
        return 'AdaBelief'
    elif isinstance(optimizer, RangerAdaBelief):
        return 'RangerAdaBelief'
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}. Options are: SGD, AdamW, AdaBelief, RangerAdaBelief")

def gen_id(length=6):
    # Generate a random hexadecimal sequence of the specified length
    random_hex = ''.join(random.choices('0123456789abcdef', k=length))
    return random_hex
