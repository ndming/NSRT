from adabelief_pytorch import AdaBelief
from ranger_adabelief import RangerAdaBelief
from torch.optim import Optimizer, AdamW, SGD
from torch.optim.lr_scheduler import LRScheduler, StepLR, ExponentialLR

def get_optimizer(config, model) -> tuple[Optimizer, LRScheduler]:
    r"""Get the optimizer and learning rate scheduler based on the configuration. 
        
        The optimizer will be initialized with the model's parameters.
    """

    name = config.get('optimization', 'optimizer')
    rate = config.getfloat('optimization', 'learning-rate')  # can be 0, in which case we use default learning rate

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
        scheduler = ExponentialLR(optimizer, gamma=0.99)
    elif name == 'RangerAdaBelief':
        # Settings are the same as AdaBelief, but with the Ranger optimizer
        initial_lr = rate or 1e-3
        optimizer = RangerAdaBelief(model.parameters(), lr=initial_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, weight_decouple=True)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Options are: SGD, AdamW, AdaBelief, RangerAdaBelief")
    
    return optimizer, scheduler