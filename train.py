from argparse import ArgumentParser
from configparser import ConfigParser

import torch
import torch.multiprocessing as mp

from model.data import HDF5Dataset, get_train_loaders
from model.nsrt import NSRT
from model.loss import Criterion
from model.step import Trainer, Validator

from utils import get_optimizer
from rich.progress import Progress, BarColumn, TextColumn

def main(rank, config):
    dataset = HDF5Dataset(
        file=config.get('dataset', 'file'),
        native_resolution=config.get('dataset', 'native-resolution'),
        target_resolution=config.get('dataset', 'target-resolution'),
        motion_threshold=config.getfloat('dataset', 'motion-mask-threshold'),
        patch_size=config.getint('training', 'patch-size'),
        chunk_size=config.getint('training', 'chunk-size'),
        spatial_stride=config.getint('training', 'spatial-stride'),
        temporal_stride=config.getint('training', 'temporal-stride'),
        train=True,
    )
    train_dataloader, val_dataloader = get_train_loaders(
        dataset=dataset, 
        batch_size=config.getint('training', 'batch-size'),
        n_workers=config.getint('training', 'num-workers'),
    )

    model = NSRT(
        frame_channels=config.getint('model', 'frame-channels'),
        context_length=config.getint('model', 'context-length'),
        upscale_factor=dataset.upsampling_factor,
        conv_features=config.getint('model', 'convo-features')
    )
    model.to(rank)

    optimizer, scheduler = get_optimizer(config, model)
    criterion = Criterion(rank)

    trainer = Trainer(rank, model, train_dataloader, optimizer, criterion)
    validator = Validator(rank, model, val_dataloader, criterion)

    # Keep training until the total number of epochs is reached
    n_epochs = config.getint('training', 'epochs')
    while scheduler.last_epoch < n_epochs:
        batch_size  = config.getint('training', 'batch-size')
        batch_count = (len(train_dataloader) + batch_size - 1) // batch_size

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            TextColumn(" [progress.percentage]{task.percentage:>3.0f}% "),
            BarColumn(),
            TextColumn(f" [{{task.completed}}/{{task.total}}] |"),
            TextColumn("train loss: {task.fields[loss]:.4f}"),
        ) as progress:
            task = progress.add_task(f"[cyan]Epoch {scheduler.last_epoch}/{n_epochs}", total=batch_count, loss=float('nan'))
            on_loss_update = lambda _, loss: progress.update(task, advance=1, loss=loss)
            trainer.step(on_loss_update)
            scheduler.step()
            validator.step()


if __name__ == "__main__":
    parser = ArgumentParser(description="Train the NSRT model")
    parser.add_argument(
        '--config', type=str, required=True, metavar='FILE',
        help="which file to read training configurations from")
    parser.add_argument(
        '--seed', type=int, default=0, 
        help="random seed for reproducibility")
    
    # Get training configurations
    args = parser.parse_args()
    config = ConfigParser()
    config.read(args.config)

    # Perform minor config verification
    if config.getint('model', 'context-length') - 1 > config.getint('training', 'chunk-size'):
        print("[!] the chunk-size is less than context-length, leading to redundant computations")
        exit(1)

    # world_size = torch.cuda.device_count()
    
    main("cuda", config)