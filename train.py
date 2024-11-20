from argparse import ArgumentParser
from configparser import ConfigParser

import torch
import torch.multiprocessing as mp

from model.data import HDF5Dataset, get_train_loaders
from model.nsrt import NSRT
from model.loss import Criterion
from model.step import Trainer, Validator

from utils import get_optimizer

def main(config):
    dataset = HDF5Dataset(
        file=config.get('dataset', 'file'),
        native_resolution=config.get('dataset', 'native-resolution'),
        target_resolution=config.get('dataset', 'target-resolution'),
        motion_threshold=config.getfloat('dataset', 'motion-threshold'),
        patch_size=config.getint('training', 'patch-size'),
        chunk_size=config.getint('training', 'chunk-size'),
        spatial_stride=config.getint('training', 'spatial-stride'),
        temporal_stride=config.getint('training', 'temporal-stride'),
        train=True,
    )
    train_dataloader, val_dataloader = get_train_loaders(
        dataset=dataset, 
        batch_size= config.getint('training', 'batch_size'),
        num_workers=config.getint('training', 'num_workers'),
    )

    model = NSRT(
        frame_channels=config.getint('model', 'frame-channels'),
        context_length=config.getint('model', 'context-length'),
        upscale_factor=dataset.upsampling_factor,
        conv_features=config.getint('model', 'convo-features')
    )

    optimizer, scheduler = get_optimizer(config, model)
    criterion = Criterion()

    trainer = Trainer(model, train_dataloader, optimizer, criterion)
    validator = Validator(model, val_dataloader, criterion)

    # Keep training until the total number of epochs is reached
    n_epochs = config.getint('training', 'epochs')
    while scheduler.last_epoch < n_epochs:
        trainer.step()
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
    if config.getint('model', 'context-length') - 1 > config.getint('dataset', 'chunk-size'):
        print("[!] the chunk-size is less than context-length, leading to redundant computations")
        exit(1)

    world_size = torch.cuda.device_count()
    
    main(config)